"""Key driver analysis (KDA) framework.

For each stakeholder group the framework:

1. computes each driver's **importance** as its correlation with the outcome
   (pairwise complete observations, so "don't know" answers do not drop a respondent),
2. computes each driver's **performance** as its mean rating,
3. places every driver in a 4-quadrant importance / performance grid using a fixed
   importance threshold (|r| >= 0.2 by default) and a performance threshold
   (the group's average rating across all drivers by default),
4. labels the quadrants, draws the chart, and
5. writes structured, testable hypothesis statements.

Quadrants
---------
=================  ==============  =================
                   low performance high performance
=================  ==============  =================
high importance    Priority fixes  Strengths
low importance     Low priority    Maintain
=================  ==============  =================

Typical use::

    from kda import run_kda
    results = run_kda(df, group_col="stakeholder_group", drivers=attrs, outcome="outcome_score")
    results["Physicians"].table, results["Physicians"].hypotheses
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

IMPORTANCE_THRESHOLD = 0.2

QUADRANTS = {
    (True, True): "Strengths",
    (True, False): "Priority fixes",
    (False, True): "Maintain",
    (False, False): "Low priority",
}
QUADRANT_COLORS = {
    "Priority fixes": "#e34948",
    "Strengths": "#1baf7a",
    "Maintain": "#2a78d6",
    "Low priority": "#9a9993",
}
QUADRANT_ORDER = ["Priority fixes", "Strengths", "Maintain", "Low priority"]


# --------------------------------------------------------------------------- core metrics
def driver_table(df: pd.DataFrame, drivers: list[str], outcome: str,
                 importance_threshold: float = IMPORTANCE_THRESHOLD,
                 performance_threshold: float | None = None,
                 method: str = "pearson") -> pd.DataFrame:
    """Importance (correlation with outcome), performance (mean rating) and quadrant per driver."""
    rows = []
    corr_fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    for d in drivers:
        pair = df[[d, outcome]].dropna()
        r, p = corr_fn(pair[d], pair[outcome])
        rows.append({"driver": d, "importance": float(r), "p_value": float(p), "n": len(pair),
                     "performance": pair[d].mean()})
    t = pd.DataFrame(rows).set_index("driver")
    perf_thr = t["performance"].mean() if performance_threshold is None else performance_threshold
    t["quadrant"] = [classify_quadrant(i, p, importance_threshold, perf_thr)
                     for i, p in zip(t["importance"], t["performance"])]
    t["importance_rank"] = t["importance"].rank(ascending=False).astype(int)
    t["performance_gap"] = t["performance"] - perf_thr
    t.attrs.update(importance_threshold=importance_threshold, performance_threshold=perf_thr)
    return t.sort_values("importance", ascending=False)


def classify_quadrant(importance: float, performance: float,
                      importance_threshold: float, performance_threshold: float) -> str:
    return QUADRANTS[(importance >= importance_threshold, performance >= performance_threshold)]


def relative_weights(df: pd.DataFrame, drivers: list[str], outcome: str) -> pd.Series:
    """Johnson's (2000) relative weights: share of R^2 attributable to each driver.

    Handles correlated drivers by projecting them onto their closest orthogonal
    counterparts. Missing ratings are mean-imputed within the frame. The weights sum
    to the model R^2 (stored in ``.attrs['r2']``).
    """
    X = df[drivers].astype(float)
    X = X.fillna(X.mean())
    y = df[outcome].astype(float)
    Rxx = np.corrcoef(X.to_numpy(), rowvar=False)
    rxy = np.array([np.corrcoef(X[c], y)[0, 1] for c in drivers])
    evals, evecs = np.linalg.eigh(Rxx)
    lam = evecs @ np.diag(np.sqrt(np.clip(evals, 0, None))) @ evecs.T   # Rxx^(1/2)
    beta = np.linalg.solve(lam, rxy)                                   # regress y on orthogonal Z
    raw = (lam ** 2) @ (beta ** 2)
    out = pd.Series(raw, index=drivers, name="relative_weight")
    out.attrs["r2"] = float((beta ** 2).sum())
    return out


def bootstrap_importance(df: pd.DataFrame, drivers: list[str], outcome: str, n_boot: int = 500,
                         threshold: float = IMPORTANCE_THRESHOLD, seed: int = 0) -> pd.DataFrame:
    """Bootstrap 95% CI for each driver's correlation and how often it clears the threshold."""
    rng = np.random.default_rng(seed)
    data = df[drivers + [outcome]].to_numpy(float)
    n = len(data)
    draws = np.empty((n_boot, len(drivers)))
    for b in range(n_boot):
        sample = pd.DataFrame(data[rng.integers(0, n, n)], columns=drivers + [outcome])
        draws[b] = sample[drivers].corrwith(sample[outcome]).to_numpy()
    return pd.DataFrame({
        "ci_low": np.percentile(draws, 2.5, axis=0),
        "ci_high": np.percentile(draws, 97.5, axis=0),
        "share_above_threshold": (draws >= threshold).mean(axis=0),
    }, index=drivers)


# --------------------------------------------------------------------------- chart
def plot_quadrant(table: pd.DataFrame, title: str, labels: dict | None = None, ax=None,
                  savepath: str | Path | None = None, annotate: str = "important"):
    """Importance (x) vs performance (y) chart with the threshold lines and labelled quadrants.

    ``annotate``: "important" labels drivers at or above the importance threshold, "all" labels every driver.
    """
    imp_thr = table.attrs["importance_threshold"]
    perf_thr = table.attrs["performance_threshold"]
    labels = labels or {}
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 6.5))

    for q in QUADRANT_ORDER:
        sub = table[table["quadrant"] == q]
        ax.scatter(sub["importance"], sub["performance"], s=70, color=QUADRANT_COLORS[q],
                   edgecolor="white", linewidth=1.5, label=f"{q} ({len(sub)})", zorder=3)
    ax.axvline(imp_thr, color="#52514e", lw=1.2, ls="--", zorder=2)
    ax.axhline(perf_thr, color="#52514e", lw=1.2, ls="--", zorder=2)

    x_lo = min(-0.05, table["importance"].min() - 0.05)
    x_hi = max(imp_thr + 0.2, table["importance"].max() + 0.08)
    y_pad = 0.35
    y_lo, y_hi = table["performance"].min() - y_pad, table["performance"].max() + y_pad
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    corners = {"Strengths": (x_hi, y_hi, "right", "top"), "Priority fixes": (x_hi, y_lo, "right", "bottom"),
               "Maintain": (x_lo, y_hi, "left", "top"), "Low priority": (x_lo, y_lo, "left", "bottom")}
    for q, (x, y, ha, va) in corners.items():
        ax.text(x, y, f" {q} ", ha=ha, va=va, fontsize=10, fontweight="bold", color=QUADRANT_COLORS[q], alpha=0.9)

    to_label = table if annotate == "all" else table[table["importance"] >= imp_thr]
    placed = []
    y_span = y_hi - y_lo
    for d, r in to_label.sort_values("performance", ascending=False).iterrows():
        x, y = r["importance"], r["performance"]
        crowded = any(abs(x - px) < 0.08 and abs(y - py) < 0.04 * y_span for px, py in placed)
        ax.annotate(labels.get(d, d), (x, y), xytext=(6, -12) if crowded else (6, 4),
                    textcoords="offset points", fontsize=8, color="#0b0b0b")
        placed.append((x, y))
    ax.set_xlabel(f"Importance: correlation with outcome (threshold r = {imp_thr:.1f})")
    ax.set_ylabel(f"Performance: mean rating, 1-10 (threshold = {perf_thr:.2f})")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    if savepath is not None:
        ax.figure.savefig(savepath)
    return ax


# --------------------------------------------------------------------------- hypotheses
TEMPLATES = {
    "Priority fixes": dict(
        hypothesis=("If {group} rate {label} higher, their {outcome} will increase: it is among the strongest "
                    "correlates of {outcome} (r = {r:.2f}) but is rated {gap:.1f} points below the group's "
                    "average attribute rating ({perf:.1f} / 10)."),
        test=("Field a targeted intervention on {label} to a test cell of {group}; compare the change in "
              "{outcome} and in the {label} rating against a matched control in the next wave."),
        action="Fix",
    ),
    "Strengths": dict(
        hypothesis=("{label} is a reason {group} support the brand (r = {r:.2f}, rated {perf:.1f} / 10); leading "
                    "with it in messaging will protect {outcome}, and any slippage in it would reduce {outcome}."),
        test="Track {label} as a guardrail metric each wave; A/B test messaging that leads with {label}.",
        action="Leverage",
    ),
    "Maintain": dict(
        hypothesis=("{label} is rated well by {group} ({perf:.1f} / 10) but is only weakly linked to {outcome} "
                    "(r = {r:.2f}); holding investment flat, or reallocating some of it, will not reduce {outcome}."),
        test="Reallocate a share of {label} spend to a priority fix and confirm {outcome} does not fall.",
        action="Hold",
    ),
    "Low priority": dict(
        hypothesis=("{label} has weak links to {outcome} among {group} (r = {r:.2f}); improving it is unlikely to "
                    "move {outcome} in the near term."),
        test="Monitor only; revisit if its correlation with {outcome} rises above the threshold in a later wave.",
        action="Monitor",
    ),
}


def generate_hypotheses(table: pd.DataFrame, group: str, outcome_label: str,
                        labels: dict | None = None, max_per_quadrant: dict | None = None) -> pd.DataFrame:
    """Structured hypothesis statements for one stakeholder group.

    Each row has: group, quadrant, driver, a falsifiable hypothesis, the evidence behind it, a proposed test,
    a recommended action and a priority score (importance x size of the performance gap for fixes,
    importance for strengths). ``max_per_quadrant`` caps how many hypotheses each quadrant contributes.
    """
    labels = labels or {}
    caps = {"Priority fixes": None, "Strengths": None, "Maintain": 2, "Low priority": 1}
    caps.update(max_per_quadrant or {})
    rows = []
    for q in QUADRANT_ORDER:
        sub = table[table["quadrant"] == q].copy()
        if q == "Priority fixes":
            sub["priority_score"] = sub["importance"] * (-sub["performance_gap"]).clip(lower=0)
        elif q == "Strengths":
            sub["priority_score"] = sub["importance"]
        else:
            sub["priority_score"] = 0.0
        order = sub.sort_values("performance", ascending=False) if q == "Maintain" else \
            sub.sort_values(["priority_score", "importance"], ascending=False)
        if caps[q] is not None:
            order = order.head(caps[q])
        for d, r in order.iterrows():
            fmt = dict(group=group.lower(), label=labels.get(d, d).lower(), outcome=outcome_label,
                       r=r["importance"], perf=r["performance"], gap=-r["performance_gap"])
            t = TEMPLATES[q]
            hyp = t["hypothesis"].format(**fmt)
            rows.append({
                "group": group, "quadrant": q, "driver": d, "driver_label": labels.get(d, d),
                "importance_r": round(r["importance"], 3), "performance": round(r["performance"], 2),
                "priority_score": round(r["priority_score"], 3), "action": t["action"],
                "hypothesis": hyp[0].upper() + hyp[1:], "proposed_test": t["test"].format(**fmt),
            })
    out = pd.DataFrame(rows)
    out.insert(0, "hypothesis_id", [f"{group[:3].upper()}-{i:02d}" for i in range(1, len(out) + 1)])
    return out


def cross_group_hypotheses(results: dict, labels: dict | None = None, min_groups: int = 2) -> pd.DataFrame:
    """Drivers that land in the same actionable quadrant for several stakeholder groups."""
    labels = labels or {}
    q = pd.DataFrame({g: r.table["quadrant"] for g, r in results.items()})
    imp = pd.DataFrame({g: r.table["importance"] for g, r in results.items()})
    rows = []
    claims = {
        "Priority fixes": ("a shared fix", "a single cross-functional initiative on it will lift the outcome in all of "
                                           "these groups at lower cost than separate group-level programmes."),
        "Strengths": ("a shared strength", "leading with it in common core messaging will sustain the outcome in all "
                                           "of these groups, and a drop in its rating would hit them together."),
    }
    for quadrant, (verb, claim) in claims.items():
        for d in q.index:
            groups = [g for g in q.columns if q.loc[d, g] == quadrant]
            if len(groups) >= min_groups:
                rs = ", ".join(f"{g} r = {imp.loc[d, g]:.2f}" for g in groups)
                rows.append({
                    "driver": d, "driver_label": labels.get(d, d), "quadrant": quadrant, "groups": ", ".join(groups),
                    "n_groups": len(groups),
                    "hypothesis": f"{labels.get(d, d)} is {verb} across {', '.join(groups)} ({rs}); {claim}",
                })
    return pd.DataFrame(rows).sort_values(["quadrant", "n_groups"], ascending=[True, False]).reset_index(drop=True) \
        if rows else pd.DataFrame(columns=["driver", "driver_label", "quadrant", "groups", "n_groups", "hypothesis"])


# --------------------------------------------------------------------------- orchestration
@dataclass
class KDAResult:
    group: str
    n_respondents: int
    table: pd.DataFrame
    hypotheses: pd.DataFrame
    seconds: float
    figure_path: Path | None = None
    timings: dict = field(default_factory=dict)


def run_kda(df: pd.DataFrame, group_col: str, drivers: list[str], outcome: str,
            outcome_labels: dict | None = None, labels: dict | None = None,
            importance_threshold: float = IMPORTANCE_THRESHOLD, fig_dir: str | Path | None = None,
            show: bool = False) -> dict[str, KDAResult]:
    """Run the full framework (metrics, quadrant chart, hypotheses) for every group and time each run."""
    outcome_labels = outcome_labels or {}
    results = {}
    for group, gdf in df.groupby(group_col, sort=False):
        t0 = time.perf_counter()
        table = driver_table(gdf, drivers, outcome, importance_threshold)
        t1 = time.perf_counter()
        path = None
        ax = plot_quadrant(table, f"{group}: key driver quadrant (n = {len(gdf):,})", labels=labels)
        if fig_dir is not None:
            path = Path(fig_dir) / f"02_quadrant_{group.lower()}.png"
            ax.figure.savefig(path)
        if show:
            plt.show()
        else:
            plt.close(ax.figure)
        t2 = time.perf_counter()
        hyps = generate_hypotheses(table, group, outcome_labels.get(group, outcome), labels)
        t3 = time.perf_counter()
        results[group] = KDAResult(group, len(gdf), table, hyps, t3 - t0, path,
                                   {"metrics_s": t1 - t0, "chart_s": t2 - t1, "hypotheses_s": t3 - t2})
    return results
