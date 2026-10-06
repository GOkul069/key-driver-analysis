"""Synthetic multi-stakeholder perception survey generator.

Creates a deterministic survey of four stakeholder groups (physicians, pharmacists,
patients, payers) rating a fictional prescription therapy, "Velarex", from the
fictional manufacturer "Northbridge Therapeutics", on 24 attributes (1-10 scale),
plus a 0-10 outcome per respondent (prescribing intent for physicians, likelihood
to recommend for pharmacists and patients, likelihood to grant preferred
formulary status for payers).

Each group has its own set of true drivers and its own performance profile, and
every rating carries a shared "halo" component, as real survey data does. Around
3% of ratings are "don't know" (missing).

Usage
-----
    python src/generate_data.py            # writes data/raw/stakeholder_survey.csv
    python src/generate_data.py --scale 5  # 5x larger sample per group
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 314

ATTRIBUTES = {
    "clinical_efficacy": "Clinical efficacy",
    "speed_of_onset": "Speed of onset",
    "durability_of_response": "Durability of response",
    "safety_profile": "Safety profile",
    "tolerability": "Tolerability / side effects",
    "drug_interactions": "Low drug-interaction risk",
    "dosing_convenience": "Dosing convenience",
    "ease_of_administration": "Ease of administration",
    "formulary_coverage": "Formulary coverage",
    "out_of_pocket_cost": "Affordable out-of-pocket cost",
    "prior_auth_ease": "Ease of prior authorisation",
    "availability_in_stock": "Product availability / stock",
    "patient_support_program": "Patient support programme",
    "adherence_tools": "Adherence tools and reminders",
    "rep_knowledge": "Field rep knowledge",
    "medical_info_responsiveness": "Medical information responsiveness",
    "educational_materials": "Educational materials",
    "real_world_evidence": "Real-world evidence",
    "guideline_endorsement": "Guideline endorsement",
    "peer_recommendation": "Peer recommendation",
    "value_for_money": "Value for money",
    "cost_effectiveness_data": "Cost-effectiveness data",
    "company_reputation": "Company reputation",
    "digital_resources": "Digital resources",
}

OUTCOME = "outcome_score"
OUTCOME_LABELS = {
    "Physicians": "prescribing intent",
    "Pharmacists": "likelihood to recommend",
    "Patients": "likelihood to recommend",
    "Payers": "likelihood to grant preferred formulary status",
}

# n respondents, true driver weights (others get small random weights) and
# performance shifts (mean rating = group base + shift).
GROUPS = {
    "Physicians": dict(
        n=600, base=6.6,
        weights={"clinical_efficacy": 0.50, "safety_profile": 0.36, "guideline_endorsement": 0.28,
                 "real_world_evidence": 0.26, "prior_auth_ease": 0.30, "rep_knowledge": 0.20,
                 "digital_resources": 0.13},
        perf={"clinical_efficacy": 1.2, "safety_profile": 0.9, "guideline_endorsement": 0.8,
              "real_world_evidence": -1.4, "prior_auth_ease": -2.0, "rep_knowledge": -0.6,
              "company_reputation": 0.9, "educational_materials": 0.7, "digital_resources": -0.5},
    ),
    "Pharmacists": dict(
        n=400, base=6.4,
        weights={"availability_in_stock": 0.45, "drug_interactions": 0.38, "prior_auth_ease": 0.32,
                 "medical_info_responsiveness": 0.26, "formulary_coverage": 0.24, "rep_knowledge": 0.14},
        perf={"availability_in_stock": -1.6, "drug_interactions": 1.0, "prior_auth_ease": -1.8,
              "medical_info_responsiveness": 0.7, "formulary_coverage": -0.8,
              "company_reputation": 0.8, "rep_knowledge": 0.6, "digital_resources": 0.5},
    ),
    "Patients": dict(
        n=1200, base=6.5,
        weights={"out_of_pocket_cost": 0.48, "tolerability": 0.38, "dosing_convenience": 0.32,
                 "patient_support_program": 0.28, "speed_of_onset": 0.22, "ease_of_administration": 0.22,
                 "adherence_tools": 0.14},
        perf={"out_of_pocket_cost": -2.2, "tolerability": 0.6, "dosing_convenience": 1.1,
              "patient_support_program": -1.2, "speed_of_onset": -0.3, "ease_of_administration": 1.0,
              "educational_materials": 0.8, "digital_resources": 0.6, "company_reputation": 0.5},
    ),
    "Payers": dict(
        n=150, base=6.0,
        weights={"cost_effectiveness_data": 0.50, "real_world_evidence": 0.38, "value_for_money": 0.34,
                 "clinical_efficacy": 0.26, "durability_of_response": 0.22, "medical_info_responsiveness": 0.15},
        perf={"cost_effectiveness_data": -1.6, "real_world_evidence": -1.3, "value_for_money": -0.9,
              "clinical_efficacy": 1.3, "durability_of_response": 0.8,
              "company_reputation": 0.9, "medical_info_responsiveness": 0.6, "guideline_endorsement": 0.9},
    ),
}

HALO = 0.30          # loading of every attribute rating on a respondent-level halo factor
OUTCOME_HALO = 0.30  # loading of the outcome on the same halo factor
RATING_SD = 1.6      # spread of ratings on the 1-10 scale
MISSING_RATE = 0.03  # "don't know" answers
REGIONS = ["Region A", "Region B", "Region C", "Region D"]


def _group_block(rng: np.random.Generator, name: str, cfg: dict, scale: int) -> pd.DataFrame:
    n = cfg["n"] * scale
    attrs = list(ATTRIBUTES)
    k = len(attrs)
    halo = rng.normal(size=n)
    unique = rng.normal(size=(n, k))
    z = HALO * halo[:, None] + np.sqrt(1 - HALO ** 2) * unique

    w = rng.uniform(0.0, 0.05, k)  # small background effects
    for a, v in cfg["weights"].items():
        w[attrs.index(a)] = v
    latent = unique @ w + OUTCOME_HALO * halo + rng.normal(0, 0.75, n)
    latent = (latent - latent.mean()) / latent.std()
    outcome = np.clip(np.rint(6.3 + 2.0 * latent), 0, 10).astype(int)

    mu = np.full(k, cfg["base"]) + rng.normal(0, 0.25, k)
    for a, v in cfg["perf"].items():
        mu[attrs.index(a)] = cfg["base"] + v
    ratings = np.clip(np.rint(mu + RATING_SD * z), 1, 10)
    ratings[rng.random((n, k)) < MISSING_RATE] = np.nan

    df = pd.DataFrame(ratings, columns=attrs)
    df.insert(0, "region", rng.choice(REGIONS, n))
    df.insert(0, "stakeholder_group", name)
    df[OUTCOME] = outcome
    return df


def generate(scale: int = 1, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.concat([_group_block(rng, g, cfg, scale) for g, cfg in GROUPS.items()], ignore_index=True)
    df.insert(0, "respondent_id", [f"R{i:05d}" for i in range(1, len(df) + 1)])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", type=int, default=1, help="multiply each group's sample size")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = args.out or root / "data" / "raw" / "stakeholder_survey.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.scale, args.seed)
    df.to_csv(out, index=False)
    pd.DataFrame({"attribute": list(ATTRIBUTES), "label": list(ATTRIBUTES.values())}).to_csv(
        out.parent / "attribute_dictionary.csv", index=False)
    print(f"Wrote {len(df):,} respondents x {df.shape[1]} cols to {out}")


if __name__ == "__main__":
    main()
