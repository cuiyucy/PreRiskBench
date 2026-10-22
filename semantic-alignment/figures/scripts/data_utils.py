from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from style import FIGURE_ROOT, PROJECT_ROOT


DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DIR = FIGURE_ROOT / "source_data"


def load_cohort() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_DIR / "cohort_summary.csv")
    expected = {"VitalDB", "MOVER", "INSPIRE"}
    if set(df["dataset"]) != expected:
        raise ValueError(f"Unexpected cohort resources: {sorted(df['dataset'].unique())}")
    return df


def load_mapping_review() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_DIR / "mapping_review.csv")
    if len(df) != 9:
        raise ValueError(f"Expected nine mapping-review strata, found {len(df)}")
    df["acceptable_rate"] = df["acceptable_n"] / df["review_n"]
    intervals = [wilson_interval(a, n) for a, n in zip(df["acceptable_n"], df["review_n"])]
    df["ci_low"] = [v[0] for v in intervals]
    df["ci_high"] = [v[1] for v in intervals]
    return df


def load_audit_summary() -> pd.DataFrame:
    return pd.read_csv(SOURCE_DIR / "audit_summary.csv")


def load_concept_space() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "analysis_concept_space.csv")
    if len(df) != 12:
        raise ValueError(f"Expected 12 concept-space rows, found {len(df)}")
    return df


def load_knn() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "analysis_knn_retrieval.csv")
    if len(df) != 18:
        raise ValueError(f"Expected 18 retrieval rows, found {len(df)}")
    if df[["mode", "direction"]].duplicated().any():
        raise ValueError("Duplicate retrieval mode-direction rows")
    return df


def load_mutual_info() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "analysis_mutual_info.csv")


def parse_mean_sd(value: object) -> tuple[float, float]:
    text = str(value).strip()
    if text.upper() == "NA" or text == "" or text.lower() == "nan":
        return math.nan, math.nan
    match = re.fullmatch(r"\s*([-+]?\d*\.?\d+)\s*[±\+/-]+\s*(\d*\.?\d+)\s*", text)
    if not match:
        raise ValueError(f"Cannot parse mean±SD value: {value!r}")
    return float(match.group(1)), float(match.group(2))


def load_model_readiness() -> pd.DataFrame:
    exp = pd.read_csv(DATA_DIR / "experiment_master.csv")
    keep = exp[
        (exp["task"] == "asa")
        & (exp["experiment_type"] == "single")
        & (exp["mode"].isin(["aligned", "clustered"]))
    ].copy()
    if len(keep) != 72:
        raise ValueError(f"Expected 72 model-readiness settings, found {len(keep)}")
    parsed = keep["Macro-F1"].map(parse_mean_sd)
    keep["macro_f1_mean"] = [x[0] for x in parsed]
    keep["macro_f1_sd"] = [x[1] for x in parsed]

    prior = pd.read_csv(DATA_DIR / "analysis_prior_baseline.csv")
    prior = prior[prior["task"] == "asa"][["dataset", "prior_baseline_mf1"]]
    keep = keep.merge(prior, on="dataset", how="left", validate="many_to_one")
    if keep["prior_baseline_mf1"].isna().any():
        raise ValueError("Missing ASA majority-class baseline after merge")
    keep["delta"] = keep["macro_f1_mean"] - keep["prior_baseline_mf1"]
    keep["baseline_result"] = np.select(
        [keep["delta"] > 5e-5, keep["delta"] < -5e-5],
        ["above", "below"],
        default="tie",
    )
    counts = keep["baseline_result"].value_counts().to_dict()
    expected = {"above": 62, "tie": 9, "below": 1}
    if counts != expected:
        raise ValueError(f"Model-readiness count changed: {counts}; expected {expected}")
    return keep


def load_joint_clinicalbert() -> pd.DataFrame:
    exp = pd.read_csv(DATA_DIR / "experiment_master.csv")
    keep = exp[
        (exp["task"] == "asa")
        & (exp["experiment_type"] == "joint")
        & (exp["encoding"] == "BERT")
        & (exp["mode"].isin(["unaligned", "aligned", "clustered"]))
    ].copy()
    if len(keep) != 18:
        raise ValueError(f"Expected 18 joint ClinicalBERT settings, found {len(keep)}")
    if keep[["mode", "algorithm"]].duplicated().any():
        raise ValueError("Duplicate joint ClinicalBERT mode-algorithm rows")
    parsed = keep["Macro-F1"].map(parse_mean_sd)
    keep["macro_f1_mean"] = [x[0] for x in parsed]
    keep["macro_f1_sd"] = [x[1] for x in parsed]
    field = keep[keep["mode"] == "unaligned"].set_index("algorithm")["macro_f1_mean"]
    aligned = keep[keep["mode"] == "aligned"].set_index("algorithm")["macro_f1_mean"]
    deltas = aligned - field
    if not (deltas > 0).all():
        raise ValueError(f"Expected positive Concept-Aligned deltas for all classifiers: {deltas.to_dict()}")
    return keep


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return centre - half, centre + half


def deterministic_offsets(n: int, width: float = 0.18) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-width, width, n)


def assert_source_totals() -> None:
    cohort = load_cohort()
    totals = cohort[["start_surgeries", "benchmark_surgeries", "final_asa_surgeries"]].sum()
    expected_totals = {
        "start_surgeries": 72807,
        "benchmark_surgeries": 58553,
        "final_asa_surgeries": 28397,
    }
    if totals.to_dict() != expected_totals:
        raise ValueError(f"Cohort totals changed: {totals.to_dict()}")

    mapping = load_mapping_review()
    if int(mapping["mapping_count"].sum()) != 47270:
        raise ValueError("Mapping count no longer totals 47,270")
    if int(mapping["acceptable_n"].sum()) != 170 or int(mapping["review_n"].sum()) != 242:
        raise ValueError("Mapping review total no longer equals 170/242")

    concept = load_concept_space()
    shared = concept[concept["pair"] == "三库共有"].set_index("mode")["intersection"]
    if [int(shared[m]) for m in ["unaligned", "aligned", "aligned_clustered"]] != [2, 64, 167]:
        raise ValueError("Three-resource shared-concept sequence changed")

    load_knn()
    load_model_readiness()
