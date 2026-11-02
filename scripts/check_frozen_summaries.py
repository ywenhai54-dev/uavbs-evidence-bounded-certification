"""Lightweight integrity checks for the compact released summaries."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"expected {expected}, found {actual}")


def main() -> None:
    methods = pd.read_csv(RESULTS / "formal_h6_confirmatory_method_summary.csv")
    indexed = methods.set_index("method")
    close(float(indexed.loc["BestStrong", "APP_LB_mean"]), 0.26020066711157147)
    close(float(indexed.loc["CSD-MR", "APP_LB_mean"]), 0.4622427024363857)

    paired = pd.read_csv(RESULTS / "formal_h6_paired_csd_mr_vs_beststrong.csv")
    if len(paired) != 48:
        raise AssertionError(f"expected 48 paired cases, found {len(paired)}")
    close(float(paired["paired_gain"].mean()), 0.20204203532481424)
    if not (paired["paired_gain"] > 0).all():
        raise AssertionError("all 48 formal H6 paired gains must be positive")

    replay = pd.read_csv(RESULTS / "ctks_replay_paired_summary.csv")
    row = replay.set_index("comparison").loc["ctks_vs_threshold_only"]
    if int(row["certifications"]) != 1692:
        raise AssertionError("expected 1692 paired replay certifications")
    close(float(row["decision_agreement_rate"]), 0.991725768321513)

    gaps = pd.read_csv(RESULTS / "formal_h6_final_certification_gap_summary.csv").set_index("method")
    close(float(gaps.loc["BestStrong", "mean_gap_pp"]), 0.000081846189101178)
    close(float(gaps.loc["CSD-MR", "mean_gap_pp"]), 0.00358823440784189)
    close(float(gaps.loc["CSD-MR", "max_gap_pp"]), 0.0156268516933067)

    print("PASS: compact frozen summaries match the released reference values")


if __name__ == "__main__":
    main()

