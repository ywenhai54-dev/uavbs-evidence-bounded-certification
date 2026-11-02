"""Create publication tables/statistics from duplicate-cache-corrected formal results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/formal_inputs"
DEFAULT_CORRECTED = ROOT / "outputs/formal_search"
DEFAULT_OUT = ROOT / "outputs/main_result_tables"
NUM_TOL = 1e-10


def bootstrap_ci(values: np.ndarray, seed: int = 20260901, draws: int = 50000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def cohen_dz(values: np.ndarray) -> float:
    std = float(np.std(values, ddof=1))
    if std <= 0:
        return float("inf")
    return float(np.mean(values) / std)


def holm(p_values: dict[str, float]) -> dict[str, float]:
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p_value) in enumerate(items):
        value = min(1.0, (m - rank) * float(p_value))
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrected-dir", type=Path, default=DEFAULT_CORRECTED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(args.corrected_dir / "formal_case_method_results.csv")
    exact = pd.read_csv(BASE / "exact_coverage_50m" / "exact_coverage_case_results.csv")
    trace = pd.read_csv(args.corrected_dir / "csd_mr_search_trace.csv")
    confirm = rows[rows["analysis_role"].eq("confirmatory")].copy()
    exact_confirm = exact[exact["analysis_role"].eq("confirmatory")].copy()
    if confirm["case_id"].nunique() != 48:
        raise RuntimeError("expected 48 confirmatory cases")

    table_methods = ["Coverage-first", "Exact Coverage-MILP", "BestStrong", "CSD-MR"]
    stage_cols = ["C_nom", "PHY", "LINK", "NET", "APP_LB", "APP_UB"]
    summary_input = pd.concat(
        [
            confirm[confirm["method"].isin(["Coverage-first", "BestStrong", "CSD-MR"])],
            exact_confirm[exact_confirm["method"].eq("Exact Coverage-MILP")],
        ],
        ignore_index=True,
        sort=False,
    )
    method_summary = summary_input.groupby("method")[stage_cols].agg(["mean", "median", "std"])
    method_summary.columns = ["_".join(col) for col in method_summary.columns]
    method_summary = method_summary.reindex(table_methods).reset_index()
    method_summary.to_csv(args.out_dir / "table4_corrected_method_summary.csv", index=False)

    stage_rows = []
    for method, group in summary_input[summary_input["method"].isin(table_methods)].groupby("method"):
        means = group[["C_nom", "PHY", "LINK", "NET", "APP_LB"]].mean()
        stage_rows.append(
            {
                "method": method,
                "C_nom_percent": 100.0 * float(means["C_nom"]),
                "C_PHY_percent": 100.0 * float(means["PHY"]),
                "C_LINK_percent": 100.0 * float(means["LINK"]),
                "C_NET_percent": 100.0 * float(means["NET"]),
                "APP_LB_percent": 100.0 * float(means["APP_LB"]),
                "loss_NC_to_AF_pp": 100.0 * float(means["C_nom"] - means["PHY"]),
                "loss_PHY_to_LINK_pp": 100.0 * float(means["PHY"] - means["LINK"]),
                "loss_AF_to_BR_pp": 100.0 * float(means["LINK"] - means["NET"]),
                "loss_BR_to_AA_pp": 100.0 * float(means["NET"] - means["APP_LB"]),
            }
        )
    stage_losses = pd.DataFrame(stage_rows).set_index("method").reindex(table_methods).reset_index()
    stage_losses.to_csv(args.out_dir / "fig6_corrected_stage_losses.csv", index=False)

    best = confirm[confirm["method"].eq("BestStrong")][["case_id", "APP_LB"]].rename(columns={"APP_LB": "BestStrong_LB"})
    csd = confirm[confirm["method"].eq("CSD-MR")][["case_id", "APP_LB", "APP_UB", "CSD_MR_gain_LB"]].rename(
        columns={"APP_LB": "CSD_MR_LB", "APP_UB": "CSD_MR_UB"}
    )
    paired = best.merge(csd, on="case_id", validate="one_to_one")
    paired["gain"] = paired["CSD_MR_LB"] - paired["BestStrong_LB"]
    paired.to_csv(args.out_dir / "corrected_paired_csd_mr_vs_beststrong.csv", index=False)
    gain = paired["gain"].to_numpy(float)
    wilcoxon = stats.wilcoxon(paired["CSD_MR_LB"], paired["BestStrong_LB"], zero_method="wilcox", alternative="greater")
    ci_low, ci_high = bootstrap_ci(gain * 100.0)
    p_raw = float(wilcoxon.pvalue)
    p_holm = holm({"CSD-MR_vs_BestStrong": p_raw})["CSD-MR_vs_BestStrong"]
    stats_payload: dict[str, Any] = {
        "source": str(args.corrected_dir.resolve()),
        "confirmatory_cases": 48,
        "BestStrong_mean_LB_percent": 100.0 * float(paired["BestStrong_LB"].mean()),
        "CSD_MR_mean_LB_percent": 100.0 * float(paired["CSD_MR_LB"].mean()),
        "CSD_MR_median_LB_percent": 100.0 * float(paired["CSD_MR_LB"].median()),
        "mean_gain_pp": 100.0 * float(gain.mean()),
        "median_gain_pp": 100.0 * float(np.median(gain)),
        "bootstrap_95_CI_gain_pp": [ci_low, ci_high],
        "wilcoxon_statistic": float(wilcoxon.statistic),
        "wilcoxon_p_raw": p_raw,
        "wilcoxon_p_holm": p_holm,
        "cohen_dz": cohen_dz(gain),
        "win_tie_loss": {
            "win": int((gain > NUM_TOL).sum()),
            "tie": int((np.abs(gain) <= NUM_TOL).sum()),
            "loss": int((gain < -NUM_TOL).sum()),
        },
    }
    (args.out_dir / "corrected_csd_mr_main_statistics.json").write_text(
        json.dumps(stats_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    duplicate_cols = [
        "duplicate_proposals",
        "duplicate_cache_accept",
        "duplicate_cache_reject",
        "duplicate_cache_refine",
        "new_state_certify",
        "unresolved_after_certification",
        "candidate_visits",
        "MILP_calls",
        "safe_pruned_candidates",
    ]
    duplicate_summary = confirm[confirm["method"].eq("CSD-MR")][duplicate_cols].sum(numeric_only=True).to_dict()
    trace_status = trace["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="count")
    trace_status.to_csv(args.out_dir / "corrected_trace_status_counts.csv", index=False)
    (args.out_dir / "corrected_duplicate_stats.json").write_text(
        json.dumps(duplicate_summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    numeric_map = {
        "CSD-MR mean AA LB": f"{stats_payload['CSD_MR_mean_LB_percent']:.2f}%",
        "CSD-MR mean gain": f"{stats_payload['mean_gain_pp']:.2f} pp",
        "CSD-MR median gain": f"{stats_payload['median_gain_pp']:.2f} pp",
        "CSD-MR gain 95% CI": f"[{ci_low:.2f}, {ci_high:.2f}] pp",
        "CSD-MR dz": f"{stats_payload['cohen_dz']:.2f}",
        "CSD-MR win/tie/loss": (
            f"{stats_payload['win_tie_loss']['win']}/"
            f"{stats_payload['win_tie_loss']['tie']}/"
            f"{stats_payload['win_tie_loss']['loss']}"
        ),
        "CSD-MR candidate visits": str(int(duplicate_summary["candidate_visits"])),
        "CSD-MR MILP calls": str(int(duplicate_summary["MILP_calls"])),
        "CSD-MR duplicate proposals": str(int(duplicate_summary["duplicate_proposals"])),
    }
    (args.out_dir / "paper_numeric_replacements.json").write_text(
        json.dumps(numeric_map, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(stats_payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
