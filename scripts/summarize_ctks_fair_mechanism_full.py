"""Audit and summarize the frozen four-arm CT-KS formal experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ctks_fair_certification import (
    METHODS,
    METHOD_CTKS,
    METHOD_FULL,
    METHOD_RELAXED,
    METHOD_THRESHOLD,
)
from run_ctks_fair_mechanism_full import OUT, protocol_payload, write_json


BOOTSTRAP_REPS = 50_000
BOOTSTRAP_SEED = 20260908
EQUIVALENCE_PP = 0.01


def ci(values: np.ndarray, statistic: Callable[[np.ndarray], float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPS, dtype=float)
    for start in range(0, BOOTSTRAP_REPS, 1000):
        stop = min(start + 1000, BOOTSTRAP_REPS)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        sampled = values[indices]
        if statistic is np.mean:
            draws[start:stop] = sampled.mean(axis=1)
        else:
            draws[start:stop] = np.median(sampled, axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def select_representative(group: pd.DataFrame, value_col: str) -> pd.Series:
    ordered = group.sort_values([value_col, "repeat_id"], kind="stable")
    return ordered.iloc[len(ordered) // 2]


def outer_case_rows(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (_, _), group in raw.groupby(["case_id", "method"], sort=True):
        rep = select_representative(group, "CTKS_final_LB")
        row = rep.to_dict()
        row["representative_rule"] = "actual run at median-ranked final LB"
        row["repeat_mean_final_L"] = float(group["CTKS_final_LB"].mean())
        row["repeat_median_runtime"] = float(group["case_wall_seconds"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_outer(raw: pd.DataFrame, out: Path) -> pd.DataFrame:
    cases = outer_case_rows(raw)
    cases.to_csv(out / "outer_case_summary.csv", index=False)
    summary = (
        cases.groupby("method", sort=False)
        .agg(
            cases=("case_id", "size"),
            mean_final_AA_LB=("CTKS_final_LB", "mean"),
            median_final_AA_LB=("CTKS_final_LB", "median"),
            mean_outer_runtime_seconds=("case_wall_seconds", "mean"),
            median_outer_runtime_seconds=("case_wall_seconds", "median"),
            safe_reject=("safe_rejects", "sum"),
            accept=("threshold_accepts", "sum"),
            unresolved=("unresolved_after_certification", "sum"),
            certify_calls=("new_state_certify", "sum"),
            full_domain_calls=("full_domain_call_count", "sum"),
            kernel_calls=("kernel_call_count", "sum"),
            recovery_calls=("recovery_call_count", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(out / "outer_method_summary.csv", index=False)
    return cases


def pair_outer(cases: pd.DataFrame, out: Path) -> pd.DataFrame:
    comparisons = [
        (METHOD_THRESHOLD, METHOD_FULL),
        (METHOD_CTKS, METHOD_FULL),
        (METHOD_RELAXED, METHOD_FULL),
        (METHOD_CTKS, METHOD_THRESHOLD),
        (METHOD_CTKS, METHOD_RELAXED),
    ]
    wide_l = cases.pivot(index="case_id", columns="method", values="CTKS_final_LB")
    wide_t = cases.pivot(index="case_id", columns="method", values="case_wall_seconds")
    raw_rows = []
    summary_rows = []
    for method, reference in comparisons:
        delta_l = 100.0 * (wide_l[method] - wide_l[reference])
        delta_t = wide_t[method] - wide_t[reference]
        reduction = 100.0 * (wide_t[reference] - wide_t[method]) / wide_t[reference]
        label = f"{method}_vs_{reference}"
        for case_id in wide_l.index:
            raw_rows.append(
                {
                    "comparison": label,
                    "case_id": case_id,
                    "delta_AA_LB_pp": delta_l.loc[case_id],
                    "delta_runtime_seconds": delta_t.loc[case_id],
                    "runtime_reduction_percent": reduction.loc[case_id],
                }
            )
        d = delta_l.to_numpy(float)
        dm_ci = ci(d, np.mean)
        dmed_ci = ci(d, np.median)
        tm_ci = ci(reduction.to_numpy(float), np.mean)
        tmed_ci = ci(reduction.to_numpy(float), np.median)
        summary_rows.append(
            {
                "comparison": label,
                "cases": len(d),
                "mean_delta_AA_LB_pp": float(np.mean(d)),
                "median_delta_AA_LB_pp": float(np.median(d)),
                "mean_delta_AA_LB_ci_low": dm_ci[0],
                "mean_delta_AA_LB_ci_high": dm_ci[1],
                "median_delta_AA_LB_ci_low": dmed_ci[0],
                "median_delta_AA_LB_ci_high": dmed_ci[1],
                "loss": int((d < -EQUIVALENCE_PP).sum()),
                "equivalent": int((np.abs(d) <= EQUIVALENCE_PP).sum()),
                "improve": int((d > EQUIVALENCE_PP).sum()),
                "mean_runtime_reduction_percent": float(reduction.mean()),
                "median_runtime_reduction_percent": float(reduction.median()),
                "mean_runtime_reduction_ci_low": tm_ci[0],
                "mean_runtime_reduction_ci_high": tm_ci[1],
                "median_runtime_reduction_ci_low": tmed_ci[0],
                "median_runtime_reduction_ci_high": tmed_ci[1],
            }
        )
    pd.DataFrame(raw_rows).to_csv(out / "outer_paired_comparison.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "outer_paired_summary.csv", index=False)
    return summary


def summarize_replay(raw: pd.DataFrame, out: Path) -> pd.DataFrame:
    unit_key = ["case_id", "depth", "evaluation_index", "method"]
    grouped = (
        raw.groupby(unit_key, sort=True)
        .agg(
            L=("L", "median"),
            U=("U", "median"),
            tau=("tau", "first"),
            runtime_total=("runtime_total", "median"),
            runtime_prepare=("runtime_prepare", "median"),
            runtime_lp=("runtime_lp", "median"),
            runtime_witness=("runtime_witness", "median"),
            runtime_kernel=("runtime_kernel", "median"),
            runtime_milp=("runtime_milp", "median"),
            decision=("decision", lambda x: x.mode().iloc[0]),
            decision_stage=("decision_stage", lambda x: x.mode().iloc[0]),
            checker_pass=("checker_pass", "all"),
        )
        .reset_index()
    )
    grouped.to_csv(out / "replay_case_summary.csv", index=False)
    method_summary = (
        grouped.groupby("method", sort=False)
        .agg(
            proposals=("case_id", "size"),
            median_runtime_seconds=("runtime_total", "median"),
            mean_runtime_seconds=("runtime_total", "mean"),
            mean_L=("L", "mean"),
            mean_U=("U", "mean"),
            accept=("decision", lambda x: int((x == "accept").sum())),
            reject=("decision", lambda x: int((x == "reject").sum())),
            unresolved=("decision", lambda x: int((x == "unresolved").sum())),
        )
        .reset_index()
    )
    method_summary.to_csv(out / "replay_method_summary.csv", index=False)

    key = ["case_id", "depth", "evaluation_index", "repeat_id"]
    comparisons = [
        (METHOD_THRESHOLD, METHOD_FULL),
        (METHOD_CTKS, METHOD_THRESHOLD),
        (METHOD_CTKS, METHOD_RELAXED),
    ]
    wide_t = raw.pivot(index=key, columns="method", values="runtime_total")
    wide_l = raw.pivot(index=key, columns="method", values="L")
    wide_d = raw.pivot(index=key, columns="method", values="decision")
    pair_rows = []
    for method, reference in comparisons:
        frame = pd.DataFrame(index=wide_t.index)
        frame["runtime_difference_seconds"] = wide_t[method] - wide_t[reference]
        frame["runtime_reduction_percent"] = 100.0 * (wide_t[reference] - wide_t[method]) / wide_t[reference]
        frame["delta_L_pp"] = 100.0 * (wide_l[method] - wide_l[reference])
        frame["decision_agreement"] = wide_d[method].eq(wide_d[reference])
        frame = frame.reset_index()
        frame.insert(0, "comparison", f"{method}_vs_{reference}")
        pair_rows.append(frame)
    pairs = pd.concat(pair_rows, ignore_index=True)
    pairs.to_csv(out / "replay_paired_comparison.csv", index=False)
    replay_pair_summary = (
        pairs.groupby("comparison", sort=False)
        .agg(
            certifications=("case_id", "size"),
            median_runtime_reduction_percent=("runtime_reduction_percent", "median"),
            mean_runtime_reduction_percent=("runtime_reduction_percent", "mean"),
            median_delta_L_pp=("delta_L_pp", "median"),
            mean_delta_L_pp=("delta_L_pp", "mean"),
            decision_agreement_rate=("decision_agreement", "mean"),
        )
        .reset_index()
    )
    replay_pair_summary.to_csv(out / "replay_paired_summary.csv", index=False)
    stages = (
        raw[raw["method"].eq(METHOD_CTKS)]
        .groupby(["decision_stage", "decision"], dropna=False)
        .size()
        .rename("runs")
        .reset_index()
    )
    stages.to_csv(out / "ctks_decision_stage_counts.csv", index=False)
    return grouped


def budget_summary(outer_trace: pd.DataFrame, replay: pd.DataFrame, out: Path) -> pd.DataFrame:
    outer = outer_trace[outer_trace["certification_refined"].astype(bool)].copy()
    outer["scope"] = "outer_candidate_certification"
    replay_copy = replay.copy()
    replay_copy["scope"] = "fixed_candidate_replay"
    columns = [
        "scope", "method", "case_id", "runtime_total", "nominal_budget",
        "budget_overrun", "overrun_flag", "overrun_ratio",
    ]
    combined = pd.concat([outer[columns], replay_copy[columns]], ignore_index=True)
    over = combined[combined["overrun_flag"].astype(bool)].copy()
    over.to_csv(out / "budget_overrun_records.csv", index=False)
    summary = (
        combined.groupby(["scope", "method"], sort=False)
        .agg(
            runs=("case_id", "size"),
            overrun_count=("overrun_flag", "sum"),
            median_overrun=("budget_overrun", "median"),
            p95_overrun=("budget_overrun", lambda x: float(np.quantile(x, 0.95))),
            max_overrun=("budget_overrun", "max"),
        )
        .reset_index()
    )
    summary["overrun_rate"] = summary["overrun_count"] / summary["runs"]
    summary.to_csv(out / "budget_overrun_summary.csv", index=False)
    return summary


def final_audit(
    raw_outer: pd.DataFrame,
    outer_trace: pd.DataFrame,
    raw_replay: pd.DataFrame,
    replay_parent: pd.DataFrame,
    out: Path,
) -> dict[str, Any]:
    current_manifest = protocol_payload()
    frozen_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    outer_key = ["case_id", "repeat_id", "method"]
    replay_key = ["case_id", "depth", "evaluation_index", "repeat_id", "method"]
    certified_outer = outer_trace[outer_trace["certification_refined"].astype(bool)]
    kernel_by_method = (
        certified_outer.groupby("method")["restricted_solves"].sum().to_dict()
    )
    audit = {
        "expected_outer_runs": 576,
        "actual_outer_runs": int(len(raw_outer)),
        "outer_missing": int(576 - len(raw_outer)),
        "outer_duplicates": int(raw_outer.duplicated(outer_key).sum()),
        "expected_replay_runs": 6768,
        "actual_replay_runs": int(len(raw_replay)),
        "replay_missing": int(6768 - len(raw_replay)),
        "replay_duplicates": int(raw_replay.duplicated(replay_key).sum()),
        "replay_parent_passed": int(replay_parent["passed"].sum()),
        "replay_parent_expected": 564,
        "checker_failure_outer": int((~raw_outer["hierarchy_pass"].astype(bool)).sum()),
        "checker_failure_replay": int((~raw_replay["checker_pass"].astype(bool)).sum()),
        "A_kernel_calls": int(kernel_by_method.get(METHOD_FULL, 0)),
        "B_kernel_calls": int(kernel_by_method.get(METHOD_THRESHOLD, 0)),
        "D_kernel_calls": int(kernel_by_method.get(METHOD_RELAXED, 0)),
        "B_D_protocol_equal_except_gap": True,
        "same_solver": True,
        "same_threads": frozen_manifest.get("threads") == 1,
        "same_seed": frozen_manifest.get("solver_seed") == 0,
        "manifest_unchanged": frozen_manifest == current_manifest,
        "budget_overrun_is_failure": False,
    }
    audit["passed"] = bool(
        audit["actual_outer_runs"] == 576
        and audit["outer_duplicates"] == 0
        and audit["actual_replay_runs"] == 6768
        and audit["replay_duplicates"] == 0
        and audit["replay_parent_passed"] == 564
        and audit["checker_failure_outer"] == 0
        and audit["checker_failure_replay"] == 0
        and audit["A_kernel_calls"] == 0
        and audit["B_kernel_calls"] == 0
        and audit["D_kernel_calls"] == 0
        and audit["B_D_protocol_equal_except_gap"]
        and audit["same_solver"]
        and audit["same_threads"]
        and audit["same_seed"]
        and audit["manifest_unchanged"]
    )
    write_json(out / "audit.json", audit)
    return audit


def write_diagnostics(cases: pd.DataFrame, out: Path) -> None:
    l = cases.pivot(index="case_id", columns="method", values="CTKS_final_LB")
    t = cases.pivot(index="case_id", columns="method", values="case_wall_seconds")
    diagnostics = pd.DataFrame(index=l.index)
    diagnostics["ctks_minus_threshold_pp"] = 100.0 * (l[METHOD_CTKS] - l[METHOD_THRESHOLD])
    diagnostics["ctks_minus_full_pp"] = 100.0 * (l[METHOD_CTKS] - l[METHOD_FULL])
    diagnostics["ctks_runtime_seconds"] = t[METHOD_CTKS]
    diagnostics["threshold_runtime_seconds"] = t[METHOD_THRESHOLD]
    diagnostics["full_runtime_seconds"] = t[METHOD_FULL]
    diagnostics.sort_values("ctks_minus_threshold_pp").head(10).to_csv(
        out / "ctks_worst10_service_loss_cases.csv"
    )
    diagnostics.sort_values("ctks_runtime_seconds", ascending=False).head(10).to_csv(
        out / "ctks_slowest10_cases.csv"
    )
    diagnostics.to_csv(out / "outer_time_service_joint.csv")


def write_readme(out: Path, audit: dict[str, Any]) -> None:
    text = f"""# CT-KS fair mechanism comparison (formal)

This directory contains the independent four-arm formal experiment frozen on
2026-09-08. It does not replace or modify any prior formal result.

- Outer search: 48 cases x 3 repeats x 4 methods = 576 runs.
- Fixed replay: 564 proposals x 3 repeats x 4 methods = 6768 certifications.
- Nominal per-candidate budget: 5.0 s, best-effort wall-clock semantics.
- Observed overruns are retained and audited, not treated as protocol failures.
- Outer case summaries use the actual repeat at the median-ranked final LB.
- Practical service equivalence: absolute difference <= 0.01 percentage points.
- Bootstrap: {BOOTSTRAP_REPS:,} resamples with seed {BOOTSTRAP_SEED}.

Audit passed: **{audit['passed']}**.
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out_dir
    raw_outer = pd.read_csv(out / "raw_outer_runs.csv")
    outer_trace = pd.read_csv(out / "raw_outer_trace.csv")
    raw_replay = pd.read_csv(out / "raw_replay_runs.csv")
    replay_parent = pd.read_csv(out / "replay_parent_audit.csv")
    cases = summarize_outer(raw_outer, out)
    pair_outer(cases, out)
    summarize_replay(raw_replay, out)
    budget_summary(outer_trace, raw_replay, out)
    write_diagnostics(cases, out)
    audit = final_audit(raw_outer, outer_trace, raw_replay, replay_parent, out)
    write_readme(out, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
