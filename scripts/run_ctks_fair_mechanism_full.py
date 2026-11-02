"""Run the frozen four-arm CT-KS comparison with resumable checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ctks_fair_certification import (
    METHODS,
    METHOD_CTKS,
    METHOD_FULL,
    METHOD_RELAXED,
    METHOD_THRESHOLD,
    RELAXED_MIP_GAP,
    TOTAL_BUDGET_SECONDS,
    certify_built_candidate,
)
from run_ctks_fixed_candidate_replay import audit_trace, run_unit
from run_ctks_outer_search_gate import run_outer_case_core


BASE = ROOT / "data/formal_inputs"
FORMAL = ROOT / "outputs/formal_search"
OUT = ROOT / "outputs/ctks_fair_mechanism"
REPEATS = 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def relevant_sources() -> list[Path]:
    return [
        ROOT / "src/ctks_service_chain_kernel.py",
        ROOT / "src/ctks_fair_certification.py",
        ROOT / "scripts/run_ctks_outer_search_gate.py",
        ROOT / "scripts/run_ctks_fixed_candidate_replay.py",
        Path(__file__).resolve(),
        ROOT / "scripts/summarize_ctks_fair_mechanism_full.py",
    ]


def protocol_payload() -> dict[str, Any]:
    try:
        import scipy.optimize._highspy._core as highs_core  # type: ignore

        highs_version = getattr(highs_core, "HIGHS_VERSION_MAJOR", "bundled-unavailable")
    except Exception:
        highs_version = "not exposed by the installed SciPy interface"
    return {
        "experiment": "CT-KS fair mechanism full comparison",
        "created_for_date": "2026-09-08",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "scipy_version": scipy.__version__,
        "highs_version": highs_version,
        "solver_interface": "scipy.optimize.milp (bundled HiGHS)",
        "methods": {
            METHOD_FULL: {"mip_rel_gap": 0.0, "threshold_aware": False, "kernel": False},
            METHOD_THRESHOLD: {"mip_rel_gap": 0.0, "threshold_aware": True, "kernel": False},
            METHOD_CTKS: {"mip_rel_gap": 0.0, "threshold_aware": True, "kernel": True},
            METHOD_RELAXED: {
                "mip_rel_gap": RELAXED_MIP_GAP,
                "threshold_aware": True,
                "kernel": False,
            },
        },
        "threads": 1,
        "solver_seed": 0,
        "nominal_per_candidate_budget_seconds": TOTAL_BUDGET_SECONDS,
        "budget_semantics": "best-effort total certification wall-clock budget; observed overrun retained",
        "ctks_stage_budget_seconds": 1.0,
        "H": 6,
        "beam_width": 2,
        "M": 4,
        "N_visit": 4,
        "confirmation_seeds": [1, 2, 3, 4],
        "outer_expected_runs": 576,
        "replay_expected_runs": 6768,
        "replay_expected_proposals": 564,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in relevant_sources()},
        "input_sha256": {
            "formal_case_method_results.csv": sha256(FORMAL / "formal_case_method_results.csv"),
            "csd_mr_search_trace.csv": sha256(FORMAL / "csd_mr_search_trace.csv"),
        },
    }


def freeze_manifest(out: Path) -> dict[str, Any]:
    payload = protocol_payload()
    target = out / "manifest.json"
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if old != payload:
            raise RuntimeError("Existing manifest differs from the current code/config snapshot")
    else:
        write_json(target, payload)
    return payload


def load_outer_schedule() -> pd.DataFrame:
    source = pd.read_csv(FORMAL / "formal_case_method_results.csv")
    confirm = source[
        source["case_id"].str.contains(r"seed[1-4]$", regex=True)
        & source["method"].eq("CSD-MR")
    ].copy()
    starts = source[
        source["case_id"].str.contains(r"seed[1-4]$", regex=True)
        & source["method"].eq("BestStrong")
    ][["case_id", "selected_candidate_ids", "source_method"]].copy()
    starts = starts.rename(
        columns={
            "selected_candidate_ids": "initial_selected_candidate_ids",
            "source_method": "initial_source_method",
        }
    )
    merged = confirm.merge(starts, on="case_id", how="inner", validate="one_to_one")
    if len(merged) != 48:
        raise RuntimeError(f"Expected 48 confirmation cases, found {len(merged)}")
    rows: list[dict[str, Any]] = []
    for base in merged.sort_values("case_id", kind="stable").to_dict("records"):
        for repeat_id in range(1, REPEATS + 1):
            for method in METHODS:
                rows.append(
                    {
                        "case_id": base["case_id"],
                        "terrain": base["terrain"],
                        "load_level": base["infrastructure_level"],
                        "seed": int(base["seed"]),
                        "repeat_id": repeat_id,
                        "method": method,
                        "selected_candidate_ids": base["initial_selected_candidate_ids"],
                        "source_method": base["initial_source_method"],
                    }
                )
    schedule = pd.DataFrame(rows)
    if len(schedule) != 576 or schedule.duplicated(["case_id", "repeat_id", "method"]).any():
        raise RuntimeError("Invalid outer schedule")
    return schedule


def run_outer_one(task: dict[str, Any], historical: pd.Series) -> dict[str, Any]:
    method = str(task["method"])
    started = perf_counter()
    result, trace, checkpoints = run_outer_case_core(
        case_id=str(task["case_id"]),
        initial_selected=str(task["selected_candidate_ids"]).split(";"),
        initial_method=str(task["source_method"]),
        historical=historical,
        stabilize_selected_diagnosis=False,
        max_horizon=6,
        offered_load_mbps=None,
        candidate_total_milp_budget_seconds=5.0,
        candidate_stage_budget_seconds=1.0,
        candidate_policy=method,
        candidate_certifier=partial(certify_built_candidate, method=method),
    )
    result.update(
        {
            "repeat_id": int(task["repeat_id"]),
            "method": method,
            "case_wall_seconds": perf_counter() - started,
        }
    )
    for row in trace:
        row.update({"repeat_id": int(task["repeat_id"]), "method": method})
    for row in checkpoints:
        row.update({"repeat_id": int(task["repeat_id"]), "method": method})
    return {"result": result, "trace": trace, "checkpoints": checkpoints}


def collect_outer(out: Path) -> None:
    blocks = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((out / "outer_runs").glob("*.json"))]
    pd.DataFrame([block["result"] for block in blocks]).to_csv(out / "raw_outer_runs.csv", index=False)
    pd.DataFrame([row for block in blocks for row in block["trace"]]).to_csv(out / "raw_outer_trace.csv", index=False)
    pd.DataFrame([row for block in blocks for row in block["checkpoints"]]).to_csv(out / "raw_outer_checkpoints.csv", index=False)


def run_outer(out: Path, resume: bool) -> None:
    schedule = load_outer_schedule()
    schedule.to_csv(out / "outer_schedule.csv", index=False)
    run_dir = out / "outer_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    formal = pd.read_csv(FORMAL / "formal_case_method_results.csv")
    historical = formal[formal["method"].eq("CSD-MR")].set_index("case_id")
    for index, task in enumerate(schedule.to_dict("records"), 1):
        run_id = f"{task['case_id']}__{task['method']}__rep{task['repeat_id']}"
        target = run_dir / f"{run_id}.json"
        if not (resume and target.exists()):
            block = run_outer_one(task, historical.loc[str(task["case_id"])].copy())
            write_json(target, block)
        if index % 12 == 0 or index == len(schedule):
            print(f"OUTER {index}/{len(schedule)}", flush=True)
    collect_outer(out)


def replay_filename(unit: dict[str, Any], method: str, repeat_id: int) -> str:
    return (
        f"{unit['case_id']}__d{int(unit['depth']):02d}__e{int(unit['evaluation_index']):03d}"
        f"__{method}__rep{repeat_id}.json"
    )


def collect_replay(out: Path) -> None:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((out / "replay_runs").rglob("*.json"))]
    pd.DataFrame(rows).to_csv(out / "raw_replay_runs.csv", index=False)


def run_replay(out: Path, resume: bool) -> None:
    audited = audit_trace()
    audited.to_csv(out / "replay_parent_audit.csv", index=False)
    run_dir = out / "replay_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    total = len(audited) * REPEATS * len(METHODS)
    position = 0
    for unit in audited.to_dict("records"):
        case_dir = run_dir / str(unit["case_id"])
        case_dir.mkdir(parents=True, exist_ok=True)
        for repeat_id in range(1, REPEATS + 1):
            for method in METHODS:
                position += 1
                target = case_dir / replay_filename(unit, method, repeat_id)
                if not (resume and target.exists()):
                    write_json(target, run_unit(unit, method, repeat_id))
                if position % 120 == 0 or position == total:
                    print(f"REPLAY {position}/{total}", flush=True)
    collect_replay(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("manifest", "outer", "replay", "all"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    freeze_manifest(args.out_dir)
    if args.phase in {"outer", "all"}:
        run_outer(args.out_dir, args.resume)
    if args.phase in {"replay", "all"}:
        run_replay(args.out_dir, args.resume)
    print("RUN PHASE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
