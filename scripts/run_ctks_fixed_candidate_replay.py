"""Fixed-candidate replay and parent-state audit for the CT-KS comparison."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ctks_fair_certification import METHODS, certify_built_candidate
from ctks_service_chain_kernel import build_service_chains
from protocol_stack_pipeline_common import load_config
from run_ctks_outer_search_gate import CONFIG, INPUTS, load_case


BASE = ROOT / "data/formal_inputs"
TRACE = ROOT / "outputs/formal_search/csd_mr_search_trace.csv"
OUT = ROOT / "outputs/ctks_replay_preflight"


def parse_ids(value: Any) -> tuple[str, ...]:
    return tuple(sorted(item for item in str(value).split(";") if item))


def reconstruct_parent(
    candidate_ids: Iterable[str], candidate_out: str, candidate_in: str
) -> tuple[str, ...]:
    candidate = set(map(str, candidate_ids))
    outgoing = str(candidate_out)
    incoming = str(candidate_in)
    parent = (candidate - {incoming}) | {outgoing}
    return tuple(sorted(parent))


def audit_trace(trace_path: Path = TRACE) -> pd.DataFrame:
    source = pd.read_csv(trace_path)
    source = source[source["case_id"].str.contains(r"seed[1-4]$", regex=True)].copy()
    rows: list[dict[str, Any]] = []
    for row in source.itertuples(index=False):
        candidate = parse_ids(row.selected_candidate_ids)
        parent = reconstruct_parent(candidate, row.candidate_out, row.candidate_in)
        parent_set = set(parent)
        candidate_set = set(candidate)
        checks = {
            "parent_size_8": len(parent) == 8,
            "candidate_size_8": len(candidate) == 8,
            "symmetric_difference_2": len(parent_set ^ candidate_set) == 2,
            "out_in_parent": str(row.candidate_out) in parent_set,
            "out_not_in_candidate": str(row.candidate_out) not in candidate_set,
            "in_not_in_parent": str(row.candidate_in) not in parent_set,
            "in_in_candidate": str(row.candidate_in) in candidate_set,
        }
        rows.append(
            {
                "case_id": row.case_id,
                "depth": int(row.depth),
                "evaluation_index": int(row.evaluation_index),
                "candidate_out": str(row.candidate_out),
                "candidate_in": str(row.candidate_in),
                "parent_ids": ";".join(parent),
                "candidate_ids": ";".join(candidate),
                "tau": float(row.parent_threshold),
                "source_status": str(row.status),
                **checks,
                "passed": all(checks.values()),
            }
        )
    audited = pd.DataFrame(rows)
    if len(audited) != 564:
        raise RuntimeError(f"Expected 564 confirmatory proposals, found {len(audited)}")
    if not audited["passed"].all():
        failed = audited.loc[~audited["passed"]]
        raise RuntimeError(f"Parent reconstruction failed for {len(failed)} proposals")
    return audited


def select_preflight_units(audited: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for terrain in ("S1", "S2", "S3", "S4"):
        group = audited[audited["case_id"].str.startswith(terrain + "_")]
        terrain_selected = []
        for pattern in (
            "safe_pruned_by_valid_bound",
            "new_state_certify_accepted",
            "unresolved_after_certification",
            "duplicate_cache_accept",
        ):
            match = group[group["source_status"].eq(pattern)]
            if not match.empty:
                terrain_selected.append(match.iloc[0].to_dict())
        if len(terrain_selected) < 4:
            terrain_selected.extend(
                group.head(4 - len(terrain_selected)).to_dict("records")
            )
        selected.extend(terrain_selected)
    result = pd.DataFrame(selected).drop_duplicates(
        ["case_id", "depth", "evaluation_index"], keep="first"
    )
    unresolved = audited[audited["source_status"].eq("unresolved_after_certification")]
    result = pd.concat([result, unresolved], ignore_index=True).drop_duplicates(
        ["case_id", "depth", "evaluation_index"], keep="first"
    )
    return result.sort_values(["case_id", "depth", "evaluation_index"], kind="stable")


def run_unit(unit: dict[str, Any], method: str, repeat_id: int) -> dict[str, Any]:
    prepare_started = perf_counter()
    case, demand, _, access, links, options = load_case(str(unit["case_id"]), INPUTS)
    cfg, _, _ = load_config(str(CONFIG))
    capacity = cfg["capacity"]
    selected = parse_ids(unit["candidate_ids"])
    built = build_service_chains(
        case_id=str(unit["case_id"]),
        terrain=str(case["terrain"]),
        selected_ids=selected,
        demand=demand,
        access_phy_edges=access,
        link_edges=links,
        backhaul_options=options,
    )
    # A replay unit is a standalone certification. Charge its common data read
    # and path-model construction equally to every method.
    build_seconds = perf_counter() - prepare_started
    result = certify_built_candidate(
        built=built,
        uav_capacity_mbps=float(capacity["uav_aggregate_capacity_mbps"]),
        gbs_capacity_mbps=float(capacity["gbs_egress_capacity_mbps"]),
        tau=float(unit["tau"]),
        method=method,
        build_seconds=build_seconds,
    )
    return {
        "case_id": unit["case_id"],
        "terrain": case["terrain"],
        "load_level": case["infrastructure_level"],
        "seed": case["seed"],
        "repeat_id": repeat_id,
        "depth": int(unit["depth"]),
        "evaluation_index": int(unit["evaluation_index"]),
        "candidate_out": unit["candidate_out"],
        "candidate_in": unit["candidate_in"],
        "parent_ids": unit["parent_ids"],
        "candidate_ids": unit["candidate_ids"],
        "source_status": unit["source_status"],
        **{key: value for key, value in result.row.items() if key != "solve_history"},
        "solve_history_json": json.dumps(result.row["solve_history"], sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audited = audit_trace()
    audited.to_csv(args.out_dir / "replay_parent_audit.csv", index=False)
    if args.audit_only:
        print(json.dumps({"rows": len(audited), "passed": True}, indent=2))
        return
    if not args.preflight:
        raise SystemExit("Use --audit-only or --preflight")
    units = select_preflight_units(audited)
    units.to_csv(args.out_dir / "replay_preflight_units.csv", index=False)
    rows = []
    for unit in units.to_dict("records"):
        for repeat_id in range(1, args.repeats + 1):
            for method in METHODS:
                rows.append(run_unit(unit, method, repeat_id))
    output = pd.DataFrame(rows)
    output.to_csv(args.out_dir / "replay_preflight_raw.csv", index=False)
    print(
        json.dumps(
            {
                "units": len(units),
                "runs": len(output),
                "methods": list(METHODS),
                "decision_stages": sorted(output["decision_stage"].unique().tolist()),
                "max_budget_overrun_seconds": float(output["budget_overrun_seconds"].max()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
