"""Twelve-case outer-search gate after the CT-KS candidate gate passed."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ctks_service_chain_kernel import (  # noqa: E402
    PolicyResult,
    PreparedKernelModel,
    build_service_chains,
    certification_from_policy,
    prepare_kernel_model,
    refine_for_beam_ranking,
    run_policy,
)
from protocol_stack_pipeline_common import load_config  # noqa: E402
from protocol_stack_v2_certifier import CertificationResult, certify_deployment_v2  # noqa: E402
from protocol_stack_v2_exp6a import (  # noqa: E402
    FAILURE_TO_OPERATOR,
    dominant_attribution,
    generate_operator_candidates,
)
from run_main_50m_formal_search import (  # noqa: E402
    BEAM_WIDTH,
    HORIZON,
    NUM_TOL,
    OPS,
    PROPOSALS_PER_OPERATOR,
    SearchNode,
    schedule_proposals,
)


BASE = ROOT / "data/formal_inputs"
INPUTS = BASE / "inputs"
FORMAL = ROOT / "outputs/formal_search"
CANDIDATE_GATE = ROOT / "outputs/ctks_candidate_gate"
OUT = ROOT / "outputs/ctks_outer_search"
CONFIG = ROOT / "configs/protocol_stack_csd_mr_go_v1.yaml"
TOTAL_MILP_BUDGET_SECONDS = 5.0
STAGE_BUDGET_SECONDS = 1.25
BUCKET_FRACTION = 0.10


@dataclass
class CandidateState:
    model: PreparedKernelModel
    policy: PolicyResult
    built: tuple[Any, ...]
    selected: tuple[str, ...]
    operator: str
    candidate_out: str
    candidate_in: str
    parent_lb: float
    result: CertificationResult | None = None


@dataclass
class PolicyCacheEntry:
    selected: tuple[str, ...]
    model: PreparedKernelModel
    policy: PolicyResult
    built: tuple[Any, ...]
    lb: float
    ub: float
    result: CertificationResult | None


def acceptance_threshold(parent: SearchNode) -> float:
    return float(parent.lb) - NUM_TOL


def cache_decision(entry: PolicyCacheEntry, tau: float) -> str:
    if entry.ub < tau:
        return "reject"
    if entry.lb >= tau:
        return "accept"
    return "insufficient"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def rescale_demand_to_total(
    demand: pd.DataFrame, offered_load_mbps: float | None
) -> tuple[pd.DataFrame, float, float]:
    """Rescale only Mbps demand; POI rows, weights, and spatial fields stay fixed."""
    native_total = float(demand["service_demand_mbps"].sum())
    if offered_load_mbps is None:
        return demand, native_total, native_total
    target = float(offered_load_mbps)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("offered_load_mbps must be finite and positive")
    if native_total <= 0.0:
        raise ValueError("native total demand must be positive")
    scaled = demand.copy()
    scaled["service_demand_mbps"] = (
        scaled["service_demand_mbps"].astype(float) * target / native_total
    )
    effective_total = float(scaled["service_demand_mbps"].sum())
    return scaled, native_total, effective_total


def protocol() -> dict[str, Any]:
    return {
        "status": "FROZEN_BEFORE_OUTER_GATE_RESULTS",
        "authorization": "candidate gate PASS_S_AND_PASS_E_CORE",
        "development_cases": "all 12 seed-0 terrain-by-infrastructure cases",
        "confirmatory_seeds_touched": [],
        "initial_state": "historical BestStrong deployment, newly recertified under the unchanged full certifier",
        "candidate_generation": "unchanged CSD-MR O1/O2/O3 proposals and first-occurrence canonical scheduling",
        "horizon": HORIZON,
        "beam_width": BEAM_WIDTH,
        "proposal_budget_per_operator": PROPOSALS_PER_OPERATOR,
        "visits_per_depth": 4,
        "candidate_policy": "CT-KS with full-domain LP safe rejection and exact-checked kernel witness",
        "candidate_total_milp_budget_seconds": TOTAL_MILP_BUDGET_SECONDS,
        "initial_stage_budget_seconds": STAGE_BUDGET_SECONDS,
        "beam_logic": (
            "accepted children first use certified LBs; when more than B children remain and "
            "selected-LB versus unselected-full-UB intervals overlap, ambiguous candidates "
            "spend only their remaining five-second budget on the complete domain"
        ),
        "ranking_resolved_condition": "minimum selected LB >= maximum unselected complete-domain UB",
        "unresolved_tie_break": "descending valid LB, descending complete-domain UB, canonical deployment",
        "gate_correctness": {
            "illegal_witness": 0,
            "unsafe_prune": 0,
            "accepted_LB_decrease": 0,
            "hierarchy_violation": 0,
        },
        "gate_effectiveness": {
            "median_final_LB_difference_vs_historical_CSD_MR_at_least": -1.0e-4,
            "material_final_LB_loss_rate_below_minus_1e_4_at_most": 0.20,
            "median_search_time_reduction_vs_historical_CSD_MR_at_least": 0.20,
            "depth_level_beam_resolution_rate_at_least": 0.50,
        },
        "forbidden": [
            "change the passed candidate-kernel construction",
            "use a kernel-domain upper bound for hard rejection or beam exclusion",
            "reuse historical candidate solutions as CT-KS warm starts",
            "touch seeds 1-4 before this gate passes",
        ],
        "hashes": {
            "candidate_gate_result": sha256(CANDIDATE_GATE / "gate_result_full.json"),
            "candidate_gate_protocol": sha256(CANDIDATE_GATE / "frozen_protocol.json"),
            "formal_results": sha256(FORMAL / "formal_case_method_results.csv"),
        },
    }


def load_case(
    case_id: str,
    input_root: Path = INPUTS,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    terrain, infrastructure, seed_text = case_id.split("_")
    if not seed_text.startswith("seed"):
        raise ValueError(case_id)
    seed = int(seed_text.removeprefix("seed"))
    demand = pd.read_csv(
        input_root / "demand_state" / terrain / f"{infrastructure}_seed_{seed}" / "poi_demand.csv"
    )
    candidates = pd.read_csv(input_root / "terrains" / terrain / "uav_candidate.csv")
    access = pd.read_csv(input_root / "primitives/by_terrain" / f"{terrain}_access_phy_edges.csv")
    links = pd.read_csv(input_root / "primitives/by_terrain" / f"{terrain}_link_edges.csv")
    options = pd.read_csv(input_root / "primitives/backhaul_options.csv")
    return (
        {"case_id": case_id, "terrain": terrain, "infrastructure_level": infrastructure, "seed": seed},
        demand,
        candidates,
        access,
        links,
        options,
    )


def initial_certification(
    *,
    case: dict[str, Any],
    selected: Sequence[str],
    method: str,
    demand: pd.DataFrame,
    access: pd.DataFrame,
    links: pd.DataFrame,
    options: pd.DataFrame,
    uav_capacity: float,
    gbs_capacity: float,
) -> CertificationResult:
    return certify_deployment_v2(
        case_id=case["case_id"],
        method=method,
        terrain=case["terrain"],
        selected_ids=selected,
        demand=demand,
        access_phy_edges=access,
        link_edges=links,
        backhaul_options=options,
        uav_capacity_mbps=uav_capacity,
        gbs_capacity_mbps=gbs_capacity,
        time_limit_seconds=5.0,
        mip_relative_gap=0.0,
        deterministic_second_stage=False,
        gap_tolerance=1.0e-6,
        lp_time_limit_seconds=5.0,
        run_milp=True,
        solver_threads=1,
        solver_seed=0,
        use_constructive_witness=True,
    )


def rebuild_result(case_id: str, state: CandidateState) -> CertificationResult:
    return certification_from_policy(
        model=state.model,
        policy=state.policy,
        case_id=case_id,
        method="CSD-MR-CTKS-candidate",
        nominal=state.built[5],
        physical=state.built[6],
        link=state.built[7],
        network=state.built[8],
    )


def run_outer_case_core(
    *,
    case_id: str,
    initial_selected: Sequence[str],
    initial_method: str,
    historical: pd.Series,
    stabilize_selected_diagnosis: bool = False,
    max_horizon: int = HORIZON,
    offered_load_mbps: float | None = None,
    candidate_total_milp_budget_seconds: float = TOTAL_MILP_BUDGET_SECONDS,
    candidate_stage_budget_seconds: float = STAGE_BUDGET_SECONDS,
    input_root: Path | None = None,
    candidate_policy: str = "CT-KS",
    candidate_certifier: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if max_horizon < 0:
        raise ValueError("max_horizon must be non-negative")
    if candidate_total_milp_budget_seconds <= 0.0:
        raise ValueError("candidate_total_milp_budget_seconds must be positive")
    if not 0.0 < candidate_stage_budget_seconds <= candidate_total_milp_budget_seconds:
        raise ValueError("candidate_stage_budget_seconds must be in (0, total budget]")
    case_started = perf_counter()
    case, demand, candidates, access, links, options = load_case(
        case_id, INPUTS if input_root is None else Path(input_root)
    )
    demand, native_total_demand_mbps, effective_total_demand_mbps = (
        rescale_demand_to_total(demand, offered_load_mbps)
    )
    cfg, _, _ = load_config(str(CONFIG))
    capacity = cfg["capacity"]
    uav_capacity = float(capacity["uav_aggregate_capacity_mbps"])
    gbs_capacity = float(capacity["gbs_egress_capacity_mbps"])
    initial_result = initial_certification(
        case=case,
        selected=initial_selected,
        method=initial_method,
        demand=demand,
        access=access,
        links=links,
        options=options,
        uav_capacity=uav_capacity,
        gbs_capacity=gbs_capacity,
    )
    initial_label, _, _ = dominant_attribution(initial_result.poi_attribution)
    initial = SearchNode(
        tuple(sorted(map(str, initial_selected))),
        initial_result,
        float(initial_result.exact_admission.normalized_lower_bound),
        float(initial_result.exact_admission.normalized_upper_bound),
        initial_label,
        (),
        0,
    )
    beam = [initial]
    archive = [initial]
    cert_cache: dict[tuple[str, ...], PolicyCacheEntry] = {}
    candidate_pool = candidates["candidate_id"].astype(str).tolist()
    trace_rows: list[dict[str, Any]] = []
    visits = safe_rejects = threshold_accepts = full_fallbacks = duplicates = 0
    duplicate_cache_accept = duplicate_cache_reject = duplicate_cache_refine = 0
    new_state_certify = unresolved_after_certification = 0
    full_domain_call_count = kernel_call_count = recovery_call_count = 0
    accounted_seconds = 0.0
    resolved_depths = attempted_depths = diagnosis_refinements = 0
    stop_reason = "horizon_reached"
    checkpoint_rows: list[dict[str, Any]] = []

    def capture_checkpoint(depth: int, status: str) -> None:
        best = sorted(archive, key=lambda node: (-node.lb, -node.ub, node.selected))[0]
        initial_set = set(initial.selected)
        best_set = set(best.selected)
        checkpoint_rows.append(
            {
                "case_id": case_id,
                "terrain": case["terrain"],
                "infrastructure_level": case["infrastructure_level"],
                "seed": case["seed"],
                "native_total_demand_mbps": native_total_demand_mbps,
                "offered_load_mbps": effective_total_demand_mbps,
                "depth_h": depth,
                "L_h": best.lb,
                "archived_state_UB": best.ub,
                "archived_selected_candidate_ids": ";".join(best.selected),
                "best_path_actions": ";".join(best.actions),
                "best_path_transition_count": len(best.actions),
                "state_distance_from_initial": len(initial_set.symmetric_difference(best_set)) / 2.0,
                "candidate_visits_cumulative": visits,
                "safe_rejections_cumulative": safe_rejects,
                "threshold_accepts_cumulative": threshold_accepts,
                "full_domain_recoveries_cumulative": full_fallbacks,
                "canonical_duplicates_cumulative": duplicates,
                "attempted_depths_cumulative": attempted_depths,
                "resolved_depths_cumulative": resolved_depths,
                "ctks_accounted_seconds_cumulative": accounted_seconds,
                "wall_seconds_cumulative": perf_counter() - case_started,
                "archive_size": len(archive),
                "checkpoint_status": status,
                "hierarchy_pass": bool(best.result.summary["hierarchy_pass"]),
                "max_capacity_violation_mbps": float(best.result.exact_admission.max_capacity_violation_mbps),
            }
        )

    # H=0 is freshly recertified under the same current certifier used by the trajectory.
    capture_checkpoint(0, "observed_initial_recertification")

    for depth in range(1, max_horizon + 1):
        pools = []
        orders = []
        for node in beam:
            pool = generate_operator_candidates(
                case_id=case_id,
                terrain=case["terrain"],
                selected_ids=node.selected,
                candidate_pool=candidate_pool,
                demand=demand,
                baseline=node.result,
                access_phy_edges=access,
                link_edges=links,
                backhaul_options=options,
                uav_capacity_mbps=uav_capacity,
                gbs_capacity_mbps=gbs_capacity,
                budget_per_operator=PROPOSALS_PER_OPERATOR,
            )
            preferred = FAILURE_TO_OPERATOR[node.diagnosis]
            pools.append(pool)
            orders.append((preferred, *tuple(op for op in OPS if op != preferred)))
        scheduled = schedule_proposals(beam, pools, orders)
        if not scheduled:
            stop_reason = "no_legal_proposals"
            break
        states: list[CandidateState] = []
        for evaluation_index, (parent_index, proposal) in enumerate(scheduled, start=1):
            parent = beam[parent_index]
            selected = tuple(sorted(proposal.selected_ids))
            tau = acceptance_threshold(parent)
            duplicate = selected in cert_cache
            if duplicate:
                duplicates += 1
            entry = cert_cache.get(selected)
            refined = False
            fair_timing: dict[str, Any] = {}
            if entry is None or cache_decision(entry, tau) == "insufficient":
                refined = True
                if entry is None:
                    new_state_certify += 1
                else:
                    duplicate_cache_refine += 1
                visits += 1
                build_started = perf_counter()
                built = build_service_chains(
                    case_id=case_id,
                    terrain=case["terrain"],
                    selected_ids=selected,
                    demand=demand,
                    access_phy_edges=access,
                    link_edges=links,
                    backhaul_options=options,
                )
                build_seconds = perf_counter() - build_started
                if candidate_certifier is None:
                    model = prepare_kernel_model(
                        poi_ids=built[0],
                        weights=built[1],
                        demands=built[2],
                        chains_by_poi=built[3],
                        edge_capacities=built[4],
                        uav_capacity_mbps=uav_capacity,
                        gbs_capacity_mbps=gbs_capacity,
                        lp_time_limit_seconds=5.0,
                    )
                    policy = run_policy(
                        model,
                        policy=candidate_policy,
                        threshold_normalized=parent.lb,
                        total_milp_budget_seconds=candidate_total_milp_budget_seconds,
                        stage_budget_seconds=candidate_stage_budget_seconds,
                        bucket_fraction=BUCKET_FRACTION,
                    )
                    accounted_seconds += policy.total_accounted_seconds
                else:
                    fair = candidate_certifier(
                        built=built,
                        uav_capacity_mbps=uav_capacity,
                        gbs_capacity_mbps=gbs_capacity,
                        tau=tau,
                        build_seconds=build_seconds,
                        total_budget_seconds=candidate_total_milp_budget_seconds,
                        stage_budget_seconds=candidate_stage_budget_seconds,
                    )
                    model = fair.model
                    policy = fair.policy
                    fair_timing = dict(fair.row)
                    accounted_seconds += float(fair_timing["runtime_total"])
                full_domain_call_count += int(policy.full_solves)
                kernel_call_count += int(policy.restricted_solves)
                recovery_call_count += int(policy.full_domain_fallback)
                full_fallbacks += int(policy.full_domain_fallback)
                result_for_cache: CertificationResult | None = None
                if policy.threshold_certified and not policy.safely_rejected:
                    state_for_result = CandidateState(
                        model=model,
                        policy=policy,
                        built=built,
                        selected=selected,
                        operator=proposal.operator,
                        candidate_out=proposal.candidate_out,
                        candidate_in=proposal.candidate_in,
                        parent_lb=parent.lb,
                    )
                    result_for_cache = rebuild_result(case_id, state_for_result)
                if entry is None:
                    entry = PolicyCacheEntry(
                        selected=selected,
                        model=model,
                        policy=policy,
                        built=built,
                        lb=float(policy.lower_bound),
                        ub=float(policy.full_domain_upper_bound),
                        result=result_for_cache,
                    )
                    cert_cache[selected] = entry
                else:
                    if policy.lower_bound > entry.lb + NUM_TOL:
                        entry.lb = float(policy.lower_bound)
                        entry.model = model
                        entry.policy = policy
                        entry.built = built
                        entry.result = result_for_cache
                    if policy.full_domain_upper_bound < entry.ub - NUM_TOL:
                        entry.ub = float(policy.full_domain_upper_bound)
                    if entry.lb > entry.ub + NUM_TOL:
                        raise RuntimeError("CT-KS duplicate evidence cache has LB above UB")
                policy = entry.policy
                built = entry.built
            else:
                assert entry is not None
                decision = cache_decision(entry, tau)
                if decision == "reject":
                    duplicate_cache_reject += 1
                elif decision == "accept":
                    duplicate_cache_accept += 1
                policy = entry.policy
                built = entry.built
            decision = cache_decision(entry, tau)
            safe_rejects += int(decision == "reject")
            if decision == "accept":
                threshold_accepts += 1
                state = CandidateState(
                    model=entry.model,
                    policy=policy,
                    built=built,
                    selected=selected,
                    operator=proposal.operator,
                    candidate_out=proposal.candidate_out,
                    candidate_in=proposal.candidate_in,
                    parent_lb=parent.lb,
                    result=entry.result,
                )
                if state.result is None:
                    state.result = rebuild_result(case_id, state)
                    entry.result = state.result
                states.append(state)
            elif refined:
                unresolved_after_certification += int(decision == "insufficient")
            trace_rows.append(
                {
                    "case_id": case_id,
                    "depth": depth,
                    "evaluation_index": evaluation_index,
                    "parent_LB": parent.lb,
                    "parent_threshold": tau,
                    "parent_selected_candidate_ids": ";".join(parent.selected),
                    "parent_action_sequence": ";".join(parent.actions),
                    "operator": proposal.operator,
                    "candidate_out": proposal.candidate_out,
                    "candidate_in": proposal.candidate_in,
                    "selected_candidate_ids": ";".join(selected),
                    "status": (
                        "duplicate_cache_accept" if duplicate and decision == "accept" and not refined
                        else "duplicate_cache_reject" if duplicate and decision == "reject" and not refined
                        else "duplicate_cache_unresolved" if duplicate and decision == "insufficient"
                        else policy.status
                    ),
                    "candidate_LB": entry.lb,
                    "candidate_UB": entry.ub,
                    "threshold": tau,
                    "threshold_certified": decision == "accept",
                    "unresolved": decision == "insufficient",
                    "safe_rejected": decision == "reject",
                    "duplicate_proposal": duplicate,
                    "certification_refined": refined,
                    "full_fallback": policy.full_domain_fallback,
                    "kernel_fraction": policy.kernel_fraction,
                    "initial_kernel_size": policy.initial_kernel_size,
                    "final_kernel_size": policy.final_kernel_size,
                    "full_variable_count": policy.full_variable_count,
                    "restricted_solves": policy.restricted_solves,
                    "full_solves": policy.full_solves,
                    "milp_seconds": policy.milp_seconds,
                    "preparation_seconds": policy.preparation_seconds,
                    "accounted_seconds": policy.total_accounted_seconds,
                    "solver_status_sequence": ";".join(
                        str(item["solver_status"]) for item in policy.solve_history
                    ),
                    "solver_node_count_total": sum(
                        int(item["mip_node_count"] or 0) for item in policy.solve_history
                    ),
                    "solver_max_mip_gap": max(
                        [float(item["mip_gap"]) for item in policy.solve_history
                         if item["mip_gap"] is not None],
                        default=float("nan"),
                    ),
                    "solve_history_json": json.dumps(
                        policy.solve_history, ensure_ascii=False, sort_keys=True
                    ),
                    "runtime_prepare": float(fair_timing.get("runtime_prepare", float("nan"))),
                    "runtime_lp": float(fair_timing.get("runtime_lp", float("nan"))),
                    "runtime_witness": float(fair_timing.get("runtime_witness", float("nan"))),
                    "runtime_kernel": float(fair_timing.get("runtime_kernel", float("nan"))),
                    "runtime_milp": float(fair_timing.get("runtime_milp", float("nan"))),
                    "runtime_total": float(fair_timing.get("runtime_total", float("nan"))),
                    "budget_overrun_seconds": float(
                        fair_timing.get("budget_overrun_seconds", float("nan"))
                    ),
                    "nominal_budget": float(
                        fair_timing.get("nominal_budget", float("nan"))
                    ),
                    "budget_overrun": float(
                        fair_timing.get("budget_overrun", float("nan"))
                    ),
                    "overrun_flag": fair_timing.get("overrun_flag", None),
                    "overrun_ratio": float(
                        fair_timing.get("overrun_ratio", float("nan"))
                    ),
                    "decision_stage": fair_timing.get("decision_stage", "legacy"),
                    "mip_rel_gap": float(
                        fair_timing.get("mip_rel_gap", policy.mip_relative_gap)
                    ),
                    "kernel_constructed": bool(
                        fair_timing.get("kernel_constructed", policy.initial_kernel_size > 0)
                    ),
                }
            )
        if not states:
            stop_reason = "no_threshold_certified_child"
            break

        attempted_depths += 1
        states.sort(key=lambda state: (-state.policy.lower_bound, -state.policy.full_domain_upper_bound, state.selected))
        if len(states) > BEAM_WIDTH:
            cutoff = states[BEAM_WIDTH - 1].policy.lower_bound
            ambiguous = [
                state
                for state in states
                if state.policy.full_domain_upper_bound + NUM_TOL >= cutoff
                and not state.policy.full_domain_fallback
            ]
            for state in ambiguous:
                before = state.policy.total_accounted_seconds
                before_full_solves = state.policy.full_solves
                before_recovery = state.policy.full_domain_fallback
                state.policy = refine_for_beam_ranking(
                    state.model,
                    state.policy,
                    total_milp_budget_seconds=candidate_total_milp_budget_seconds,
                )
                accounted_seconds += state.policy.total_accounted_seconds - before
                full_domain_call_count += state.policy.full_solves - before_full_solves
                recovery_call_count += int(
                    state.policy.full_domain_fallback and not before_recovery
                )
                full_fallbacks += int(state.policy.full_domain_fallback)
                state.result = rebuild_result(case_id, state)
            states.sort(key=lambda state: (-state.policy.lower_bound, -state.policy.full_domain_upper_bound, state.selected))
            selected_states = states[:BEAM_WIDTH]
            unselected_states = states[BEAM_WIDTH:]
            rank_resolved = bool(
                min(state.policy.lower_bound for state in selected_states) + NUM_TOL
                >= max(state.policy.full_domain_upper_bound for state in unselected_states)
            )
        else:
            selected_states = states
            rank_resolved = True

        if stabilize_selected_diagnosis:
            # A threshold witness is sufficient for acceptance but may be too
            # weak to define the APP-failure composition used at the next
            # search depth.  Only states that actually enter the beam spend
            # their remaining per-candidate budget on complete-domain witness
            # refinement.  No discarded candidate receives extra budget.
            for state in list(selected_states):
                if state.policy.safely_rejected or state.policy.full_domain_fallback:
                    continue
                before = state.policy.total_accounted_seconds
                before_full_solves = state.policy.full_solves
                before_recovery = state.policy.full_domain_fallback
                state.policy = refine_for_beam_ranking(
                    state.model,
                    state.policy,
                    total_milp_budget_seconds=candidate_total_milp_budget_seconds,
                )
                accounted_seconds += state.policy.total_accounted_seconds - before
                full_domain_call_count += state.policy.full_solves - before_full_solves
                recovery_call_count += int(
                    state.policy.full_domain_fallback and not before_recovery
                )
                full_fallbacks += int(state.policy.full_domain_fallback)
                diagnosis_refinements += int(state.policy.full_domain_fallback)
                state.result = rebuild_result(case_id, state)
            states.sort(key=lambda state: (-state.policy.lower_bound, -state.policy.full_domain_upper_bound, state.selected))
            selected_states = states[:BEAM_WIDTH]
            if len(states) > BEAM_WIDTH:
                cutoff = min(state.policy.lower_bound for state in selected_states)
                remaining_ambiguous = [
                    state
                    for state in states[BEAM_WIDTH:]
                    if state.policy.full_domain_upper_bound + NUM_TOL >= cutoff
                    and not state.policy.full_domain_fallback
                ]
                for state in remaining_ambiguous:
                    before = state.policy.total_accounted_seconds
                    before_full_solves = state.policy.full_solves
                    before_recovery = state.policy.full_domain_fallback
                    state.policy = refine_for_beam_ranking(
                        state.model,
                        state.policy,
                        total_milp_budget_seconds=candidate_total_milp_budget_seconds,
                    )
                    accounted_seconds += state.policy.total_accounted_seconds - before
                    full_domain_call_count += state.policy.full_solves - before_full_solves
                    recovery_call_count += int(
                        state.policy.full_domain_fallback and not before_recovery
                    )
                    full_fallbacks += int(state.policy.full_domain_fallback)
                    state.result = rebuild_result(case_id, state)
                states.sort(key=lambda state: (-state.policy.lower_bound, -state.policy.full_domain_upper_bound, state.selected))
                selected_states = states[:BEAM_WIDTH]
                unselected_states = states[BEAM_WIDTH:]
                rank_resolved = bool(
                    min(state.policy.lower_bound for state in selected_states) + NUM_TOL
                    >= max(state.policy.full_domain_upper_bound for state in unselected_states)
                )
            else:
                rank_resolved = True
        resolved_depths += int(rank_resolved)

        children: list[SearchNode] = []
        for state in states:
            if state.result is None:
                continue
            label, _, _ = dominant_attribution(state.result.poi_attribution)
            parent_action = f"{state.operator}:{state.candidate_out}->{state.candidate_in}"
            # The exact parent action history is only needed for the selected beam.
            parent = next(
                node for node in beam
                if abs(node.lb - state.parent_lb) <= NUM_TOL
                and set(state.selected) == (set(node.selected) - {state.candidate_out}) | {state.candidate_in}
            )
            child = SearchNode(
                state.selected,
                state.result,
                state.policy.lower_bound,
                state.policy.full_domain_upper_bound,
                label,
                parent.actions + (parent_action,),
                depth,
            )
            children.append(child)
        children.sort(key=lambda node: (-node.lb, -node.ub, node.selected))
        beam = children[:BEAM_WIDTH]
        archive.extend(children)
        capture_checkpoint(depth, "observed")

    # A stopped search defines the same archived best for every larger allowed depth.
    while len(checkpoint_rows) < max_horizon + 1:
        capture_checkpoint(len(checkpoint_rows), "carried_forward_after_stop")

    final = sorted(archive, key=lambda node: (-node.lb, -node.ub, node.selected))[0]
    final_difference = final.lb - float(historical["APP_LB"])
    time_reduction = 1.0 - accounted_seconds / max(float(historical["search_wall_seconds"]), 1.0e-12)
    case_row = {
        "case_id": case_id,
        "terrain": case["terrain"],
        "infrastructure_level": case["infrastructure_level"],
        "initial_method": initial_method,
        "initial_selected_candidate_ids": ";".join(sorted(initial_selected)),
        "initial_LB_new_run": initial.lb,
        "historical_BestStrong_LB": float(historical["initial_BestStrong_LB"]),
        "CTKS_final_LB": final.lb,
        "CTKS_final_UB": final.ub,
        "historical_CSD_MR_LB": float(historical["APP_LB"]),
        "final_LB_difference": final_difference,
        "CTKS_search_accounted_seconds": accounted_seconds,
        "historical_search_wall_seconds": float(historical["search_wall_seconds"]),
        "search_time_reduction_fraction": time_reduction,
        "candidate_visits": visits,
        "safe_rejects": safe_rejects,
        "threshold_accepts": threshold_accepts,
        "full_fallbacks": full_fallbacks,
        "diagnosis_refinements": diagnosis_refinements,
        "stabilize_selected_diagnosis": stabilize_selected_diagnosis,
        "canonical_duplicates": duplicates,
        "duplicate_cache_accept": duplicate_cache_accept,
        "duplicate_cache_reject": duplicate_cache_reject,
        "duplicate_cache_refine": duplicate_cache_refine,
        "new_state_certify": new_state_certify,
        "unresolved_after_certification": unresolved_after_certification,
        "full_domain_call_count": full_domain_call_count,
        "kernel_call_count": kernel_call_count,
        "recovery_call_count": recovery_call_count,
        "candidate_policy": candidate_policy,
        "attempted_depths": attempted_depths,
        "resolved_depths": resolved_depths,
        "beam_resolution_rate": resolved_depths / max(attempted_depths, 1),
        "final_selected_candidate_ids": ";".join(final.selected),
        "final_actions": ";".join(final.actions),
        "stop_reason": stop_reason,
        "hierarchy_pass": bool(final.result.summary["hierarchy_pass"]),
        "max_capacity_violation_mbps": float(final.result.exact_admission.max_capacity_violation_mbps),
        "candidate_total_milp_budget_seconds": float(candidate_total_milp_budget_seconds),
        "candidate_stage_budget_seconds": float(candidate_stage_budget_seconds),
    }
    return case_row, trace_rows, checkpoint_rows


def run_outer_case(
    *,
    case_id: str,
    initial_selected: Sequence[str],
    initial_method: str,
    historical: pd.Series,
    stabilize_selected_diagnosis: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Backward-compatible formal H=3 entry point."""
    case_row, trace_rows, _ = run_outer_case_core(
        case_id=case_id,
        initial_selected=initial_selected,
        initial_method=initial_method,
        historical=historical,
        stabilize_selected_diagnosis=stabilize_selected_diagnosis,
        max_horizon=HORIZON,
        offered_load_mbps=None,
    )
    return case_row, trace_rows


def summarize(rows: pd.DataFrame, elapsed: float) -> dict[str, Any]:
    violations = {
        "illegal_witness": int((rows["max_capacity_violation_mbps"] > 1.0e-8).sum()),
        "hierarchy_violation": int((~rows["hierarchy_pass"].astype(bool)).sum()),
        "final_below_new_initial": int((rows["CTKS_final_LB"] + NUM_TOL < rows["initial_LB_new_run"]).sum()),
    }
    correctness = all(value == 0 for value in violations.values())
    metrics = {
        "case_count": int(len(rows)),
        "median_final_LB_difference_vs_historical": float(rows["final_LB_difference"].median()),
        "mean_final_LB_difference_vs_historical": float(rows["final_LB_difference"].mean()),
        "material_final_LB_loss_rate_1e_4": float((rows["final_LB_difference"] < -1.0e-4).mean()),
        "positive_final_LB_rate": float((rows["final_LB_difference"] > 1.0e-10).mean()),
        "median_search_time_reduction": float(rows["search_time_reduction_fraction"].median()),
        "overall_beam_resolution_rate": float(rows["resolved_depths"].sum() / max(rows["attempted_depths"].sum(), 1)),
        "total_candidate_visits": int(rows["candidate_visits"].sum()),
        "full_fallback_rate_per_visit": float(rows["full_fallbacks"].sum() / max(rows["candidate_visits"].sum(), 1)),
    }
    effectiveness = {
        "median_final_LB": metrics["median_final_LB_difference_vs_historical"] >= -1.0e-4,
        "material_LB_loss_rate": metrics["material_final_LB_loss_rate_1e_4"] <= 0.20,
        "time_reduction": metrics["median_search_time_reduction"] >= 0.20,
        "beam_resolution": metrics["overall_beam_resolution_rate"] >= 0.50,
    }
    return {
        "status": "PASS" if correctness and all(effectiveness.values()) else "FAIL",
        "correctness_pass": correctness,
        "correctness_violations": violations,
        "effectiveness_components": effectiveness,
        "metrics": metrics,
        "elapsed_wall_seconds": elapsed,
        "scope": "seed-0 development only; no confirmatory claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "frozen_protocol.json"
    frozen = protocol()
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != frozen:
            raise RuntimeError("Outer CT-KS protocol mismatch")
    else:
        write_json(protocol_path, frozen)

    formal = pd.read_csv(FORMAL / "formal_case_method_results.csv")
    historical_csd = formal[(formal["seed"] == 0) & formal["method"].eq("CSD-MR")].copy()
    best = formal[(formal["seed"] == 0) & formal["method"].eq("BestStrong")].copy()
    cases = sorted(historical_csd["case_id"].tolist())
    if args.case:
        cases = [args.case]
    started = perf_counter()
    case_rows = []
    traces = []
    for case_id in cases:
        historical = historical_csd[historical_csd["case_id"].eq(case_id)].iloc[0]
        initial = best[best["case_id"].eq(case_id)].iloc[0]
        row, trace = run_outer_case(
            case_id=case_id,
            initial_selected=str(initial["selected_candidate_ids"]).split(";"),
            initial_method=str(initial["source_method"]),
            historical=historical,
        )
        case_rows.append(row)
        traces.extend(trace)
        pd.DataFrame(case_rows).to_csv(OUT / "outer_case_results_checkpoint.csv", index=False)
        pd.DataFrame(traces).to_csv(OUT / "outer_search_trace_checkpoint.csv", index=False)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    rows = pd.DataFrame(case_rows)
    suffix = "full" if not args.case else "smoke"
    rows.to_csv(OUT / f"outer_case_results_{suffix}.csv", index=False)
    pd.DataFrame(traces).to_csv(OUT / f"outer_search_trace_{suffix}.csv", index=False)
    result = summarize(rows, perf_counter() - started)
    write_json(OUT / f"outer_gate_result_{suffix}.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
