"""Formal deployment search under the frozen 50 m demand model.

The runner keeps every historical experiment read-only.  It evaluates seven
independently generated starting deployments, derives a clearly labelled
post-hoc BestStrong comparator, and applies the existing diagnosis-guided,
fully recertified CSD-MR search to that incumbent.  Candidate visits, MILP
calls, numerical checks, and per-case checkpoints are retained.

KMeans-50m and Fiedler-50m are transparent reimplementations because the old
archived results did not retain executable source for those labels.  They are
therefore never described as native-code reproductions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protocol_stack_pipeline_common import load_config  # noqa: E402
from protocol_stack_v2_certifier import CertificationResult, certify_deployment_v2  # noqa: E402
from protocol_stack_v2_exp6a import (  # noqa: E402
    FAILURE_TO_OPERATOR,
    dominant_attribution,
    generate_operator_candidates,
)
from openbase_adapters import (  # noqa: E402
    adapt_flying_relay,
    adapt_prm_abs,
    adapt_urban_ga,
    normalize,
    weighted_candidate_features,
    weighted_overlap_matrix,
)
from run_protocol_stack_exp5_service_fragmentation_certifier import (  # noqa: E402
    coverage_first_selection,
)


BASE = ROOT / "data/formal_inputs"
INPUTS = BASE / "inputs"
OUT = ROOT / "outputs/formal_search"
WORK = ROOT / "outputs/_formal_search_work"
REFERENCE_MODE = False
CONFIG = ROOT / "configs/protocol_stack_csd_mr_go_v1.yaml"
TERRAINS = ("S1", "S2", "S3", "S4")
LEVELS = ("low", "medium", "high")
SEEDS = tuple(range(5))
K = 8
INITIAL_METHODS = (
    "Random",
    "KMeans-50m",
    "Fiedler-50m",
    "Coverage-first",
    "PRM-ABS",
    "UrbanInfrastructure-GA",
    "FlyingRelay",
)
STRONG_POOL = tuple(method for method in INITIAL_METHODS if method != "Random")
OPS = ("O1", "O2", "O3")
HORIZON = 3
BEAM_WIDTH = 2
PROPOSALS_PER_OPERATOR = 4
VISITS_PER_DEPTH = 4
TIME_LIMIT_SECONDS = 5.0
TOL = 1e-6
NUM_TOL = 1e-10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def weighted_kmeans_selection(
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    k: int,
    seed: int,
    max_iter: int = 100,
) -> list[str]:
    points = demand[["x_m", "y_m"]].to_numpy(float)
    weights = demand["service_demand_weight"].to_numpy(float)
    weights = weights / max(float(weights.sum()), 1e-12)
    rng = np.random.default_rng(seed)
    centers = [points[int(rng.choice(len(points), p=weights))]]
    for _ in range(1, k):
        dist2 = np.min(
            np.stack([np.sum((points - center) ** 2, axis=1) for center in centers], axis=1),
            axis=1,
        )
        probs = weights * dist2
        if float(probs.sum()) <= 1e-15:
            index = int(rng.choice(len(points), p=weights))
        else:
            index = int(rng.choice(len(points), p=probs / probs.sum()))
        centers.append(points[index])
    centers_array = np.asarray(centers, dtype=float)
    labels = np.zeros(len(points), dtype=int)
    for _ in range(max_iter):
        distance = np.sum((points[:, None, :] - centers_array[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distance, axis=1)
        new_centers = centers_array.copy()
        for cluster in range(k):
            mask = new_labels == cluster
            if mask.any():
                new_centers[cluster] = np.average(points[mask], axis=0, weights=weights[mask])
            else:
                farthest = int(np.argmax(np.min(distance, axis=1) * weights))
                new_centers[cluster] = points[farthest]
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers_array, atol=1e-7):
            centers_array = new_centers
            break
        labels, centers_array = new_labels, new_centers
    candidate_xy = candidates[["x_m", "y_m"]].to_numpy(float)
    cost = np.sum((centers_array[:, None, :] - candidate_xy[None, :, :]) ** 2, axis=2)
    rows, columns = linear_sum_assignment(cost)
    chosen = [str(candidates.iloc[int(columns[np.where(rows == row)[0][0]])]["candidate_id"]) for row in range(k)]
    return chosen


def _spectral_split(indices: np.ndarray, adjacency: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(indices) < 2:
        return indices, np.array([], dtype=int)
    sub = adjacency[np.ix_(indices, indices)]
    degree = sub.sum(axis=1)
    laplacian = np.diag(degree) - sub
    try:
        values, vectors = np.linalg.eigh(laplacian)
        fiedler = vectors[:, 1] if len(values) > 1 else np.zeros(len(indices))
        threshold = float(np.median(fiedler))
        left = indices[fiedler <= threshold]
        right = indices[fiedler > threshold]
    except np.linalg.LinAlgError:
        left = right = np.array([], dtype=int)
    if len(left) == 0 or len(right) == 0:
        local_xy = xy[indices]
        axis = int(np.argmax(np.ptp(local_xy, axis=0)))
        ordered = indices[np.argsort(local_xy[:, axis], kind="stable")]
        midpoint = max(1, len(ordered) // 2)
        left, right = ordered[:midpoint], ordered[midpoint:]
    return left, right


def fiedler_selection(
    terrain: str,
    case_id: str,
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    access: pd.DataFrame,
    links: pd.DataFrame,
    options: pd.DataFrame,
    k: int,
) -> list[str]:
    candidate_ids = candidates["candidate_id"].astype(str).tolist()
    index = {candidate_id: i for i, candidate_id in enumerate(candidate_ids)}
    adjacency = np.zeros((len(candidate_ids), len(candidate_ids)), dtype=float)
    relay = links[
        links["link_type"].eq("relay")
        & links["link_feasible"].astype(bool)
    ]
    for row in relay.itertuples(index=False):
        left, right = str(row.source_id), str(row.target_id)
        if left in index and right in index and left != right:
            weight = max(float(row.link_rate_mbps), 1e-9)
            i, j = index[left], index[right]
            adjacency[i, j] = max(adjacency[i, j], weight)
            adjacency[j, i] = max(adjacency[j, i], weight)
    positive = adjacency[adjacency > 0]
    if positive.size:
        adjacency /= float(positive.max())
    xy = candidates[["x_m", "y_m"]].to_numpy(float)
    groups = [np.arange(len(candidate_ids), dtype=int)]
    while len(groups) < k:
        splittable = [i for i, group in enumerate(groups) if len(group) >= 2]
        if not splittable:
            break
        chosen_group = max(splittable, key=lambda i: (len(groups[i]), float(adjacency[np.ix_(groups[i], groups[i])].sum())))
        group = groups.pop(chosen_group)
        left, right = _spectral_split(group, adjacency, xy)
        groups.extend([left, right])
    demand_score, network_score, relay_score, _ = weighted_candidate_features(
        terrain, case_id, candidate_ids, access, options, demand
    )
    score = 0.65 * normalize(demand_score) + 0.25 * normalize(network_score) + 0.10 * normalize(relay_score)
    selected: list[int] = []
    for group in groups[:k]:
        winner = sorted(group.tolist(), key=lambda i: (-float(score[i]), candidate_ids[i]))[0]
        selected.append(winner)
    if len(selected) < k:
        remaining = [i for i in range(len(candidate_ids)) if i not in selected]
        remaining.sort(key=lambda i: (-float(score[i]), candidate_ids[i]))
        selected.extend(remaining[: k - len(selected)])
    return [candidate_ids[i] for i in selected[:k]]


def reason_fractions(result: CertificationResult) -> dict[str, float | int]:
    table = result.poi_attribution
    total_weight = float(table["service_demand_weight"].sum())
    grouped_weight = table.groupby("first_blocking_reason")["service_demand_weight"].sum()
    grouped_count = table.groupby("first_blocking_reason").size()
    row: dict[str, float | int] = {}
    for reason in ("fail_nominal", "fail_phy", "fail_link", "fail_net", "fail_app", "served"):
        row[f"weight_fraction_{reason}"] = float(grouped_weight.get(reason, 0.0) / total_weight) if total_weight > 0 else 0.0
        row[f"unit_count_{reason}"] = int(grouped_count.get(reason, 0))
    return row


def certification_row(
    result: CertificationResult,
    *,
    case: dict[str, Any],
    method: str,
    selected: Sequence[str],
    source_method: str,
    wall_seconds: float,
    cache_reused: bool,
) -> dict[str, Any]:
    summary = result.summary
    exact = result.exact_admission
    return {
        "case_id": case["case_id"],
        "terrain": case["terrain"],
        "infrastructure_level": case["infrastructure_level"],
        "seed": int(case["seed"]),
        "analysis_role": case["analysis_role"],
        "method": method,
        "source_method": source_method,
        "selected_candidate_ids": ";".join(sorted(map(str, selected))),
        "K_selected": len(set(selected)),
        "C_nom": float(summary["C_nom"]),
        "PHY": float(summary["PHY"]),
        "LINK": float(summary["LINK"]),
        "NET": float(summary["NET"]),
        "APP_LB": float(summary["APP_LB"]),
        "APP_UB": float(summary["APP_UB"]),
        "hierarchy_pass": bool(summary["hierarchy_pass"]),
        "served_demand_unit_count": int(summary["served_poi_count"]),
        "demand_unit_count": int(summary["poi_count"]),
        "certification_status": str(exact.certification_status),
        "bound_valid": bool(exact.bound_valid),
        "solver_proven_optimal": bool(exact.solver_proven_optimal),
        "solver_has_incumbent": bool(exact.solver_has_incumbent),
        "external_witness_available": bool(exact.external_feasible_LB_available),
        "safe_pruned_by_valid_bound": bool(exact.safe_pruned_by_valid_bound),
        "solver_call_performed": bool(exact.solver_call_performed),
        "max_capacity_violation_mbps": float(exact.max_capacity_violation_mbps),
        "num_serviceable_units": int(exact.num_serviceable_users),
        "num_path_variables": int(exact.num_path_variables),
        "raw_path_variable_count": int(exact.raw_path_variable_count),
        "num_binary_variables": int(exact.num_binary_variables),
        "num_constraints": int(exact.num_constraints),
        "num_nonzeros": int(exact.num_nonzeros),
        "APP_solver_runtime_seconds": float(exact.runtime_seconds),
        "full_certification_wall_seconds": float(wall_seconds),
        "cache_reused": bool(cache_reused),
        **reason_fractions(result),
    }


def certify(
    *,
    case: dict[str, Any],
    method: str,
    selected: Sequence[str],
    demand: pd.DataFrame,
    access: pd.DataFrame,
    links: pd.DataFrame,
    options: pd.DataFrame,
    uav_capacity: float,
    gbs_capacity: float,
    safe_prune_below: float | None = None,
) -> tuple[CertificationResult, float]:
    started = time.perf_counter()
    result = certify_deployment_v2(
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
        time_limit_seconds=TIME_LIMIT_SECONDS,
        mip_relative_gap=0.0,
        deterministic_second_stage=False,
        gap_tolerance=1e-6,
        lp_time_limit_seconds=TIME_LIMIT_SECONDS,
        run_milp=True,
        solver_threads=1,
        solver_seed=0,
        safe_prune_below_normalized=safe_prune_below,
        use_constructive_witness=True,
        explicit_x_variables=False,
        add_cover_inequalities=False,
    )
    return result, time.perf_counter() - started


@dataclass
class SearchNode:
    selected: tuple[str, ...]
    result: CertificationResult
    lb: float
    ub: float
    diagnosis: str
    actions: tuple[str, ...]
    depth: int


@dataclass
class EvidenceCacheEntry:
    selected: tuple[str, ...]
    lb: float
    ub: float
    lb_result: CertificationResult
    ub_result: CertificationResult
    certifications: int = 1


def acceptance_threshold(parent: SearchNode) -> float:
    return float(parent.lb) - NUM_TOL


def merge_evidence(
    cache: dict[tuple[str, ...], EvidenceCacheEntry],
    selected: tuple[str, ...],
    result: CertificationResult,
) -> EvidenceCacheEntry:
    exact = result.exact_admission
    lb = float(exact.normalized_lower_bound)
    ub = float(exact.normalized_upper_bound)
    if lb > ub + NUM_TOL:
        raise RuntimeError(f"Invalid evidence interval for {selected}: LB={lb}, UB={ub}")
    if selected not in cache:
        entry = EvidenceCacheEntry(selected, lb, ub, result, result)
        cache[selected] = entry
        return entry
    entry = cache[selected]
    if lb > entry.lb:
        entry.lb = lb
        entry.lb_result = result
    if ub < entry.ub:
        entry.ub = ub
        entry.ub_result = result
    entry.certifications += 1
    if entry.lb > entry.ub + NUM_TOL:
        raise RuntimeError(
            f"Merged evidence interval is invalid for {selected}: LB={entry.lb}, UB={entry.ub}"
        )
    return entry


def cached_evidence_decision(entry: EvidenceCacheEntry, tau: float) -> str:
    if entry.ub < tau:
        return "reject"
    if entry.lb >= tau:
        return "accept"
    return "insufficient"


def cached_result_for_acceptance(entry: EvidenceCacheEntry) -> CertificationResult:
    return entry.lb_result


def schedule_proposals(
    beam: list[SearchNode],
    pools: list[dict[str, list[Any]]],
    orders: list[tuple[str, ...]],
) -> list[tuple[int, Any]]:
    scheduled: list[tuple[int, Any]] = []
    round_index = 0
    max_rounds = PROPOSALS_PER_OPERATOR * 3
    while len(scheduled) < VISITS_PER_DEPTH and round_index < max_rounds:
        for node_index, (_, pool, order) in enumerate(zip(beam, pools, orders)):
            operator = order[round_index % len(order)]
            rank = round_index // len(order)
            if rank < len(pool[operator]):
                scheduled.append((node_index, pool[operator][rank]))
                if len(scheduled) >= VISITS_PER_DEPTH:
                    break
        round_index += 1
    return scheduled


def run_csd_mr(
    *,
    case: dict[str, Any],
    initial_method: str,
    initial_selected: Sequence[str],
    initial_result: CertificationResult,
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    access: pd.DataFrame,
    links: pd.DataFrame,
    options: pd.DataFrame,
    uav_capacity: float,
    gbs_capacity: float,
    reference_mode: bool = False,
) -> tuple[SearchNode, list[dict[str, Any]], dict[str, int | float | str]]:
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
    cert_cache: dict[tuple[str, ...], EvidenceCacheEntry] = {
        initial.selected: EvidenceCacheEntry(
            initial.selected,
            initial.lb,
            initial.ub,
            initial.result,
            initial.result,
        )
    }
    trace: list[dict[str, Any]] = []
    visits = milp_calls = safe_prunes = accepted = 0
    duplicate_proposals = duplicate_cache_accept = duplicate_cache_reject = 0
    duplicate_cache_refine = new_state_certify = unresolved_after_certification = 0
    total_wall = 0.0
    candidate_pool = candidates["candidate_id"].astype(str).tolist()
    stop_reason = "horizon_reached"
    for depth in range(1, HORIZON + 1):
        pools: list[dict[str, list[Any]]] = []
        orders: list[tuple[str, ...]] = []
        for node in beam:
            pool = generate_operator_candidates(
                case_id=case["case_id"],
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
        children: list[SearchNode] = []
        for evaluation_index, (parent_index, proposal) in enumerate(scheduled, 1):
            parent = beam[parent_index]
            selected = tuple(sorted(proposal.selected_ids))
            tau = acceptance_threshold(parent)
            duplicate = selected in cert_cache
            if duplicate:
                duplicate_proposals += 1
            entry = cert_cache.get(selected)
            wall = 0.0
            result: CertificationResult | None = None
            safe = False
            refined = False
            if reference_mode:
                visits += 1
                result, wall = certify(
                    case=case,
                    method="CSD-MR-candidate",
                    selected=selected,
                    demand=demand,
                    access=access,
                    links=links,
                    options=options,
                    uav_capacity=uav_capacity,
                    gbs_capacity=gbs_capacity,
                    safe_prune_below=tau,
                )
                total_wall += wall
                exact = result.exact_admission
                milp_calls += int(bool(exact.solver_call_performed))
                safe = bool(exact.safe_pruned_by_valid_bound)
                safe_prunes += int(safe)
                lb = float(exact.normalized_lower_bound)
                ub = float(exact.normalized_upper_bound)
                if lb > ub + NUM_TOL:
                    raise RuntimeError(f"Invalid evidence interval for {selected}: LB={lb}, UB={ub}")
                entry = EvidenceCacheEntry(selected, lb, ub, result, result)
                refined = True
                status_prefix = "reference_certify"
            elif entry is None:
                visits += 1
                new_state_certify += int(entry is None)
                result, wall = certify(
                    case=case,
                    method="CSD-MR-candidate",
                    selected=selected,
                    demand=demand,
                    access=access,
                    links=links,
                    options=options,
                    uav_capacity=uav_capacity,
                    gbs_capacity=gbs_capacity,
                    safe_prune_below=tau,
                )
                total_wall += wall
                exact = result.exact_admission
                milp_calls += int(bool(exact.solver_call_performed))
                safe = bool(exact.safe_pruned_by_valid_bound)
                safe_prunes += int(safe)
                entry = merge_evidence(cert_cache, selected, result)
                refined = True
                status_prefix = "new_state_certify"
            elif cached_evidence_decision(entry, tau) == "reject":
                duplicate_cache_reject += 1
                status_prefix = "duplicate_cache_reject"
            elif cached_evidence_decision(entry, tau) == "accept":
                duplicate_cache_accept += 1
                status_prefix = "duplicate_cache_accept"
            else:
                duplicate_cache_refine += 1
                visits += 1
                result, wall = certify(
                    case=case,
                    method="CSD-MR-candidate",
                    selected=selected,
                    demand=demand,
                    access=access,
                    links=links,
                    options=options,
                    uav_capacity=uav_capacity,
                    gbs_capacity=gbs_capacity,
                    safe_prune_below=tau,
                )
                total_wall += wall
                exact = result.exact_admission
                milp_calls += int(bool(exact.solver_call_performed))
                safe = bool(exact.safe_pruned_by_valid_bound)
                safe_prunes += int(safe)
                entry = merge_evidence(cert_cache, selected, result)
                refined = True
                status_prefix = "duplicate_cache_refine"

            lb = float(entry.lb)
            ub = float(entry.ub)
            result_for_validity = result if result is not None else entry.lb_result
            exact_for_validity = result_for_validity.exact_admission
            valid = bool(
                exact_for_validity.bound_valid
                and lb <= ub + NUM_TOL
                and float(exact_for_validity.max_capacity_violation_mbps) <= 1e-8
                and result_for_validity.summary["hierarchy_pass"]
            )
            accept = bool(valid and lb >= tau)
            if not valid:
                status = "invalid_certification"
            elif accept:
                status = (
                    status_prefix if status_prefix == "duplicate_cache_accept"
                    else f"{status_prefix}_accepted"
                )
            elif ub < tau:
                status = (
                    status_prefix if status_prefix == "duplicate_cache_reject"
                    else "safe_pruned_by_valid_bound"
                )
            else:
                unresolved_after_certification += int(refined)
                status = (
                    "unresolved_after_certification"
                    if refined
                    else "duplicate_cache_unresolved"
                )
            if accept:
                accepted += 1
                accepted_result = cached_result_for_acceptance(entry)
                label, _, _ = dominant_attribution(accepted_result.poi_attribution)
                child = SearchNode(
                    selected, accepted_result, lb, ub, label,
                    parent.actions + (f"{proposal.operator}:{proposal.candidate_out}->{proposal.candidate_in}",),
                    depth,
                )
                children.append(child)
            trace.append(
                {
                    "case_id": case["case_id"], "depth": depth, "evaluation_index": evaluation_index,
                    "parent_LB": parent.lb, "parent_threshold": tau, "operator": proposal.operator,
                    "candidate_out": proposal.candidate_out, "candidate_in": proposal.candidate_in,
                    "selected_candidate_ids": ";".join(selected), "status": status,
                    "candidate_LB": lb, "candidate_UB": ub,
                    "solver_call_performed": bool(
                        result.exact_admission.solver_call_performed
                    ) if result is not None else False,
                    "safe_pruned": bool(ub < tau),
                    "accepted": accept,
                    "duplicate_proposal": duplicate,
                    "cache_certifications": int(entry.certifications),
                    "certification_refined": refined,
                    "wall_seconds": wall,
                }
            )
        if not children:
            stop_reason = "no_accepted_transition"
            break
        children.sort(key=lambda node: (-node.lb, -node.ub, node.selected))
        beam = children[:BEAM_WIDTH]
        archive.extend(children)
    final = sorted(archive, key=lambda node: (-node.lb, -node.ub, node.selected))[0]
    stats: dict[str, int | float | str] = {
        "candidate_visits": visits,
        "MILP_calls": milp_calls,
        "safe_pruned_candidates": safe_prunes,
        "accepted_candidates": accepted,
        "duplicate_proposals": duplicate_proposals,
        "duplicate_cache_accept": duplicate_cache_accept,
        "duplicate_cache_reject": duplicate_cache_reject,
        "duplicate_cache_refine": duplicate_cache_refine,
        "new_state_certify": new_state_certify,
        "unresolved_after_certification": unresolved_after_certification,
        "canonical_duplicate_reuse": duplicate_proposals,
        "search_wall_seconds": total_wall,
        "accepted_depth": final.depth,
        "operator_sequence": ";".join(action.split(":", 1)[0] for action in final.actions),
        "action_sequence": ";".join(final.actions),
        "stop_reason": stop_reason,
    }
    return final, trace, stats


def case_block(case: dict[str, Any]) -> dict[str, Any]:
    cfg, _, _ = load_config(str(CONFIG))
    capacity = cfg["capacity"]
    uav_capacity = float(capacity["uav_aggregate_capacity_mbps"])
    gbs_capacity = float(capacity["gbs_egress_capacity_mbps"])
    settings = cfg["openbase"]["adapter_settings"]
    terrain, case_id = case["terrain"], case["case_id"]
    demand = pd.read_csv(INPUTS / "demand_state" / terrain / f"{case['infrastructure_level']}_seed_{case['seed']}" / "poi_demand.csv")
    candidates = pd.read_csv(INPUTS / "terrains" / terrain / "uav_candidate.csv")
    access = pd.read_csv(INPUTS / "primitives/by_terrain" / f"{terrain}_access_phy_edges.csv")
    links = pd.read_csv(INPUTS / "primitives/by_terrain" / f"{terrain}_link_edges.csv")
    options_all = pd.read_csv(INPUTS / "primitives/backhaul_options.csv")
    options = options_all[options_all["case_id"].eq(case_id)].copy()
    candidate_ids = candidates["candidate_id"].astype(str).tolist()
    xy = candidates[["x_m", "y_m"]].to_numpy(float)
    demand_map = demand.set_index("poi_id")["service_demand_weight"].to_dict()
    nominal_by_uav = {
        str(uav): set(group["poi_id"].astype(str))
        for uav, group in access.groupby("uav_id")
    }
    feasible_subsets = {
        str(uav): set(group.loc[group["phy_feasible"].astype(bool), "poi_id"].astype(str))
        for uav, group in access.groupby("uav_id")
    }
    demand_score, network_score, relay_score, direct_score = weighted_candidate_features(
        terrain, case_id, candidate_ids, access, options, demand
    )
    overlap = weighted_overlap_matrix(candidate_ids, feasible_subsets, demand)
    satisfaction = 0.65 * normalize(demand_score) + 0.20 * normalize(network_score) + 0.15 * normalize(direct_score)
    method_seed = stable_seed(f"50m|{case_id}")
    selections: dict[str, list[str]] = {
        "Random": sorted(np.random.default_rng(method_seed).choice(candidate_ids, size=K, replace=False).tolist()),
        "KMeans-50m": weighted_kmeans_selection(demand, candidates, K, method_seed),
        "Fiedler-50m": fiedler_selection(terrain, case_id, demand, candidates, access, links, options, K),
        "Coverage-first": coverage_first_selection(candidate_ids, nominal_by_uav, demand_map, K),
        "PRM-ABS": adapt_prm_abs(
            candidate_ids, demand_score, network_score, xy, K,
            float(settings["prm_backhaul_weight"]),
        ),
        "UrbanInfrastructure-GA": adapt_urban_ga(
            candidate_ids, satisfaction, overlap, xy, K,
            seed=10_000 + int(case["seed"]) + 100 * TERRAINS.index(terrain),
            settings=settings["urban_ga"],
        ),
        "FlyingRelay": adapt_flying_relay(
            candidate_ids, demand_score, network_score, relay_score, xy, K,
            settings["flying_relay"],
        ),
    }
    for method, selected in selections.items():
        if len(selected) != K or len(set(selected)) != K or not set(selected) <= set(candidate_ids):
            raise RuntimeError(f"{case_id}/{method}: illegal K={K} deployment")

    rows: list[dict[str, Any]] = []
    result_by_method: dict[str, CertificationResult] = {}
    cache: dict[tuple[str, ...], tuple[CertificationResult, float]] = {}
    for method in INITIAL_METHODS:
        selected = tuple(sorted(selections[method]))
        reused = selected in cache
        if reused:
            result, wall = cache[selected]
            wall = 0.0
        else:
            result, wall = certify(
                case=case, method=method, selected=selected, demand=demand,
                access=access, links=links, options=options,
                uav_capacity=uav_capacity, gbs_capacity=gbs_capacity,
            )
            cache[selected] = (result, wall)
        result_by_method[method] = result
        rows.append(
            certification_row(
                result, case=case, method=method, selected=selected,
                source_method=method, wall_seconds=wall, cache_reused=reused,
            )
        )

    best_method = sorted(
        STRONG_POOL,
        key=lambda method: (
            -float(result_by_method[method].exact_admission.normalized_lower_bound),
            -float(result_by_method[method].exact_admission.normalized_upper_bound),
            method,
        ),
    )[0]
    best_selected = tuple(sorted(selections[best_method]))
    best_result = result_by_method[best_method]
    rows.append(
        certification_row(
            best_result, case=case, method="BestStrong", selected=best_selected,
            source_method=best_method, wall_seconds=0.0, cache_reused=True,
        )
    )
    final, trace, search_stats = run_csd_mr(
        case=case,
        initial_method=best_method,
        initial_selected=best_selected,
        initial_result=best_result,
        demand=demand,
        candidates=candidates,
        access=access,
        links=links,
        options=options,
        uav_capacity=uav_capacity,
        gbs_capacity=gbs_capacity,
        reference_mode=REFERENCE_MODE,
    )
    csd_row = certification_row(
        final.result, case=case, method="CSD-MR", selected=final.selected,
        source_method=best_method, wall_seconds=float(search_stats["search_wall_seconds"]), cache_reused=False,
    )
    best_lb = float(best_result.exact_admission.normalized_lower_bound)
    csd_row.update(
        {
            "initial_BestStrong_LB": best_lb,
            "CSD_MR_gain_LB": float(final.lb - best_lb),
            **search_stats,
        }
    )
    rows.append(csd_row)
    return {"case": case, "rows": rows, "trace": trace}


def preflight() -> None:
    if not (BASE / "PREPARED.flag").exists():
        raise RuntimeError("50 m inputs have not passed preparation")
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    protocol = {
        "status": "FROZEN_BEFORE_FORMAL_RESULTS",
        "input_protocol_hash": sha256(BASE / "protocol/frozen_protocol.json"),
        "input_manifest_hash": sha256(BASE / "protocol/preparation_manifest.json"),
        "runner_hash": sha256(script_path),
        "initial_methods": list(INITIAL_METHODS),
        "strong_pool": list(STRONG_POOL),
        "BestStrong_semantics": "post-hoc per-case strongest certified lower bound; derived comparator, not an independent algorithm",
        "reimplementation_boundary": {
            "KMeans-50m": "deterministic weighted Lloyd clustering plus one-to-one nearest-candidate assignment",
            "Fiedler-50m": "recursive Fiedler partition of the relay-feasible candidate graph plus demand/network representative selection",
            "old_executable_source_reused": False,
        },
        "CSD_MR": {
            "initial_state": "BestStrong",
            "horizon": HORIZON,
            "beam_width": BEAM_WIDTH,
            "visits_per_depth": VISITS_PER_DEPTH,
            "candidate_budget_per_operator": PROPOSALS_PER_OPERATOR,
            "time_limit_seconds_per_certification": TIME_LIMIT_SECONDS,
            "safe_valid_bound_precheck": True,
            "complete_recertification": True,
            "certified_LB_non_degradation": True,
            "duplicate_state_evidence_cache": not REFERENCE_MODE,
            "reference_mode": REFERENCE_MODE,
            "acceptance_threshold": "parent certified lower bound minus numerical tolerance",
        },
        "statistical_design": {
            "case_count": 60,
            "development_seed": 0,
            "confirmatory_seeds": [1, 2, 3, 4],
            "paired_cluster": "terrain x infrastructure x seed",
        },
        "forbidden": [
            "overwrite old results", "post-result threshold changes", "DD", "LBBD", "Branch-and-Certify",
            "claim KMeans/Fiedler native reproduction", "claim BestStrong independent sample",
        ],
    }
    path = OUT / "formal_search_protocol.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError("existing formal search protocol differs from current runner")
    else:
        write_json(path, protocol)
    print(json.dumps({"status": "PASS", "protocol": str(path)}, ensure_ascii=False))


def finalize(blocks: list[dict[str, Any]], elapsed: float) -> None:
    rows = pd.DataFrame([row for block in blocks for row in block["rows"]])
    trace = pd.DataFrame([row for block in blocks for row in block["trace"]])
    if len(rows) != 60 * (len(INITIAL_METHODS) + 2):
        raise RuntimeError(f"expected 540 method rows, found {len(rows)}")
    if rows.duplicated(["case_id", "method"]).any():
        raise RuntimeError("duplicate method rows")
    rows.to_csv(OUT / "formal_case_method_results.csv", index=False)
    trace.to_csv(OUT / "csd_mr_search_trace.csv", index=False)
    confirm = rows[rows["analysis_role"].eq("confirmatory")].copy()
    metric_cols = ["C_nom", "PHY", "LINK", "NET", "APP_LB", "APP_UB"]
    summary = confirm.groupby("method")[metric_cols].agg(["mean", "std", "median"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(OUT / "confirmatory_method_summary.csv", index=False)
    best = confirm[confirm["method"].eq("BestStrong")][["case_id", "APP_LB"]].rename(columns={"APP_LB": "BestStrong_LB"})
    csd = confirm[confirm["method"].eq("CSD-MR")][["case_id", "APP_LB", "CSD_MR_gain_LB"]].rename(columns={"APP_LB": "CSD_MR_LB"})
    paired = best.merge(csd, on="case_id", validate="one_to_one")
    paired["paired_gain"] = paired["CSD_MR_LB"] - paired["BestStrong_LB"]
    paired.to_csv(OUT / "paired_csd_mr_vs_beststrong.csv", index=False)
    reason_cols = [column for column in rows.columns if column.startswith("weight_fraction_")]
    confirm.groupby("method")[reason_cols].mean().reset_index().to_csv(OUT / "confirmatory_failure_layer_summary.csv", index=False)
    complexity_cols = [
        "method", "num_path_variables", "raw_path_variable_count", "num_binary_variables",
        "num_constraints", "num_nonzeros", "APP_solver_runtime_seconds", "full_certification_wall_seconds",
    ]
    confirm[complexity_cols].to_csv(OUT / "confirmatory_complexity_rows.csv", index=False)
    gates = {
        "case_method_row_count": int(len(rows)),
        "expected_case_method_rows": 540,
        "hierarchy_violations": int((~rows["hierarchy_pass"].astype(bool)).sum()),
        "invalid_bounds": int((~rows["bound_valid"].astype(bool)).sum()),
        "LB_above_UB": int((rows["APP_LB"] > rows["APP_UB"] + NUM_TOL).sum()),
        "capacity_violations": int((rows["max_capacity_violation_mbps"] > 1e-8).sum()),
        "wrong_K": int((rows["K_selected"] != K).sum()),
        "accepted_search_decrease": int(
            ((trace.get("accepted", pd.Series(dtype=bool)).astype(bool))
             & (trace.get("candidate_LB", pd.Series(dtype=float)) + NUM_TOL < trace.get("parent_LB", pd.Series(dtype=float)))).sum()
        ),
    }
    passed = all(
        gates[key] == 0
        for key in ("hierarchy_violations", "invalid_bounds", "LB_above_UB", "capacity_violations", "wrong_K", "accepted_search_decrease")
    ) and gates["case_method_row_count"] == gates["expected_case_method_rows"]
    result = {
        "status": "PASS" if passed else "FAIL",
        "gates": gates,
        "elapsed_wall_seconds": elapsed,
        "confirmatory_case_count": int(confirm["case_id"].nunique()),
        "CSD_MR_vs_BestStrong": {
            "mean_LB_gain": float(paired["paired_gain"].mean()),
            "median_LB_gain": float(paired["paired_gain"].median()),
            "positive_case_rate": float((paired["paired_gain"] > TOL).mean()),
            "negative_case_count": int((paired["paired_gain"] < -NUM_TOL).sum()),
        },
        "complexity": {
            "median_path_variables_all_confirmatory": float(confirm["num_path_variables"].median()),
            "median_solver_seconds_all_confirmatory": float(confirm["APP_solver_runtime_seconds"].median()),
            "CSD_MR_candidate_visits": int(rows.loc[rows["method"].eq("CSD-MR"), "candidate_visits"].fillna(0).sum()),
            "CSD_MR_MILP_calls": int(rows.loc[rows["method"].eq("CSD-MR"), "MILP_calls"].fillna(0).sum()),
            "CSD_MR_safe_pruned": int(rows.loc[rows["method"].eq("CSD-MR"), "safe_pruned_candidates"].fillna(0).sum()),
        },
    }
    write_json(OUT / "formal_search_result.json", result)
    (OUT / "FORMAL_SEARCH_DONE.flag").write_text(("PASS" if passed else "FAIL") + "\n", encoding="utf-8")
    report = f"""# 50 m formal deployment search

Status: **{result['status']}**.

## Frozen scope

- 60 cases: four terrains, three infrastructure levels, five seeds.
- Seed 0 is developmental; seeds 1--4 form 48 confirmatory paired case blocks.
- Critical facilities remain singleton demand units; background POIs use the frozen 50 m aggregation.
- Total low/medium/high demand remains 1000/1400/1800 Mbps.
- Historical full-catalog experiments were not overwritten.

## Method identity

- `KMeans-50m` and `Fiedler-50m` are formula-defined reimplementations, because executable source for the archived labels was not retained.
- `BestStrong` is a post-hoc per-case comparator selected from {', '.join(STRONG_POOL)} by certified lower bound. It is not an independent algorithm or independent sample.
- `CSD-MR` starts from the same BestStrong incumbent and uses diagnosis only for proposal priority. Every unique unpruned candidate is completely recertified; valid upper bounds may safely skip APP MILP; accepted lower bounds cannot decrease.

## Confirmatory outcome

- CSD-MR minus BestStrong mean/median certified-LB gain: `{result['CSD_MR_vs_BestStrong']['mean_LB_gain']:.8f}` / `{result['CSD_MR_vs_BestStrong']['median_LB_gain']:.8f}`.
- Positive-gain case rate: `{result['CSD_MR_vs_BestStrong']['positive_case_rate']:.2%}`.
- Negative CSD-MR final cases: `{result['CSD_MR_vs_BestStrong']['negative_case_count']}`.
- CSD-MR candidate visits/MILP calls/safe prunes over all 60 cases: `{result['complexity']['CSD_MR_candidate_visits']}` / `{result['complexity']['CSD_MR_MILP_calls']}` / `{result['complexity']['CSD_MR_safe_pruned']}`.

## Correctness

```json
{json.dumps(gates, ensure_ascii=False, indent=2, sort_keys=True)}
```

The fixed 5 s budget yields certified intervals. Solver runtime is therefore interpreted as budgeted certification cost, not time to proven optimality.
"""
    (OUT / "formal_search_report.md").write_text(report, encoding="utf-8")


def run(workers: int) -> None:
    preflight()
    if (OUT / "FORMAL_SEARCH_DONE.flag").exists():
        print(json.dumps({"status": "ALREADY_COMPLETE", "result": str(OUT / "formal_search_result.json")}, ensure_ascii=False))
        return
    cases = pd.read_csv(INPUTS / "case_index_50m.csv").sort_values(["terrain", "infrastructure_level", "seed"], kind="stable")
    completed = {path.stem.replace("case_", "") for path in WORK.glob("case_*.json")}
    tasks = [row._asdict() for row in cases.itertuples(index=False) if str(row.case_id) not in completed]
    started = time.perf_counter()
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(case_block, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                result = future.result()
                write_json(WORK / f"case_{task['case_id']}.json", result)
                print(f"50M FORMAL {len(completed)+index}/60 {task['case_id']}", flush=True)
    files = sorted(WORK.glob("case_*.json"))
    if len(files) != 60:
        raise RuntimeError(f"incomplete checkpoints: {len(files)}/60")
    blocks = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    finalize(blocks, time.perf_counter() - started)
    print((OUT / "formal_search_result.json").read_text(encoding="utf-8"))


def main() -> None:
    global OUT, WORK, REFERENCE_MODE, HORIZON
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--reference-mode", action="store_true")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    args = parser.parse_args()
    OUT = args.out_dir
    WORK = args.work_dir if args.work_dir is not None else OUT.parent / f"_{OUT.name}_work"
    REFERENCE_MODE = bool(args.reference_mode)
    HORIZON = int(args.horizon)
    if args.preflight_only:
        preflight()
    elif args.run:
        run(max(1, args.workers))
    else:
        raise SystemExit("Use --preflight-only or --run")


if __name__ == "__main__":
    main()
