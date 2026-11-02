"""Candidate-level service-chain kernel certification for the 50 m model.

This module is deliberately separate from the production certifier.  It is a
development gate for Certificate-Threshold Kernel Search (CT-KS): restricted
kernels may construct valid lower-bound witnesses, while every hard rejection
continues to rely on an upper bound computed over the complete path domain.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, vstack

from protocol_stack_v2_certifier import (
    CertificationResult,
    edge_capacities_for_case,
    first_blocking_reason,
    hierarchy_valid,
    valid_options_for_selection,
    weighted_ratio,
)
from protocol_stack_v2_exact_admission import (
    AdmissionResult,
    EdgeKey,
    _constructive_witness_from_lp,
    _prepare_variables_audited,
    greedy_capacity_admission,
    normalize_edge_key,
)


FEASIBILITY_TOLERANCE = 1.0e-8
NUMERICAL_TOLERANCE = 1.0e-10
LP_SUPPORT_TOLERANCE = 1.0e-9


@dataclass
class PreparedKernelModel:
    poi_ids: list[str]
    weights: np.ndarray
    demands: np.ndarray
    variables: list[dict[str, Any]]
    matrix: Any
    upper: np.ndarray
    utility: np.ndarray
    resource_names: list[str]
    lp_solution: np.ndarray
    reduced_costs: np.ndarray
    full_domain_upper_bound: float
    external_witness_indices: np.ndarray
    external_witness_utility: float
    normalization_weight: float
    preparation_seconds: float
    lp_seconds: float
    reduction_audit: dict[str, int]
    base_prepare_seconds: float = 0.0
    witness_seconds: float = 0.0
    has_full_domain_lp: bool = True


@dataclass
class RestrictedSolve:
    indices: np.ndarray
    witness_indices: np.ndarray
    utility: float
    solver_seconds: float
    status: int
    has_incumbent: bool
    exact_checker_pass: bool
    max_capacity_violation: float
    threshold_met: bool
    solver_dual_upper_bound: float | None
    mip_node_count: int | None
    mip_gap: float | None
    solver_message: str


@dataclass
class PolicyResult:
    policy: str
    lower_bound: float
    full_domain_upper_bound: float
    threshold: float
    threshold_certified: bool
    safely_rejected: bool
    unresolved: bool
    full_domain_fallback: bool
    ranking_resolved: bool
    initial_kernel_size: int
    final_kernel_size: int
    full_variable_count: int
    kernel_fraction: float
    restricted_solves: int
    full_solves: int
    milp_seconds: float
    preparation_seconds: float
    total_accounted_seconds: float
    exact_checker_pass: bool
    unsafe_upper_bound_use: bool
    non_nested_kernel: bool
    lower_bound_archive_drop: bool
    status: str
    witness_indices: tuple[int, ...]
    solve_history: tuple[dict[str, Any], ...]
    kernel_seconds: float = 0.0
    policy_wall_seconds: float = 0.0
    lp_seconds: float = 0.0
    witness_seconds: float = 0.0
    mip_relative_gap: float = 0.0
    strict_total_budget: bool = False


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def build_service_chains(
    *,
    case_id: str,
    terrain: str,
    selected_ids: Sequence[str],
    demand: pd.DataFrame,
    access_phy_edges: pd.DataFrame,
    link_edges: pd.DataFrame,
    backhaul_options: pd.DataFrame,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    dict[str, list[dict[str, Any]]],
    dict[EdgeKey, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    selected = set(map(str, selected_ids))
    poi_ids = demand["poi_id"].astype(str).tolist()
    weights = demand["service_demand_weight"].to_numpy(float)
    demands = demand["service_demand_mbps"].to_numpy(float)
    poi_index = {poi_id: index for index, poi_id in enumerate(poi_ids)}

    access = access_phy_edges[
        access_phy_edges["terrain"].eq(terrain)
        & access_phy_edges["uav_id"].astype(str).isin(selected)
    ].copy()
    nominal = np.zeros(len(poi_ids), dtype=bool)
    physical = np.zeros(len(poi_ids), dtype=bool)
    physical_pairs: set[tuple[str, str]] = set()
    for row in access.itertuples(index=False):
        poi_id = str(row.poi_id)
        if poi_id in poi_index:
            nominal[poi_index[poi_id]] = True
    for row in access[_truthy(access["phy_feasible"])].itertuples(index=False):
        poi_id = str(row.poi_id)
        if poi_id in poi_index:
            physical[poi_index[poi_id]] = True
            physical_pairs.add((str(row.uav_id), poi_id))
    access_links = link_edges[
        link_edges["scope_id"].eq(terrain)
        & link_edges["link_type"].eq("access")
        & link_edges["source_id"].astype(str).isin(selected)
        & _truthy(link_edges["link_feasible"])
    ]
    access_rate = {
        (str(row.source_id), str(row.target_id)): float(row.link_rate_mbps)
        for row in access_links.itertuples(index=False)
    }
    access_sources: dict[str, list[tuple[str, float]]] = {}
    link = np.zeros(len(poi_ids), dtype=bool)
    for source, poi_id in sorted(physical_pairs):
        rate = access_rate.get((source, poi_id))
        if rate is None or demands[poi_index[poi_id]] > rate + NUMERICAL_TOLERANCE:
            continue
        link[poi_index[poi_id]] = True
        access_sources.setdefault(poi_id, []).append((source, rate))

    case_options = backhaul_options[backhaul_options["case_id"].eq(case_id)]
    options_by_source = valid_options_for_selection(case_options, selected)
    chains_by_poi: dict[str, list[dict[str, Any]]] = {}
    for poi_id, sources in access_sources.items():
        for source, rate in sorted(sources):
            for option in options_by_source.get(source, []):
                chains_by_poi.setdefault(poi_id, []).append(
                    {**option, "access_rate_mbps": rate}
                )
    network = np.asarray([bool(chains_by_poi.get(poi_id)) for poi_id in poi_ids], dtype=bool)
    return (
        poi_ids,
        weights,
        demands,
        chains_by_poi,
        edge_capacities_for_case(link_edges, case_id, terrain),
        nominal,
        physical,
        link,
        network,
    )


def prepare_kernel_model(
    *,
    poi_ids: Sequence[str],
    weights: np.ndarray,
    demands: np.ndarray,
    chains_by_poi: Mapping[str, Sequence[Mapping[str, Any]]],
    edge_capacities: Mapping[EdgeKey, float],
    uav_capacity_mbps: float,
    gbs_capacity_mbps: float,
    lp_time_limit_seconds: float = 5.0,
    include_lp_witness: bool = True,
    wall_clock_budget_seconds: float | None = None,
) -> PreparedKernelModel:
    started = perf_counter()
    normalized_edges = {
        normalize_edge_key(key): float(value) for key, value in edge_capacities.items()
    }
    variables, reduction_audit = _prepare_variables_audited(
        list(poi_ids),
        np.asarray(weights, dtype=float),
        np.asarray(demands, dtype=float),
        chains_by_poi,
        normalized_edges,
        float(uav_capacity_mbps),
        float(gbs_capacity_mbps),
        True,
    )
    if not variables:
        raise ValueError("CT-KS gate requires at least one feasible service-chain variable")

    resource_caps: dict[str, float] = {}
    memberships: list[list[str]] = []
    for variable in variables:
        resources = [f"poi:{variable['poi_id']}"]
        resource_caps[resources[0]] = 1.0
        resources.append(f"uav:{variable['source']}")
        resource_caps[resources[-1]] = float(uav_capacity_mbps)
        if variable["relay"]:
            resources.append(f"uav:{variable['relay']}")
            resource_caps[resources[-1]] = float(uav_capacity_mbps)
        resources.append(f"gbs:{variable['gbs']}")
        resource_caps[resources[-1]] = float(gbs_capacity_mbps)
        for edge in variable["edges"]:
            name = "edge:" + ":".join(map(str, edge))
            resources.append(name)
            resource_caps[name] = float(normalized_edges[edge])
        memberships.append(resources)

    resource_names = sorted(resource_caps)
    row_index = {name: index for index, name in enumerate(resource_names)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for column, (variable, resources) in enumerate(zip(variables, memberships)):
        for resource in resources:
            rows.append(row_index[resource])
            cols.append(column)
            data.append(1.0 if resource.startswith("poi:") else float(variable["demand"]))
    matrix = coo_matrix(
        (data, (rows, cols)),
        shape=(len(resource_names), len(variables)),
        dtype=float,
    ).tocsr()
    upper = np.asarray([resource_caps[name] for name in resource_names], dtype=float)
    utility = np.asarray([float(variable["utility"]) for variable in variables], dtype=float)
    base_prepare_seconds = perf_counter() - started

    lp_seconds = 0.0
    witness_seconds = 0.0
    lp_solution = np.zeros(len(variables), dtype=float)
    reduced_costs = np.zeros(len(variables), dtype=float)
    full_domain_upper_bound = float(np.asarray(weights, dtype=float).sum())
    witness_indices_array = np.asarray([], dtype=int)
    external_utility = 0.0
    if include_lp_witness:
        lp_allowance = float(lp_time_limit_seconds)
        if wall_clock_budget_seconds is not None:
            lp_allowance = min(
                lp_allowance,
                max(float(wall_clock_budget_seconds) - base_prepare_seconds, 0.01),
            )
        lp_started = perf_counter()
        lp = linprog(
            c=-utility,
            A_ub=matrix,
            b_ub=upper,
            bounds=(0.0, 1.0),
            method="highs",
            options={"time_limit": lp_allowance, "presolve": True},
        )
        lp_seconds = perf_counter() - lp_started
        if int(lp.status) != 0 or lp.x is None or lp.fun is None:
            raise RuntimeError(f"Full-domain LP failed: status={lp.status}, message={lp.message}")
        lp_solution = np.asarray(lp.x, dtype=float)
        lower_marginals = getattr(getattr(lp, "lower", None), "marginals", None)
        reduced_costs = (
            np.asarray(lower_marginals, dtype=float)
            if lower_marginals is not None
            else np.zeros(len(variables), dtype=float)
        )
        full_domain_upper_bound = -float(lp.fun)

        witness_started = perf_counter()
        greedy = greedy_capacity_admission(
            poi_ids,
            weights,
            demands,
            chains_by_poi,
            normalized_edges,
            float(uav_capacity_mbps),
            float(gbs_capacity_mbps),
        )
        constructive = _constructive_witness_from_lp(
            poi_ids=poi_ids,
            weights=np.asarray(weights, dtype=float),
            variables=variables,
            matrix=matrix,
            upper=upper,
            resource_names=resource_names,
            lp_solution=lp_solution,
            starting_result=greedy,
            feasibility_tolerance=FEASIBILITY_TOLERANCE,
        )
        external = (
            constructive
            if constructive.objective_utility >= greedy.objective_utility - NUMERICAL_TOLERANCE
            else greedy
        )
        by_name = {
            (int(variable["poi_index"]), str(variable["name"])): index
            for index, variable in enumerate(variables)
        }
        witness_indices = []
        for poi_index_value, admitted in enumerate(external.admitted):
            if not admitted:
                continue
            index = by_name.get((poi_index_value, str(external.selected_chain[poi_index_value])))
            if index is None:
                raise RuntimeError("External witness path is absent from the reduced full domain")
            witness_indices.append(index)
        witness_indices_array = np.asarray(sorted(set(witness_indices)), dtype=int)
        _check_witness(matrix, upper, witness_indices_array)
        external_utility = float(external.objective_utility)
        witness_seconds = perf_counter() - witness_started
    return PreparedKernelModel(
        poi_ids=list(poi_ids),
        weights=np.asarray(weights, dtype=float),
        demands=np.asarray(demands, dtype=float),
        variables=variables,
        matrix=matrix,
        upper=upper,
        utility=utility,
        resource_names=resource_names,
        lp_solution=lp_solution,
        reduced_costs=reduced_costs,
        full_domain_upper_bound=full_domain_upper_bound,
        external_witness_indices=witness_indices_array,
        external_witness_utility=external_utility,
        normalization_weight=float(np.asarray(weights, dtype=float).sum()),
        preparation_seconds=perf_counter() - started,
        lp_seconds=lp_seconds,
        reduction_audit={key: int(value) for key, value in reduction_audit.items()},
        base_prepare_seconds=base_prepare_seconds,
        witness_seconds=witness_seconds,
        has_full_domain_lp=bool(include_lp_witness),
    )


def _check_witness(matrix: Any, upper: np.ndarray, indices: np.ndarray) -> float:
    vector = np.zeros(matrix.shape[1], dtype=float)
    vector[np.asarray(indices, dtype=int)] = 1.0
    lhs = np.asarray(matrix @ vector).reshape(-1)
    violation = float(np.maximum(lhs - upper, 0.0).max(initial=0.0))
    if violation > FEASIBILITY_TOLERANCE:
        raise RuntimeError(f"Restricted witness violates the complete model by {violation}")
    return violation


def initial_kernel_and_buckets(
    model: PreparedKernelModel,
    *,
    bucket_fraction: float = 0.10,
) -> tuple[np.ndarray, list[np.ndarray]]:
    lp_support = np.where(model.lp_solution > LP_SUPPORT_TOLERANCE)[0]
    kernel = np.asarray(
        sorted(set(lp_support.tolist()) | set(model.external_witness_indices.tolist())),
        dtype=int,
    )
    kernel_set = set(kernel.tolist())
    remaining = [index for index in range(len(model.variables)) if index not in kernel_set]
    remaining.sort(
        key=lambda index: (
            float(model.reduced_costs[index]),
            -float(model.utility[index]),
            str(model.variables[index]["name"]),
        )
    )
    bucket_size = max(32, int(np.ceil(bucket_fraction * len(model.variables))))
    buckets = [
        np.asarray(remaining[offset : offset + bucket_size], dtype=int)
        for offset in range(0, len(remaining), bucket_size)
    ]
    return kernel, buckets


def solve_restricted(
    model: PreparedKernelModel,
    indices: Sequence[int],
    *,
    time_limit_seconds: float,
    threshold_utility: float | None,
    inherited_witness: Sequence[int] | None = None,
    mip_relative_gap: float = 0.0,
) -> RestrictedSolve:
    indices_array = np.asarray(sorted(set(map(int, indices))), dtype=int)
    if not len(indices_array):
        raise ValueError("Restricted domain cannot be empty")
    restricted_matrix = model.matrix[:, indices_array]
    restricted_utility = model.utility[indices_array]
    lower = np.full(model.matrix.shape[0], -np.inf, dtype=float)
    upper = model.upper.copy()
    solve_matrix = restricted_matrix
    if threshold_utility is not None:
        solve_matrix = vstack(
            [solve_matrix, coo_matrix(restricted_utility.reshape(1, -1)).tocsr()],
            format="csr",
        )
        lower = np.concatenate([lower, [float(threshold_utility)]])
        upper = np.concatenate([upper, [np.inf]])
    options = {
        "time_limit": max(float(time_limit_seconds), 0.01),
        "mip_rel_gap": float(mip_relative_gap),
        "mip_feasibility_tolerance": 1.0e-9,
        "presolve": True,
        "threads": 1,
        "random_seed": 0,
    }
    started = perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Unrecognized options detected:.*passed to HiGHS verbatim"
        )
        result = milp(
            c=-restricted_utility,
            integrality=np.ones(len(indices_array), dtype=np.uint8),
            bounds=Bounds(np.zeros(len(indices_array)), np.ones(len(indices_array))),
            constraints=LinearConstraint(solve_matrix, lower, upper),
            options=options,
        )
    seconds = perf_counter() - started

    inherited_values = (
        model.external_witness_indices.tolist()
        if inherited_witness is None
        else list(inherited_witness)
    )
    best_indices = np.asarray(
        sorted(set(map(int, inherited_values))),
        dtype=int,
    )
    if not set(best_indices.tolist()) <= set(indices_array.tolist()):
        best_indices = np.asarray([], dtype=int)
    best_utility = float(model.utility[best_indices].sum()) if len(best_indices) else 0.0
    max_violation = _check_witness(model.matrix, model.upper, best_indices)
    has_incumbent = result.x is not None
    checker_pass = True
    if has_incumbent:
        rounded = (np.asarray(result.x, dtype=float) > 0.5).astype(float)
        chosen = indices_array[np.where(rounded > 0.5)[0]]
        try:
            violation = _check_witness(model.matrix, model.upper, chosen)
        except RuntimeError:
            checker_pass = False
            violation = float("inf")
        candidate_utility = float(model.utility[chosen].sum()) if checker_pass else -np.inf
        if checker_pass and candidate_utility > best_utility + NUMERICAL_TOLERANCE:
            best_indices = chosen
            best_utility = candidate_utility
            max_violation = violation
    dual = getattr(result, "mip_dual_bound", None)
    dual_upper = -float(dual) if dual is not None and np.isfinite(float(dual)) else None
    node_count_raw = getattr(result, "mip_node_count", None)
    node_count = (
        int(node_count_raw)
        if node_count_raw is not None and np.isfinite(float(node_count_raw))
        else None
    )
    mip_gap_raw = getattr(result, "mip_gap", None)
    mip_gap = (
        float(mip_gap_raw)
        if mip_gap_raw is not None and np.isfinite(float(mip_gap_raw))
        else None
    )
    threshold_met = bool(
        threshold_utility is None
        or best_utility + NUMERICAL_TOLERANCE >= float(threshold_utility)
    )
    return RestrictedSolve(
        indices=indices_array,
        witness_indices=best_indices,
        utility=best_utility,
        solver_seconds=seconds,
        status=int(result.status),
        has_incumbent=bool(has_incumbent and checker_pass),
        exact_checker_pass=checker_pass,
        max_capacity_violation=max_violation,
        threshold_met=threshold_met,
        solver_dual_upper_bound=dual_upper,
        mip_node_count=node_count,
        mip_gap=mip_gap,
        solver_message=str(getattr(result, "message", "")),
    )


def run_policy(
    model: PreparedKernelModel,
    *,
    policy: str,
    threshold_normalized: float,
    total_milp_budget_seconds: float = 5.0,
    stage_budget_seconds: float = 1.25,
    bucket_fraction: float = 0.10,
    mip_relative_gap: float = 0.0,
    strict_wall_clock_budget: bool = False,
    threshold_already_tolerance_adjusted: bool = False,
) -> PolicyResult:
    policy_started = perf_counter()
    threshold = max(0.0, float(threshold_normalized) * model.normalization_weight)
    if not threshold_already_tolerance_adjusted:
        threshold = max(0.0, threshold - NUMERICAL_TOLERANCE)
    full_indices = np.arange(len(model.variables), dtype=int)
    kernel0 = np.asarray([], dtype=int)
    buckets: list[np.ndarray] = []
    kernel_seconds = 0.0
    archive_indices = model.external_witness_indices.copy()
    archive_utility = float(model.external_witness_utility)
    archive_history = [archive_utility]
    nested_violation = False
    checker_pass = True
    restricted_solves = 0
    full_solves = 0
    milp_seconds = 0.0
    full_fallback = False
    safely_rejected = False
    unsafe_ub_use = False
    final_kernel = np.asarray([], dtype=int)
    policy_upper_bound = float(model.full_domain_upper_bound)
    status = ""
    solve_history: list[dict[str, Any]] = []

    def available_budget(requested: float) -> float:
        if strict_wall_clock_budget:
            remaining = total_milp_budget_seconds - (perf_counter() - policy_started)
        else:
            remaining = total_milp_budget_seconds - milp_seconds
        return max(0.0, min(float(requested), float(remaining)))

    def record_solve(
        phase: str, solve: RestrictedSolve, *, time_limit: float, domain_size: int
    ) -> None:
        solve_history.append(
            {
                "solve_index": len(solve_history) + 1,
                "phase": phase,
                "time_limit_seconds": float(time_limit),
                "domain_size": int(domain_size),
                "solver_seconds": float(solve.solver_seconds),
                "solver_status": int(solve.status),
                "solver_message": solve.solver_message,
                "has_incumbent": bool(solve.has_incumbent),
                "threshold_met": bool(solve.threshold_met),
                "mip_node_count": solve.mip_node_count,
                "mip_gap": solve.mip_gap,
                "solver_dual_upper_bound": solve.solver_dual_upper_bound,
                "witness_utility": float(solve.utility),
            }
        )

    if policy == "Full-5s":
        allowance = available_budget(total_milp_budget_seconds)
        if allowance <= 0.01:
            status = "budget_exhausted_before_full_domain"
        else:
            solve = solve_restricted(
                model,
                full_indices,
                time_limit_seconds=allowance,
                threshold_utility=None,
                inherited_witness=archive_indices,
                mip_relative_gap=mip_relative_gap,
            )
            record_solve(
                "full_domain", solve,
                time_limit=allowance, domain_size=len(full_indices),
            )
            full_solves = 1
            milp_seconds += solve.solver_seconds
            checker_pass &= solve.exact_checker_pass
            if solve.utility > archive_utility + NUMERICAL_TOLERANCE:
                archive_indices, archive_utility = solve.witness_indices, solve.utility
            if solve.solver_dual_upper_bound is not None:
                policy_upper_bound = min(
                    policy_upper_bound,
                    max(float(solve.solver_dual_upper_bound), archive_utility),
                )
            archive_history.append(archive_utility)
            final_kernel = full_indices
            status = "full_domain_completed_or_timed"
    elif policy == "Threshold-only":
        if not model.has_full_domain_lp:
            raise ValueError("Threshold-only requires a complete-domain LP bound")
        if model.full_domain_upper_bound + NUMERICAL_TOLERANCE < threshold:
            safely_rejected = True
            status = "full_domain_lp_safe_reject"
        else:
            allowance = available_budget(total_milp_budget_seconds)
            if allowance <= 0.01:
                status = "budget_exhausted_before_threshold_full_domain"
            else:
                solve = solve_restricted(
                    model,
                    full_indices,
                    time_limit_seconds=allowance,
                    threshold_utility=threshold,
                    inherited_witness=archive_indices,
                    mip_relative_gap=mip_relative_gap,
                )
                record_solve(
                    "threshold_full_domain", solve,
                    time_limit=allowance, domain_size=len(full_indices),
                )
                full_solves = 1
                milp_seconds += solve.solver_seconds
                checker_pass &= solve.exact_checker_pass
                if solve.utility > archive_utility + NUMERICAL_TOLERANCE:
                    archive_indices, archive_utility = solve.witness_indices, solve.utility
                archive_history.append(archive_utility)
                final_kernel = full_indices
                status = "threshold_full_domain_solve"
    elif policy in {"KS-base", "CT-KS"}:
        if not model.has_full_domain_lp:
            raise ValueError(f"{policy} requires a complete-domain LP bound")
        if policy == "CT-KS" and model.full_domain_upper_bound + NUMERICAL_TOLERANCE < threshold:
            safely_rejected = True
            status = "full_domain_lp_safe_reject"
        else:
            kernel_started = perf_counter()
            kernel0, buckets = initial_kernel_and_buckets(
                model, bucket_fraction=bucket_fraction
            )
            kernel_seconds = perf_counter() - kernel_started
            current = kernel0.copy()
            previous = set(current.tolist())
            final_kernel = current
            allowance = available_budget(stage_budget_seconds)
            if allowance <= 0.01:
                status = "budget_exhausted_before_initial_kernel"
            else:
                solve = solve_restricted(
                    model,
                    current,
                    time_limit_seconds=allowance,
                    threshold_utility=threshold if policy == "CT-KS" else None,
                    inherited_witness=archive_indices,
                    mip_relative_gap=mip_relative_gap,
                )
                record_solve(
                    "initial_kernel", solve,
                    time_limit=allowance, domain_size=len(current),
                )
                restricted_solves += 1
                milp_seconds += solve.solver_seconds
                checker_pass &= solve.exact_checker_pass
                if solve.utility > archive_utility + NUMERICAL_TOLERANCE:
                    archive_indices, archive_utility = solve.witness_indices, solve.utility
                archive_history.append(archive_utility)
                if policy == "CT-KS" and archive_utility + NUMERICAL_TOLERANCE >= threshold:
                    status = "threshold_certified_in_initial_kernel"
                else:
                    if buckets:
                        current = np.asarray(sorted(previous | set(buckets[0].tolist())), dtype=int)
                        nested_violation |= not previous <= set(current.tolist())
                        previous = set(current.tolist())
                        allowance = available_budget(stage_budget_seconds)
                        if allowance > 0.01:
                            solve = solve_restricted(
                                model,
                                current,
                                time_limit_seconds=allowance,
                                threshold_utility=threshold if policy == "CT-KS" else None,
                                inherited_witness=archive_indices,
                                mip_relative_gap=mip_relative_gap,
                            )
                            record_solve(
                                "expanded_kernel", solve,
                                time_limit=allowance, domain_size=len(current),
                            )
                            restricted_solves += 1
                            milp_seconds += solve.solver_seconds
                            checker_pass &= solve.exact_checker_pass
                            if solve.utility > archive_utility + NUMERICAL_TOLERANCE:
                                archive_indices, archive_utility = solve.witness_indices, solve.utility
                            archive_history.append(archive_utility)
                            final_kernel = current
                    if policy == "CT-KS" and archive_utility + NUMERICAL_TOLERANCE >= threshold:
                        status = "threshold_certified_after_expansion"
                    else:
                        allowance = available_budget(total_milp_budget_seconds)
                        if allowance <= 0.01:
                            status = "budget_exhausted_before_full_domain_fallback"
                        else:
                            full_fallback = True
                            solve = solve_restricted(
                                model,
                                full_indices,
                                time_limit_seconds=allowance,
                                threshold_utility=threshold if policy == "CT-KS" else None,
                                inherited_witness=archive_indices,
                                mip_relative_gap=mip_relative_gap,
                            )
                            record_solve(
                                "full_domain_fallback", solve,
                                time_limit=allowance, domain_size=len(full_indices),
                            )
                            full_solves += 1
                            milp_seconds += solve.solver_seconds
                            checker_pass &= solve.exact_checker_pass
                            if solve.utility > archive_utility + NUMERICAL_TOLERANCE:
                                archive_indices, archive_utility = solve.witness_indices, solve.utility
                            archive_history.append(archive_utility)
                            final_kernel = full_indices
                            status = "full_domain_fallback"
    else:
        raise ValueError(f"Unknown policy: {policy}")

    lower_drop = any(
        right + NUMERICAL_TOLERANCE < left
        for left, right in zip(archive_history, archive_history[1:])
    )
    threshold_certified = archive_utility + NUMERICAL_TOLERANCE >= threshold
    unresolved = not threshold_certified and not safely_rejected
    ranking_resolved = bool(
        policy_upper_bound - archive_utility <= NUMERICAL_TOLERANCE
    )
    policy_wall_seconds = perf_counter() - policy_started
    return PolicyResult(
        policy=policy,
        lower_bound=archive_utility / model.normalization_weight,
        full_domain_upper_bound=policy_upper_bound / model.normalization_weight,
        threshold=threshold / model.normalization_weight,
        threshold_certified=threshold_certified,
        safely_rejected=safely_rejected,
        unresolved=unresolved,
        full_domain_fallback=full_fallback,
        ranking_resolved=ranking_resolved,
        initial_kernel_size=len(kernel0),
        final_kernel_size=len(final_kernel),
        full_variable_count=len(model.variables),
        kernel_fraction=len(kernel0) / max(len(model.variables), 1),
        restricted_solves=restricted_solves,
        full_solves=full_solves,
        milp_seconds=milp_seconds,
        preparation_seconds=model.preparation_seconds,
        total_accounted_seconds=model.preparation_seconds + kernel_seconds + milp_seconds,
        exact_checker_pass=checker_pass,
        unsafe_upper_bound_use=unsafe_ub_use,
        non_nested_kernel=nested_violation,
        lower_bound_archive_drop=lower_drop,
        status=status,
        witness_indices=tuple(map(int, archive_indices.tolist())),
        solve_history=tuple(solve_history),
        kernel_seconds=kernel_seconds,
        policy_wall_seconds=policy_wall_seconds,
        lp_seconds=model.lp_seconds,
        witness_seconds=model.witness_seconds,
        mip_relative_gap=float(mip_relative_gap),
        strict_total_budget=bool(strict_wall_clock_budget),
    )


def refine_for_beam_ranking(
    model: PreparedKernelModel,
    result: PolicyResult,
    *,
    total_milp_budget_seconds: float = 5.0,
) -> PolicyResult:
    """Spend only the candidate's remaining budget on the complete domain."""
    if result.safely_rejected or result.full_domain_fallback:
        return result
    consumed = (
        float(result.total_accounted_seconds)
        if result.strict_total_budget
        else float(result.milp_seconds)
    )
    remaining = total_milp_budget_seconds - consumed
    if remaining <= 0.01:
        return result
    solve = solve_restricted(
        model,
        np.arange(len(model.variables), dtype=int),
        time_limit_seconds=remaining,
        threshold_utility=None,
        inherited_witness=result.witness_indices,
        mip_relative_gap=result.mip_relative_gap,
    )
    solve_record = {
        "solve_index": len(result.solve_history) + 1,
        "phase": "beam_full_fallback",
        "time_limit_seconds": float(remaining),
        "domain_size": int(len(model.variables)),
        "solver_seconds": float(solve.solver_seconds),
        "solver_status": int(solve.status),
        "solver_message": solve.solver_message,
        "has_incumbent": bool(solve.has_incumbent),
        "threshold_met": bool(solve.threshold_met),
        "mip_node_count": solve.mip_node_count,
        "mip_gap": solve.mip_gap,
        "solver_dual_upper_bound": solve.solver_dual_upper_bound,
        "witness_utility": float(solve.utility),
    }
    old_utility = float(result.lower_bound) * model.normalization_weight
    if solve.utility > old_utility + NUMERICAL_TOLERANCE:
        witness = tuple(map(int, solve.witness_indices.tolist()))
        lower = solve.utility / model.normalization_weight
    else:
        witness = result.witness_indices
        lower = result.lower_bound
    total_milp = float(result.milp_seconds) + float(solve.solver_seconds)
    return replace(
        result,
        lower_bound=float(lower),
        threshold_certified=bool(float(lower) + NUMERICAL_TOLERANCE >= result.threshold),
        unresolved=bool(float(lower) + NUMERICAL_TOLERANCE < result.threshold),
        full_domain_fallback=True,
        ranking_resolved=bool(
            result.full_domain_upper_bound - float(lower) <= NUMERICAL_TOLERANCE
        ),
        final_kernel_size=len(model.variables),
        full_solves=result.full_solves + 1,
        milp_seconds=total_milp,
        total_accounted_seconds=(
            result.total_accounted_seconds + solve.solver_seconds
            if result.strict_total_budget
            else model.preparation_seconds + result.kernel_seconds + total_milp
        ),
        policy_wall_seconds=result.policy_wall_seconds + solve.solver_seconds,
        exact_checker_pass=bool(result.exact_checker_pass and solve.exact_checker_pass),
        lower_bound_archive_drop=bool(result.lower_bound_archive_drop),
        status=result.status + "+beam_full_fallback",
        witness_indices=witness,
        solve_history=result.solve_history + (solve_record,),
    )


def certification_from_policy(
    *,
    model: PreparedKernelModel,
    policy: PolicyResult,
    case_id: str,
    method: str,
    nominal: np.ndarray,
    physical: np.ndarray,
    link: np.ndarray,
    network: np.ndarray,
) -> CertificationResult:
    """Reconstruct the standard certifier object from an exactly checked witness."""
    admitted = np.zeros(len(model.poi_ids), dtype=bool)
    selected_chain = [""] * len(model.poi_ids)
    for index in policy.witness_indices:
        variable = model.variables[int(index)]
        poi_index = int(variable["poi_index"])
        if admitted[poi_index]:
            raise RuntimeError("CT-KS witness selected multiple paths for one demand unit")
        admitted[poi_index] = True
        selected_chain[poi_index] = str(variable["name"])
    vector = np.zeros(len(model.variables), dtype=float)
    vector[np.asarray(policy.witness_indices, dtype=int)] = 1.0
    usage_vector = np.asarray(model.matrix @ vector).reshape(-1)
    violation = float(np.maximum(usage_vector - model.upper, 0.0).max(initial=0.0))
    if violation > FEASIBILITY_TOLERANCE:
        raise RuntimeError("CT-KS witness failed reconstruction in certification adapter")
    usage = {
        name: float(value)
        for name, value in zip(model.resource_names, usage_vector)
        if not name.startswith("poi:") and value > FEASIBILITY_TOLERANCE
    }
    exact = AdmissionResult(
        admitted=admitted,
        selected_chain=selected_chain,
        objective_utility=float(model.weights[admitted].sum()),
        admitted_count=int(admitted.sum()),
        optimal=False,
        solver_status=-3 if policy.safely_rejected else 1,
        solver_message=policy.status,
        mip_gap=None,
        feasible_chain_count=len(model.variables),
        deterministic_second_stage_used=False,
        max_capacity_violation_mbps=violation,
        resource_usage_mbps=usage,
        certification_status=(
            "bound_pruned_without_integer_solve"
            if policy.safely_rejected
            else "bounded_time_limited"
        ),
        bounded_time_limited=not policy.safely_rejected,
        utility_lower_bound=policy.lower_bound * model.normalization_weight,
        utility_upper_bound=policy.full_domain_upper_bound * model.normalization_weight,
        normalized_lower_bound=policy.lower_bound,
        normalized_upper_bound=policy.full_domain_upper_bound,
        normalization_weight=model.normalization_weight,
        external_feasible_lower_bound=policy.lower_bound * model.normalization_weight,
        lp_relaxation_upper_bound=policy.full_domain_upper_bound * model.normalization_weight,
        matrix_upper_bound=1.0,
        effective_relative_gap=(
            (policy.full_domain_upper_bound - policy.lower_bound)
            / max(abs(policy.lower_bound), 1.0e-12)
        ),
        bound_valid=policy.lower_bound <= policy.full_domain_upper_bound + NUMERICAL_TOLERANCE,
        termination_reason=policy.status,
        runtime_seconds=policy.total_accounted_seconds,
        solver_call_performed=bool(policy.restricted_solves or policy.full_solves),
        num_serviceable_users=int(len({int(v["poi_index"]) for v in model.variables})),
        num_path_variables=len(model.variables),
        num_binary_variables=len(model.variables),
        num_constraints=int(model.matrix.shape[0]),
        num_nonzeros=int(model.matrix.nnz),
        raw_path_variable_count=model.reduction_audit["raw_path_variable_count"],
        combined_valid_upper_bound=policy.full_domain_upper_bound * model.normalization_weight,
        absolute_utility_interval_width=(
            policy.full_domain_upper_bound - policy.lower_bound
        ) * model.normalization_weight,
        normalized_app_interval_width=policy.full_domain_upper_bound - policy.lower_bound,
        safe_prune_threshold_utility=policy.threshold * model.normalization_weight,
        safe_pruned_by_valid_bound=policy.safely_rejected,
        solver_has_incumbent=bool(policy.witness_indices),
        external_feasible_LB_available=True,
        effective_LB=policy.lower_bound * model.normalization_weight,
        effective_LB_source="ctks_exact_checked_witness",
    )
    reasons = first_blocking_reason(nominal, physical, link, network, admitted)
    attribution = pd.DataFrame(
        {
            "case_id": case_id,
            "method": method,
            "poi_id": model.poi_ids,
            "service_demand_weight": model.weights,
            "service_demand_mbps": model.demands,
            "nominal_pass": nominal,
            "phy_pass": physical,
            "link_pass": link,
            "net_pass": network,
            "app_admitted": admitted,
            "first_blocking_reason": reasons,
            "selected_service_chain": selected_chain,
        }
    )
    metrics = [
        weighted_ratio(mask, model.weights)
        for mask in (nominal, physical, link, network, admitted)
    ]
    summary = {
        "case_id": case_id,
        "method": method,
        "C_nom": metrics[0],
        "PHY": metrics[1],
        "LINK": metrics[2],
        "NET": metrics[3],
        "APP": metrics[4],
        "hierarchy_pass": hierarchy_valid(metrics),
        "poi_count": len(model.poi_ids),
        "served_poi_count": int(admitted.sum()),
        "exact_optimal": False,
        "certification_status": exact.certification_status,
        "APP_LB": policy.lower_bound,
        "APP_UB": policy.full_domain_upper_bound,
        "effective_relative_gap": exact.effective_relative_gap,
        "solver_status": exact.solver_status,
        "solver_mip_gap": None,
        "max_capacity_violation_mbps": violation,
        "greedy_APP": policy.lower_bound,
        "exact_minus_greedy_APP": 0.0,
    }
    return CertificationResult(summary, attribution, exact, exact)
