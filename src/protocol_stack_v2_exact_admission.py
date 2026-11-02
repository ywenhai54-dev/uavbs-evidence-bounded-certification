from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, hstack, vstack


EdgeKey = tuple[str, str, str]


def audit_binary_reconstruction(matrix: Any, upper: np.ndarray, raw_solution: np.ndarray, n_variables: int) -> dict[str, Any]:
    """Audit raw and deterministic 0/1 vectors against the exact same rows."""
    raw = np.asarray(raw_solution[:n_variables], dtype=float)
    rounded = (raw > 0.5).astype(float)
    raw_lhs = np.asarray(matrix @ raw).reshape(-1)
    rounded_lhs = np.asarray(matrix @ rounded).reshape(-1)
    return {
        "raw": raw,
        "rounded": rounded,
        "raw_lhs": raw_lhs,
        "rounded_lhs": rounded_lhs,
        "max_integrality_residual": float(np.minimum(np.abs(raw), np.abs(1.0-raw)).max(initial=0.0)),
        "max_raw_violation": float(np.maximum(raw_lhs-upper,0.0).max(initial=0.0)),
        "max_rounded_violation": float(np.maximum(rounded_lhs-upper,0.0).max(initial=0.0)),
    }


def _capture_feasibility_failure(
    *, variables: Sequence[Mapping[str, Any]], resource_names: Sequence[str],
    matrix: Any, upper: np.ndarray, raw_solution: np.ndarray,
    rounded_solution: np.ndarray, solver_result: Any,
    feasibility_tolerance: float, max_violation: float,
) -> None:
    """Persist a forensic bundle only when explicitly enabled by an env var."""
    root_text = os.environ.get("PROTOCOL_STACK_FEASIBILITY_TRACE_DIR", "").strip()
    if not root_text:
        return
    root = Path(root_text)
    root.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(raw_solution[: len(variables)], dtype=float)
    rounded = np.asarray(rounded_solution[: len(variables)], dtype=float)
    raw_lhs = np.asarray(matrix @ raw).reshape(-1)
    rounded_lhs = np.asarray(matrix @ rounded).reshape(-1)
    with (root / "raw_solver_solution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variable_index","poi_index","poi_id","path_name","source_uav","relay_uav","gbs_id","demand_mbps","raw_value","rounded_value","integrality_residual"])
        writer.writeheader()
        for index, (variable, value, bit) in enumerate(zip(variables, raw, rounded)):
            writer.writerow({"variable_index":index,"poi_index":variable["poi_index"],"poi_id":variable["poi_id"],"path_name":variable["name"],
                "source_uav":variable["source"],"relay_uav":variable["relay"],"gbs_id":variable["gbs"],"demand_mbps":repr(float(variable["demand"])),
                "raw_value":repr(float(value)),"rounded_value":int(bit),"integrality_residual":repr(float(min(abs(value),abs(1-value))))})
    with (root / "model_capacity_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        fields=["row_index","resource_id","resource_type","model_lhs_raw","model_lhs_rounded","capacity_rhs","raw_slack","rounded_slack","raw_violation","rounded_violation"]
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for index,(name,raw_value,rounded_value,cap) in enumerate(zip(resource_names,raw_lhs,rounded_lhs,upper)):
            writer.writerow({"row_index":index,"resource_id":name,"resource_type":str(name).split(":",1)[0],"model_lhs_raw":repr(float(raw_value)),
                "model_lhs_rounded":repr(float(rounded_value)),"capacity_rhs":repr(float(cap)),"raw_slack":repr(float(cap-raw_value)),
                "rounded_slack":repr(float(cap-rounded_value)),"raw_violation":repr(float(max(raw_value-cap,0))),"rounded_violation":repr(float(max(rounded_value-cap,0)))})
    coo=matrix.tocoo()
    with (root / "model_path_resource_incidence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["row_index","resource_id","variable_index","poi_id","path_name","coefficient"]); writer.writeheader()
        for row,column,value in zip(coo.row,coo.col,coo.data):
            variable=variables[int(column)]
            writer.writerow({"row_index":int(row),"resource_id":resource_names[int(row)],"variable_index":int(column),"poi_id":variable["poi_id"],"path_name":variable["name"],"coefficient":repr(float(value))})
    chosen=rounded.astype(bool)
    with (root / "selected_paths.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["variable_index","poi_id","path_name","demand_mbps","source_uav","relay_uav","gbs_id"]); writer.writeheader()
        for index,variable in enumerate(variables):
            if chosen[index]: writer.writerow({"variable_index":index,"poi_id":variable["poi_id"],"path_name":variable["name"],"demand_mbps":repr(float(variable["demand"])),"source_uav":variable["source"],"relay_uav":variable["relay"],"gbs_id":variable["gbs"]})
    (root / "solver_config.json").write_text(json.dumps({"solver_status":int(solver_result.status),"solver_message":str(solver_result.message),
        "solver_mip_gap":float(solver_result.mip_gap) if getattr(solver_result,"mip_gap",None) is not None else None,
        "checker_feasibility_tolerance_mbps":feasibility_tolerance,"max_rounded_violation_mbps":max_violation,
        "max_integrality_residual":float(np.max(np.minimum(np.abs(raw),np.abs(1-raw)),initial=0.0)),
        "max_raw_row_violation_mbps":float(np.maximum(raw_lhs-upper,0).max(initial=0.0))},indent=2),encoding="utf-8")


def normalize_edge_key(edge: EdgeKey) -> EdgeKey:
    kind, source, target = map(str, edge)
    if kind == "relay":
        left, right = sorted((source, target))
        return kind, left, right
    return kind, source, target


@dataclass(frozen=True)
class AdmissionResult:
    admitted: np.ndarray
    selected_chain: list[str]
    objective_utility: float
    admitted_count: int
    optimal: bool
    solver_status: int
    solver_message: str
    mip_gap: float | None
    feasible_chain_count: int
    deterministic_second_stage_used: bool
    max_capacity_violation_mbps: float
    resource_usage_mbps: dict[str, float]
    certification_status: str = "not_applicable"
    solver_proven_optimal: bool = False
    certified_within_tolerance: bool = False
    bounded_time_limited: bool = False
    utility_lower_bound: float | None = None
    utility_upper_bound: float | None = None
    normalized_lower_bound: float | None = None
    normalized_upper_bound: float | None = None
    normalization_weight: float | None = None
    solver_incumbent_utility: float | None = None
    solver_dual_upper_bound: float | None = None
    external_feasible_lower_bound: float | None = None
    lp_relaxation_upper_bound: float | None = None
    matrix_upper_bound: float | None = None
    effective_relative_gap: float | None = None
    bound_valid: bool = False
    termination_reason: str = ""
    runtime_seconds: float = 0.0
    mip_node_count: int | None = None
    solver_call_performed: bool = False
    num_serviceable_users: int = 0
    num_path_variables: int = 0
    num_binary_variables: int = 0
    num_constraints: int = 0
    num_nonzeros: int = 0
    mean_paths_per_user: float = 0.0
    max_paths_per_user: int = 0
    direct_path_count: int = 0
    relay_path_count: int = 0
    shared_uav_constraint_count: int = 0
    shared_gbs_constraint_count: int = 0
    shared_backhaul_edge_constraint_count: int = 0
    shared_relay_edge_constraint_count: int = 0
    raw_path_variable_count: int = 0
    duplicate_path_columns_removed: int = 0
    dominated_paths_removed: int = 0
    individually_infeasible_paths_removed: int = 0
    redundant_x_variables_eliminated: bool = True
    individual_feasibility_upper_bound: float | None = None
    aggregate_capacity_upper_bound: float | None = None
    component_upper_bound: float | None = None
    combined_valid_upper_bound: float | None = None
    root_lp_gap: float | None = None
    absolute_utility_interval_width: float | None = None
    normalized_app_interval_width: float | None = None
    conflict_component_count: int = 0
    objective_lattice_proven: bool = False
    objective_lattice_increment: float | None = None
    model_template_cache_hit: bool = False
    decision_threshold_utility: float | None = None
    decision_infeasible_proven: bool = False
    safe_prune_threshold_utility: float | None = None
    safe_pruned_by_valid_bound: bool = False
    solver_has_incumbent: bool = False
    external_feasible_LB_available: bool = False
    effective_LB: float | None = None
    effective_LB_source: str = "none"
    constructive_external_utility: float | None = None
    time_to_first_incumbent: float | None = None


NORMAL_CERTIFICATION_STATES = {
    "solver_proven_optimal",
    "certified_within_tolerance",
    "bounded_time_limited",
    "externally_certified_optimal_by_objective_lattice",
}
EXCEPTION_CERTIFICATION_STATES = {
    "infeasible_proven",
    "no_incumbent",
    "invalid_bound",
    "numerical_failure",
    "solver_error",
}


@dataclass(frozen=True)
class BoundInterpretation:
    certification_status: str
    solver_proven_optimal: bool
    certified_within_tolerance: bool
    bounded_time_limited: bool
    utility_lower_bound: float | None
    utility_upper_bound: float | None
    normalized_lower_bound: float | None
    normalized_upper_bound: float | None
    solver_incumbent_utility: float | None
    solver_dual_upper_bound: float | None
    effective_relative_gap: float | None
    bound_valid: bool
    termination_reason: str


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if np.isfinite(converted) else None


def interpret_milp_bounds(
    result: Any,
    *,
    normalization_weight: float,
    external_feasible_lower_bound: float = 0.0,
    valid_external_upper_bound: float | None = None,
    gap_tolerance: float = 1e-6,
    comparison_tolerance: float = 1e-10,
) -> BoundInterpretation:
    """Convert SciPy's minimization bounds into maximization utility bounds."""
    if normalization_weight < 0 or not np.isfinite(normalization_weight):
        raise ValueError("normalization_weight must be finite and non-negative")
    status = int(getattr(result, "status", 4))
    message = str(getattr(result, "message", "missing solver message"))
    fun = _finite_or_none(getattr(result, "fun", None))
    dual = _finite_or_none(getattr(result, "mip_dual_bound", None))
    solver_incumbent = -fun if fun is not None else None
    solver_upper = -dual if dual is not None else None
    lower = max(float(external_feasible_lower_bound), solver_incumbent or 0.0)
    upper_candidates = [float(normalization_weight)]
    # HiGHS' reported dual bound and status are tolerance-qualified solver
    # diagnostics.  They are not, by themselves, proof-valid bounds under the
    # stricter certifier semantics (the exact audit produced counterexamples).
    # Hard bounds therefore use only independently proved relaxations.
    if valid_external_upper_bound is not None and np.isfinite(valid_external_upper_bound):
        upper_candidates.append(float(valid_external_upper_bound))
    upper = min(upper_candidates)

    if status == 2:
        return BoundInterpretation(
            "infeasible_proven", False, False, False, None, None, None, None,
            solver_incumbent, solver_upper, None, False, message,
        )
    if status == 3:
        return BoundInterpretation(
            "invalid_bound", False, False, False, lower, None,
            lower / normalization_weight if normalization_weight > 0 else 0.0,
            None, solver_incumbent, solver_upper, None, False, message,
        )
    if status == 4 and "numer" in message.lower():
        state = "numerical_failure"
    elif status not in {0, 1, 4}:
        state = "solver_error"
    else:
        state = ""

    valid = (
        np.isfinite(lower)
        and np.isfinite(upper)
        and lower >= -comparison_tolerance
        and lower <= upper + comparison_tolerance
        and upper <= normalization_weight + comparison_tolerance
    )
    if not valid:
        state = state or "invalid_bound"
    absolute_gap = max(upper - lower, 0.0) if valid else None
    effective_gap = (
        absolute_gap / max(abs(lower), 1e-12) if absolute_gap is not None else None
    )
    if not state:
        if solver_incumbent is None:
            state = "no_incumbent"
        elif status == 0 and absolute_gap is not None and absolute_gap <= comparison_tolerance:
            # Solver success is an exact certificate only when an independent,
            # proof-valid upper bound also closes the interval.
            state = "solver_proven_optimal"
        elif effective_gap is not None and effective_gap <= gap_tolerance:
            state = "certified_within_tolerance"
        else:
            state = "bounded_time_limited"
    normalized_lower = lower / normalization_weight if normalization_weight > 0 else 0.0
    normalized_upper = upper / normalization_weight if normalization_weight > 0 else 0.0
    return BoundInterpretation(
        certification_status=state,
        solver_proven_optimal=state == "solver_proven_optimal",
        certified_within_tolerance=state == "certified_within_tolerance",
        bounded_time_limited=state == "bounded_time_limited",
        utility_lower_bound=float(lower) if valid else None,
        utility_upper_bound=float(upper) if valid else None,
        normalized_lower_bound=float(normalized_lower) if valid else None,
        normalized_upper_bound=float(normalized_upper) if valid else None,
        solver_incumbent_utility=solver_incumbent,
        solver_dual_upper_bound=solver_upper,
        effective_relative_gap=float(effective_gap) if effective_gap is not None else None,
        bound_valid=bool(valid),
        termination_reason=message,
    )


def chain_edges(chain: Mapping[str, Any]) -> tuple[EdgeKey, ...]:
    source = str(chain["source_uav_id"])
    relay_value = chain.get("relay_uav_id", "")
    relay = "" if relay_value is None else str(relay_value)
    gbs = str(chain["gbs_id"])
    if relay:
        if relay == source:
            raise ValueError("A relay UAV cannot equal the source UAV")
        return (
            normalize_edge_key(("relay", source, relay)),
            normalize_edge_key(("backhaul", relay, gbs)),
        )
    return (normalize_edge_key(("backhaul", source, gbs)),)


def _chain_name(chain: Mapping[str, Any]) -> str:
    if chain.get("path_nodes"):
        return str(chain["path_nodes"])
    source = str(chain["source_uav_id"])
    relay_value = chain.get("relay_uav_id", "")
    relay = "" if relay_value is None else str(relay_value)
    gbs = str(chain["gbs_id"])
    return ">".join([source] + ([relay] if relay else []) + [gbs])


def _resource_name(kind: str, *parts: str) -> str:
    return ":".join((kind, *parts))


def _variable_resource_set(variable: Mapping[str, Any]) -> frozenset[str]:
    resources = {_resource_name("uav", str(variable["source"]))}
    relay = str(variable.get("relay", ""))
    if relay:
        resources.add(_resource_name("uav", relay))
    resources.add(_resource_name("gbs", str(variable["gbs"])))
    resources.update(_resource_name("edge", *edge) for edge in variable["edges"])
    return frozenset(resources)


def _reduce_resource_columns(
    raw_variables: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    variables: list[dict[str, Any]] = []
    duplicate_removed = 0
    dominated_removed = 0
    by_poi: dict[int, list[dict[str, Any]]] = {}
    for raw in raw_variables:
        variable = dict(raw)
        if "resource_set" not in variable:
            variable["resource_set"] = _variable_resource_set(variable)
        by_poi.setdefault(int(variable["poi_index"]), []).append(variable)
    for poi_index in sorted(by_poi):
        unique: list[dict[str, Any]] = []
        seen_resources: set[frozenset[str]] = set()
        for variable in sorted(by_poi[poi_index], key=lambda item: str(item["name"])):
            resource_set = frozenset(variable["resource_set"])
            if resource_set in seen_resources:
                duplicate_removed += 1
                continue
            seen_resources.add(resource_set)
            variable["resource_set"] = resource_set
            unique.append(variable)
        for index, variable in enumerate(unique):
            resources = variable["resource_set"]
            if any(
                other_index != index and other["resource_set"] < resources
                for other_index, other in enumerate(unique)
            ):
                dominated_removed += 1
            else:
                variables.append(variable)
    return variables, duplicate_removed, dominated_removed


def _prepare_variables_audited(
    poi_ids: Sequence[str],
    utility_weights: np.ndarray,
    demands_mbps: np.ndarray,
    chains_by_poi: Mapping[str, Sequence[Mapping[str, Any]]],
    edge_capacities_mbps: Mapping[EdgeKey, float],
    uav_capacity_mbps: float = float("inf"),
    gbs_capacity_mbps: float = float("inf"),
    reduce_paths: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    raw_variables: list[dict[str, Any]] = []
    individually_infeasible = 0
    for poi_index, poi_id in enumerate(poi_ids):
        demand = float(demands_mbps[poi_index])
        ordered_chains = sorted(chains_by_poi.get(str(poi_id), ()), key=_chain_name)
        for chain in ordered_chains:
            access_rate = float(chain["access_rate_mbps"])
            if demand > access_rate + 1e-10:
                individually_infeasible += 1
                continue
            edges = chain_edges(chain)
            if any(edge not in edge_capacities_mbps for edge in edges):
                individually_infeasible += 1
                continue
            if any(demand > float(edge_capacities_mbps[edge]) + 1e-10 for edge in edges):
                individually_infeasible += 1
                continue
            if demand > uav_capacity_mbps + 1e-10 or demand > gbs_capacity_mbps + 1e-10:
                individually_infeasible += 1
                continue
            name = _chain_name(chain)
            relay_value = chain.get("relay_uav_id", "")
            relay = "" if relay_value is None else str(relay_value)
            raw_variables.append(
                {
                    "poi_index": poi_index,
                    "poi_id": str(poi_id),
                    "utility": float(utility_weights[poi_index]),
                    "demand": demand,
                    "source": str(chain["source_uav_id"]),
                    "relay": relay,
                    "gbs": str(chain["gbs_id"]),
                    "edges": edges,
                    "name": name,
                }
            )
    if not reduce_paths:
        return raw_variables, {
            "raw_path_variable_count": len(raw_variables),
            "duplicate_path_columns_removed": 0,
            "dominated_paths_removed": 0,
            "individually_infeasible_paths_removed": individually_infeasible,
        }
    variables, duplicate_removed, dominated_removed = _reduce_resource_columns(raw_variables)
    return variables, {
        "raw_path_variable_count": len(raw_variables),
        "duplicate_path_columns_removed": duplicate_removed,
        "dominated_paths_removed": dominated_removed,
        "individually_infeasible_paths_removed": individually_infeasible,
    }


def _prepare_variables(
    poi_ids: Sequence[str],
    utility_weights: np.ndarray,
    demands_mbps: np.ndarray,
    chains_by_poi: Mapping[str, Sequence[Mapping[str, Any]]],
    edge_capacities_mbps: Mapping[EdgeKey, float],
    uav_capacity_mbps: float = float("inf"),
    gbs_capacity_mbps: float = float("inf"),
) -> list[dict[str, Any]]:
    variables, _ = _prepare_variables_audited(
        poi_ids,
        utility_weights,
        demands_mbps,
        chains_by_poi,
        edge_capacities_mbps,
        uav_capacity_mbps,
        gbs_capacity_mbps,
    )
    return variables


def _validate_inputs(
    poi_ids: Sequence[str], utility_weights: np.ndarray, demands_mbps: np.ndarray
) -> None:
    if len(poi_ids) != len(set(map(str, poi_ids))):
        raise ValueError("poi_ids must be unique")
    if len(poi_ids) != len(utility_weights) or len(poi_ids) != len(demands_mbps):
        raise ValueError("poi_ids, utility_weights, and demands_mbps must have equal length")
    if not np.isfinite(utility_weights).all() or (utility_weights < 0).any():
        raise ValueError("utility_weights must be finite and non-negative")
    if not np.isfinite(demands_mbps).all() or (demands_mbps < 0).any():
        raise ValueError("demands_mbps must be finite and non-negative")


_SPARSE_TEMPLATE_CACHE: dict[tuple[Any, ...], tuple[list[str], Any, np.ndarray, np.ndarray]] = {}
_SPARSE_TEMPLATE_CACHE_MAX = 8
_BOUND_TEMPLATE_CACHE: dict[
    tuple[Any, ...], tuple[float | None, np.ndarray | None, list[np.ndarray], float | None]
] = {}


def _fractional_knapsack_upper_bound(
    weights: np.ndarray, demands: np.ndarray, capacity: float
) -> float:
    if capacity < 0:
        return 0.0
    total = 0.0
    remaining = float(capacity)
    zero = demands <= 1e-15
    total += float(weights[zero].sum())
    indices = np.where(~zero)[0]
    order = sorted(indices, key=lambda idx: (-weights[idx] / demands[idx], int(idx)))
    for idx in order:
        if remaining <= 0:
            break
        take = min(1.0, remaining / float(demands[idx]))
        total += float(weights[idx]) * take
        remaining -= float(demands[idx]) * take
    return float(total)


def _conflict_components(matrix: Any) -> list[np.ndarray]:
    n_variables = int(matrix.shape[1])
    parent = list(range(n_variables))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    csr = matrix.tocsr()
    for row in range(csr.shape[0]):
        columns = csr.indices[csr.indptr[row] : csr.indptr[row + 1]]
        if len(columns) > 1:
            anchor = int(columns[0])
            for column in columns[1:]:
                union(anchor, int(column))
    groups: dict[int, list[int]] = {}
    for column in range(n_variables):
        groups.setdefault(find(column), []).append(column)
    return [np.asarray(columns, dtype=int) for _, columns in sorted(groups.items())]


def _component_lp_upper_bound(
    matrix: Any,
    upper: np.ndarray,
    utility: np.ndarray,
    components: Sequence[np.ndarray],
    time_limit_seconds: float,
) -> float | None:
    total = 0.0
    for columns in components:
        result = linprog(
            c=-utility[columns],
            A_ub=matrix[:, columns],
            b_ub=upper,
            bounds=(0.0, 1.0),
            method="highs",
            options={"time_limit": float(time_limit_seconds), "presolve": True},
        )
        if int(result.status) != 0 or result.fun is None:
            return None
        total += -float(result.fun)
    return float(total)


def detect_objective_lattice(
    weights: np.ndarray,
    *,
    max_decimal_scale: int = 1_000_000,
    tolerance: float = 1e-12,
) -> tuple[bool, float | None]:
    """Conservatively prove a decimal integer lattice up to a frozen scale."""
    positive = np.asarray(weights, dtype=float)
    positive = positive[positive > tolerance]
    if not len(positive):
        return True, 0.0
    scale = 1
    while scale <= max_decimal_scale:
        scaled = positive * scale
        rounded = np.rint(scaled)
        if np.max(np.abs(scaled - rounded)) <= tolerance:
            integers = rounded.astype(np.int64)
            gcd = int(np.gcd.reduce(integers))
            if gcd > 0:
                return True, float(gcd / scale)
        scale *= 10
    return False, None


def _constructive_witness_from_lp(
    *,
    poi_ids: Sequence[str],
    weights: np.ndarray,
    variables: Sequence[Mapping[str, Any]],
    matrix: Any,
    upper: np.ndarray,
    resource_names: Sequence[str],
    lp_solution: np.ndarray | None,
    starting_result: AdmissionResult | None,
    feasibility_tolerance: float,
) -> AdmissionResult:
    n_paths = len(variables)
    chosen = np.zeros(n_paths, dtype=bool)
    by_poi: dict[int, list[int]] = {}
    by_name: dict[tuple[int, str], int] = {}
    for index, variable in enumerate(variables):
        poi_index = int(variable["poi_index"])
        by_poi.setdefault(poi_index, []).append(index)
        by_name[(poi_index, str(variable["name"]))] = index
    csc = matrix.tocsc()
    usage = np.zeros(matrix.shape[0], dtype=float)
    if starting_result is not None:
        for poi_index, admitted in enumerate(starting_result.admitted):
            if not admitted:
                continue
            index = by_name.get((poi_index, str(starting_result.selected_chain[poi_index])))
            if index is None and by_poi.get(poi_index):
                index = by_poi[poi_index][0]
            if index is not None:
                start, end = csc.indptr[index], csc.indptr[index + 1]
                rows = csc.indices[start:end]
                values = csc.data[start:end]
                if np.all(usage[rows] + values <= upper[rows] + feasibility_tolerance):
                    chosen[index] = True
                    usage[rows] += values
    support = (
        np.asarray(lp_solution, dtype=float)
        if lp_solution is not None and len(lp_solution) == n_paths
        else np.zeros(n_paths, dtype=float)
    )
    order = sorted(
        range(n_paths),
        key=lambda index: (
            -support[index],
            -float(variables[index]["utility"]) / max(float(variables[index]["demand"]), 1e-12),
            -float(variables[index]["utility"]),
            str(variables[index]["name"]),
        ),
    )
    chosen_pois = {int(variables[index]["poi_index"]) for index in np.where(chosen)[0]}
    for index in order:
        poi_index = int(variables[index]["poi_index"])
        if poi_index in chosen_pois:
            continue
        start, end = csc.indptr[index], csc.indptr[index + 1]
        rows = csc.indices[start:end]
        values = csc.data[start:end]
        if np.all(usage[rows] + values <= upper[rows] + feasibility_tolerance):
            chosen[index] = True
            usage[rows] += values
            chosen_pois.add(poi_index)
    # One-for-one utility-improving swaps, still checked against the exact matrix.
    unadmitted_order = [
        index for index in order if int(variables[index]["poi_index"]) not in chosen_pois
    ][:500]
    for index in unadmitted_order:
        poi_index = int(variables[index]["poi_index"])
        selected_order = sorted(
            np.where(chosen)[0],
            key=lambda selected: (
                float(variables[int(selected)]["utility"]), str(variables[int(selected)]["name"])
            ),
        )[:64]
        in_start, in_end = csc.indptr[index], csc.indptr[index + 1]
        in_rows = csc.indices[in_start:in_end]
        in_values = csc.data[in_start:in_end]
        for selected_index in selected_order:
            if float(variables[index]["utility"]) <= float(variables[selected_index]["utility"]):
                continue
            out_start, out_end = csc.indptr[int(selected_index)], csc.indptr[int(selected_index) + 1]
            out_rows = csc.indices[out_start:out_end]
            out_values = csc.data[out_start:out_end]
            trial_usage = usage.copy()
            trial_usage[out_rows] -= out_values
            if np.all(
                trial_usage[in_rows] + in_values <= upper[in_rows] + feasibility_tolerance
            ):
                chosen[int(selected_index)] = False
                chosen[index] = True
                usage = trial_usage
                usage[in_rows] += in_values
                chosen_pois.remove(int(variables[int(selected_index)]["poi_index"]))
                chosen_pois.add(poi_index)
                break
    usage = np.asarray(matrix @ chosen.astype(float)).reshape(-1)
    violation = float(np.maximum(usage - upper, 0.0).max(initial=0.0))
    if violation > feasibility_tolerance:
        raise RuntimeError("Constructive witness failed exact capacity validation")
    admitted = np.zeros(len(poi_ids), dtype=bool)
    selected_chain = [""] * len(poi_ids)
    for index in np.where(chosen)[0]:
        poi_index = int(variables[int(index)]["poi_index"])
        if admitted[poi_index]:
            raise RuntimeError("Constructive witness selected multiple paths for one user")
        admitted[poi_index] = True
        selected_chain[poi_index] = str(variables[int(index)]["name"])
    resource_usage = {
        name: float(value)
        for name, value in zip(resource_names, usage)
        if not name.startswith("poi:") and value > feasibility_tolerance
    }
    objective = float(weights[admitted].sum())
    return AdmissionResult(
        admitted=admitted,
        selected_chain=selected_chain,
        objective_utility=objective,
        admitted_count=int(admitted.sum()),
        optimal=False,
        solver_status=-3,
        solver_message="Threshold-aware LP-support constructive external witness",
        mip_gap=None,
        feasible_chain_count=n_paths,
        deterministic_second_stage_used=False,
        max_capacity_violation_mbps=violation,
        resource_usage_mbps=resource_usage,
        certification_status="not_applicable",
        external_feasible_LB_available=True,
        effective_LB=objective,
        effective_LB_source="constructive_external",
        constructive_external_utility=objective,
    )


def exact_capacity_admission(
    poi_ids: Sequence[str],
    utility_weights: Sequence[float],
    demands_mbps: Sequence[float],
    chains_by_poi: Mapping[str, Sequence[Mapping[str, Any]]],
    edge_capacities_mbps: Mapping[EdgeKey, float],
    uav_capacity_mbps: float,
    gbs_capacity_mbps: float,
    *,
    time_limit_seconds: float = 120.0,
    mip_relative_gap: float = 0.0,
    primary_objective_tolerance: float = 1e-10,
    deterministic_second_stage: bool = True,
    feasibility_tolerance: float = 1e-8,
    gap_tolerance: float = 1e-6,
    lp_time_limit_seconds: float = 30.0,
    external_feasible_result: AdmissionResult | None = None,
    run_milp: bool = True,
    solver_threads: int = 1,
    solver_seed: int = 0,
    reduce_paths: bool = True,
    decision_threshold_utility: float | None = None,
    safe_prune_threshold_utility: float | None = None,
    use_constructive_witness: bool = True,
    explicit_x_variables: bool = False,
    add_cover_inequalities: bool = False,
) -> AdmissionResult:
    """Solve weighted service admission exactly using actual Mbps demands.

    Utility weights appear only in the objective. All node and edge constraints use
    ``demands_mbps``. A result is certified exact only when HiGHS returns status 0.
    """
    poi_ids = list(map(str, poi_ids))
    weights = np.asarray(utility_weights, dtype=float)
    demands = np.asarray(demands_mbps, dtype=float)
    _validate_inputs(poi_ids, weights, demands)
    if uav_capacity_mbps < 0 or gbs_capacity_mbps < 0:
        raise ValueError("Node capacities must be non-negative")
    if any((not np.isfinite(value) or value < 0) for value in edge_capacities_mbps.values()):
        raise ValueError("Edge capacities must be finite and non-negative")

    variables, reduction_audit = _prepare_variables_audited(
        poi_ids,
        weights,
        demands,
        chains_by_poi,
        {normalize_edge_key(key): value for key, value in edge_capacities_mbps.items()},
        uav_capacity_mbps,
        gbs_capacity_mbps,
        reduce_paths,
    )
    edge_capacities_mbps = {
        normalize_edge_key(key): value for key, value in edge_capacities_mbps.items()
    }
    empty_admitted = np.zeros(len(poi_ids), dtype=bool)
    if not variables:
        return AdmissionResult(
            admitted=empty_admitted,
            selected_chain=[""] * len(poi_ids),
            objective_utility=0.0,
            admitted_count=0,
            optimal=True,
            solver_status=0,
            solver_message="No individually feasible service chain",
            mip_gap=0.0,
            feasible_chain_count=0,
            deterministic_second_stage_used=False,
            max_capacity_violation_mbps=0.0,
            resource_usage_mbps={},
            certification_status="solver_proven_optimal",
            solver_proven_optimal=True,
            utility_lower_bound=0.0,
            utility_upper_bound=0.0,
            normalized_lower_bound=0.0,
            normalized_upper_bound=0.0,
            normalization_weight=float(weights.sum()),
            solver_incumbent_utility=0.0,
            solver_dual_upper_bound=0.0,
            external_feasible_lower_bound=0.0,
            lp_relaxation_upper_bound=0.0,
            matrix_upper_bound=0.0,
            effective_relative_gap=0.0,
            bound_valid=True,
            termination_reason="No individually feasible service chain",
            solver_call_performed=False,
            absolute_utility_interval_width=0.0,
            normalized_app_interval_width=0.0,
            raw_path_variable_count=int(reduction_audit["raw_path_variable_count"]),
            duplicate_path_columns_removed=int(reduction_audit["duplicate_path_columns_removed"]),
            dominated_paths_removed=int(reduction_audit["dominated_paths_removed"]),
            individually_infeasible_paths_removed=int(reduction_audit["individually_infeasible_paths_removed"]),
            solver_has_incumbent=False,
            external_feasible_LB_available=True,
            effective_LB=0.0,
            effective_LB_source="zero_service_external",
        )

    resource_caps: dict[str, float] = {}
    memberships: list[list[str]] = []
    for variable in variables:
        resources = [_resource_name("poi", variable["poi_id"])]
        resource_caps[resources[0]] = 1.0
        source_key = _resource_name("uav", variable["source"])
        resources.append(source_key)
        resource_caps[source_key] = float(uav_capacity_mbps)
        if variable["relay"]:
            relay_key = _resource_name("uav", variable["relay"])
            resources.append(relay_key)
            resource_caps[relay_key] = float(uav_capacity_mbps)
        gbs_key = _resource_name("gbs", variable["gbs"])
        resources.append(gbs_key)
        resource_caps[gbs_key] = float(gbs_capacity_mbps)
        for edge in variable["edges"]:
            edge_key = _resource_name("edge", *edge)
            resources.append(edge_key)
            resource_caps[edge_key] = float(edge_capacities_mbps[edge])
        memberships.append(resources)

    resource_names = sorted(resource_caps)
    template_key = tuple(
        (
            int(variable["poi_index"]),
            float(variable["utility"]),
            float(variable["demand"]),
            str(variable["source"]),
            str(variable["relay"]),
            str(variable["gbs"]),
            tuple(variable["edges"]),
        )
        for variable in variables
    ) + tuple((name, float(resource_caps[name])) for name in resource_names)
    cache_hit = template_key in _SPARSE_TEMPLATE_CACHE
    if cache_hit:
        resource_names, matrix, upper, utility_by_variable = _SPARSE_TEMPLATE_CACHE[template_key]
    else:
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
            (data, (rows, cols)), shape=(len(resource_names), len(variables)), dtype=float
        ).tocsr()
        upper = np.array([resource_caps[name] for name in resource_names], dtype=float)
        utility_by_variable = np.array([variable["utility"] for variable in variables])
        if len(_SPARSE_TEMPLATE_CACHE) >= _SPARSE_TEMPLATE_CACHE_MAX:
            _SPARSE_TEMPLATE_CACHE.pop(next(iter(_SPARSE_TEMPLATE_CACHE)))
        _SPARSE_TEMPLATE_CACHE[template_key] = (
            list(resource_names), matrix, upper.copy(), utility_by_variable.copy()
        )
    lower = np.full(len(resource_names), -np.inf, dtype=float)
    normalization_weight = float(weights.sum())
    represented_poi_indices = {int(variable["poi_index"]) for variable in variables}
    matrix_upper_bound = float(weights[list(represented_poi_indices)].sum())
    represented_indices = np.asarray(sorted(represented_poi_indices), dtype=int)
    aggregate_caps = []
    for prefix in ("uav:", "gbs:", "edge:backhaul:"):
        cap = sum(
            float(resource_caps[name])
            for name in resource_names
            if name.startswith(prefix)
        )
        if cap > 0:
            aggregate_caps.append(cap)
    aggregate_capacity_upper_bound = min(
        [matrix_upper_bound]
        + [
            _fractional_knapsack_upper_bound(
                weights[represented_indices], demands[represented_indices], cap
            )
            for cap in aggregate_caps
        ]
    )
    if template_key in _BOUND_TEMPLATE_CACHE:
        lp_upper_bound, lp_solution, components, component_upper_bound = _BOUND_TEMPLATE_CACHE[
            template_key
        ]
        lp_runtime = 0.0
    else:
        lp_started = perf_counter()
        lp = linprog(
            c=-utility_by_variable,
            A_ub=matrix,
            b_ub=upper,
            bounds=(0.0, 1.0),
            method="highs",
            options={"time_limit": float(lp_time_limit_seconds), "presolve": True},
        )
        lp_runtime = perf_counter() - lp_started
        lp_upper_bound = -float(lp.fun) if int(lp.status) == 0 and lp.fun is not None else None
        lp_solution = np.asarray(lp.x) if int(lp.status) == 0 and lp.x is not None else None
        components = _conflict_components(matrix)
        component_upper_bound = _component_lp_upper_bound(
            matrix,
            upper,
            utility_by_variable,
            components,
            max(float(lp_time_limit_seconds) / max(len(components), 1), 0.01),
        )
        if len(_BOUND_TEMPLATE_CACHE) >= _SPARSE_TEMPLATE_CACHE_MAX:
            _BOUND_TEMPLATE_CACHE.pop(next(iter(_BOUND_TEMPLATE_CACHE)))
        _BOUND_TEMPLATE_CACHE[template_key] = (
            lp_upper_bound,
            lp_solution.copy() if lp_solution is not None else None,
            components,
            component_upper_bound,
        )
    valid_external_upper_bound = min(
        value
        for value in [
            normalization_weight,
            matrix_upper_bound,
            aggregate_capacity_upper_bound,
            lp_upper_bound,
            component_upper_bound,
        ]
        if value is not None and np.isfinite(value)
    )
    constructive_result = (
        _constructive_witness_from_lp(
            poi_ids=poi_ids,
            weights=weights,
            variables=variables,
            matrix=matrix,
            upper=upper,
            resource_names=resource_names,
            lp_solution=lp_solution,
            starting_result=external_feasible_result,
            feasibility_tolerance=feasibility_tolerance,
        )
        if use_constructive_witness
        else external_feasible_result
    )
    if (
        constructive_result is not None
        and (
            external_feasible_result is None
            or constructive_result.objective_utility
            > external_feasible_result.objective_utility + feasibility_tolerance
        )
    ):
        external_feasible_result = constructive_result
    external_lower_bound = (
        float(external_feasible_result.objective_utility)
        if external_feasible_result is not None
        else 0.0
    )
    external_source = (
        "constructive_external"
        if use_constructive_witness and external_feasible_result is constructive_result
        else "greedy_external"
    )
    root_lp_gap = (
        (lp_upper_bound - external_lower_bound) / max(abs(external_lower_bound), 1e-12)
        if lp_upper_bound is not None
        else None
    )
    lattice_proven, lattice_increment = detect_objective_lattice(weights)
    path_counts = pd_counts = np.bincount(
        [int(variable["poi_index"]) for variable in variables], minlength=len(poi_ids)
    )
    serviceable_counts = path_counts[path_counts > 0]
    structural_fields = {
        "num_serviceable_users": int(len(represented_poi_indices)),
        "num_path_variables": int(len(variables)),
        "num_binary_variables": int(len(variables)),
        "num_constraints": int(matrix.shape[0]),
        "num_nonzeros": int(matrix.nnz),
        "mean_paths_per_user": float(serviceable_counts.mean()) if len(serviceable_counts) else 0.0,
        "max_paths_per_user": int(serviceable_counts.max()) if len(serviceable_counts) else 0,
        "direct_path_count": int(sum(not variable["relay"] for variable in variables)),
        "relay_path_count": int(sum(bool(variable["relay"]) for variable in variables)),
        "shared_uav_constraint_count": int(sum(name.startswith("uav:") for name in resource_names)),
        "shared_gbs_constraint_count": int(sum(name.startswith("gbs:") for name in resource_names)),
        "shared_backhaul_edge_constraint_count": int(sum(name.startswith("edge:backhaul:") for name in resource_names)),
        "shared_relay_edge_constraint_count": int(sum(name.startswith("edge:relay:") for name in resource_names)),
        "raw_path_variable_count": int(reduction_audit["raw_path_variable_count"]),
        "duplicate_path_columns_removed": int(reduction_audit["duplicate_path_columns_removed"]),
        "dominated_paths_removed": int(reduction_audit["dominated_paths_removed"]),
        "individually_infeasible_paths_removed": int(reduction_audit["individually_infeasible_paths_removed"]),
        "individual_feasibility_upper_bound": matrix_upper_bound,
        "aggregate_capacity_upper_bound": float(aggregate_capacity_upper_bound),
        "component_upper_bound": component_upper_bound,
        "combined_valid_upper_bound": float(valid_external_upper_bound),
        "root_lp_gap": root_lp_gap,
        "conflict_component_count": int(len(components)),
        "objective_lattice_proven": bool(lattice_proven),
        "objective_lattice_increment": lattice_increment,
        "model_template_cache_hit": bool(cache_hit),
    }
    safe_pruned = bool(
        run_milp
        and safe_prune_threshold_utility is not None
        and np.isfinite(float(safe_prune_threshold_utility))
        and valid_external_upper_bound + feasibility_tolerance
        < float(safe_prune_threshold_utility)
    )
    if not run_milp or safe_pruned:
        admitted = (
            external_feasible_result.admitted.copy()
            if external_feasible_result is not None
            else empty_admitted
        )
        selected_chain = (
            list(external_feasible_result.selected_chain)
            if external_feasible_result is not None
            else [""] * len(poi_ids)
        )
        upper_bound = valid_external_upper_bound
        effective_gap = (upper_bound - external_lower_bound) / max(
            abs(external_lower_bound), 1e-12
        )
        return AdmissionResult(
            admitted=admitted,
            selected_chain=selected_chain,
            objective_utility=external_lower_bound,
            admitted_count=int(admitted.sum()),
            optimal=False,
            solver_status=-3 if safe_pruned else -2,
            solver_message=(
                "Valid upper bound is strictly below the parent-edge acceptance threshold"
                if safe_pruned
                else "Pre-MILP valid bounds only"
            ),
            mip_gap=None,
            feasible_chain_count=len(variables),
            deterministic_second_stage_used=False,
            max_capacity_violation_mbps=(
                external_feasible_result.max_capacity_violation_mbps
                if external_feasible_result is not None
                else 0.0
            ),
            resource_usage_mbps=(
                dict(external_feasible_result.resource_usage_mbps)
                if external_feasible_result is not None
                else {}
            ),
            certification_status=(
                "bound_pruned_without_integer_solve"
                if safe_pruned
                else "bounded_time_limited"
            ),
            bounded_time_limited=not safe_pruned,
            utility_lower_bound=external_lower_bound,
            utility_upper_bound=upper_bound,
            normalized_lower_bound=(
                external_lower_bound / normalization_weight if normalization_weight > 0 else 0.0
            ),
            normalized_upper_bound=(
                upper_bound / normalization_weight if normalization_weight > 0 else 0.0
            ),
            normalization_weight=normalization_weight,
            external_feasible_lower_bound=external_lower_bound,
            lp_relaxation_upper_bound=lp_upper_bound,
            matrix_upper_bound=matrix_upper_bound,
            effective_relative_gap=effective_gap,
            bound_valid=external_lower_bound <= upper_bound + feasibility_tolerance,
            termination_reason=(
                "valid_upper_bound_strictly_below_parent_edge_threshold"
                if safe_pruned
                else "pre_milp_bounds_only"
            ),
            runtime_seconds=lp_runtime,
            solver_call_performed=False,
            absolute_utility_interval_width=float(upper_bound - external_lower_bound),
            normalized_app_interval_width=(
                float((upper_bound - external_lower_bound) / normalization_weight)
                if normalization_weight > 0
                else 0.0
            ),
            **structural_fields,
            solver_has_incumbent=False,
            external_feasible_LB_available=external_feasible_result is not None,
            effective_LB=external_lower_bound,
            effective_LB_source=external_source,
            safe_prune_threshold_utility=safe_prune_threshold_utility,
            safe_pruned_by_valid_bound=safe_pruned,
            constructive_external_utility=(
                constructive_result.objective_utility if constructive_result is not None else None
            ),
        )
    options = {
        "time_limit": float(time_limit_seconds),
        "mip_rel_gap": float(mip_relative_gap),
        # HiGHS' default MIP feasibility/integrality tolerance can admit a
        # near-binary raw vector whose deterministic 0/1 reconstruction
        # violates the already-frozen exact checker.  This only tightens the
        # numerical solve; it does not alter coefficients, RHS, or feasible set.
        "mip_feasibility_tolerance": min(1.0e-9, float(feasibility_tolerance) / 10.0),
        "presolve": True,
        "threads": int(solver_threads),
        "random_seed": int(solver_seed),
    }
    solve_matrix = matrix
    solve_lower = lower
    solve_upper = upper
    solve_objective = utility_by_variable.copy()
    solve_variable_count = len(variables)
    if explicit_x_variables:
        serviceable = sorted(represented_poi_indices)
        x_index = {poi_index: offset for offset, poi_index in enumerate(serviceable)}
        zero_block = coo_matrix((matrix.shape[0], len(serviceable))).tocsr()
        solve_matrix = hstack([matrix, zero_block], format="csr")
        equality_rows = []
        equality_cols = []
        equality_data = []
        for row_number, poi_index in enumerate(serviceable):
            for path_index, variable in enumerate(variables):
                if int(variable["poi_index"]) == poi_index:
                    equality_rows.append(row_number)
                    equality_cols.append(path_index)
                    equality_data.append(1.0)
            equality_rows.append(row_number)
            equality_cols.append(len(variables) + x_index[poi_index])
            equality_data.append(-1.0)
        equality_matrix = coo_matrix(
            (equality_data, (equality_rows, equality_cols)),
            shape=(len(serviceable), len(variables) + len(serviceable)),
        ).tocsr()
        solve_matrix = vstack([solve_matrix, equality_matrix], format="csr")
        solve_lower = np.concatenate([solve_lower, np.zeros(len(serviceable))])
        solve_upper = np.concatenate([solve_upper, np.zeros(len(serviceable))])
        solve_objective = np.concatenate([solve_objective, np.zeros(len(serviceable))])
        solve_variable_count += len(serviceable)
    cover_count = 0
    if add_cover_inequalities:
        cover_rows = []
        cover_cols = []
        cover_data = []
        cover_upper = []
        csr = matrix.tocsr()
        for resource_row, resource_name in enumerate(resource_names):
            if resource_name.startswith("poi:"):
                continue
            columns = csr.indices[csr.indptr[resource_row] : csr.indptr[resource_row + 1]]
            coefficients = csr.data[csr.indptr[resource_row] : csr.indptr[resource_row + 1]]
            ordered = sorted(
                zip(columns, coefficients), key=lambda item: (-float(item[1]), int(item[0]))
            )
            selected_cover = []
            total_demand = 0.0
            for column, coefficient in ordered:
                selected_cover.append(int(column))
                total_demand += float(coefficient)
                if total_demand > float(upper[resource_row]) + feasibility_tolerance:
                    break
            if total_demand <= float(upper[resource_row]) + feasibility_tolerance or len(selected_cover) < 2:
                continue
            for column in selected_cover:
                cover_rows.append(cover_count)
                cover_cols.append(column)
                cover_data.append(1.0)
            cover_upper.append(float(len(selected_cover) - 1))
            cover_count += 1
        if cover_count:
            cover_matrix = coo_matrix(
                (cover_data, (cover_rows, cover_cols)),
                shape=(cover_count, solve_variable_count),
            ).tocsr()
            solve_matrix = vstack([solve_matrix, cover_matrix], format="csr")
            solve_lower = np.concatenate([solve_lower, np.full(cover_count, -np.inf)])
            solve_upper = np.concatenate([solve_upper, np.asarray(cover_upper)])
    if decision_threshold_utility is not None:
        threshold_row = coo_matrix(solve_objective.reshape(1, -1)).tocsr()
        solve_matrix = vstack([solve_matrix, threshold_row], format="csr")
        solve_lower = np.concatenate([solve_lower, [float(decision_threshold_utility)]])
        solve_upper = np.concatenate([solve_upper, [np.inf]])
    structural_fields["num_binary_variables"] = int(solve_variable_count)
    structural_fields["num_constraints"] = int(solve_matrix.shape[0])
    structural_fields["num_nonzeros"] = int(solve_matrix.nnz)
    solve_started = perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Unrecognized options detected:.*passed to HiGHS verbatim"
        )
        first = milp(
            c=-solve_objective,
            integrality=np.ones(solve_variable_count, dtype=np.uint8),
            bounds=Bounds(np.zeros(solve_variable_count), np.ones(solve_variable_count)),
            constraints=LinearConstraint(solve_matrix, solve_lower, solve_upper),
            options=options,
        )
    solve_runtime = perf_counter() - solve_started
    interpretation = interpret_milp_bounds(
        first,
        normalization_weight=normalization_weight,
        external_feasible_lower_bound=external_lower_bound,
        valid_external_upper_bound=valid_external_upper_bound,
        gap_tolerance=gap_tolerance,
        comparison_tolerance=feasibility_tolerance,
    )
    decision_infeasible = bool(
        decision_threshold_utility is not None and int(first.status) == 2
    )
    lattice_certified = bool(
        lattice_proven
        and lattice_increment is not None
        and interpretation.bound_valid
        and interpretation.utility_lower_bound is not None
        and interpretation.utility_upper_bound is not None
        and interpretation.utility_upper_bound - interpretation.utility_lower_bound
        < lattice_increment - feasibility_tolerance
        and not interpretation.solver_proven_optimal
    )
    effective_status = (
        "decision_infeasible_proven"
        if decision_infeasible
        else (
            "externally_certified_optimal_by_objective_lattice"
            if lattice_certified
            else interpretation.certification_status
        )
    )
    if first.x is None:
        admitted = (
            external_feasible_result.admitted.copy()
            if external_feasible_result is not None
            else empty_admitted
        )
        selected_chain = (
            list(external_feasible_result.selected_chain)
            if external_feasible_result is not None
            else [""] * len(poi_ids)
        )
        return AdmissionResult(
            admitted=admitted,
            selected_chain=selected_chain,
            objective_utility=external_lower_bound,
            admitted_count=int(admitted.sum()),
            optimal=False,
            solver_status=int(first.status),
            solver_message=str(first.message),
            mip_gap=float(first.mip_gap) if getattr(first, "mip_gap", None) is not None else None,
            feasible_chain_count=len(variables),
            deterministic_second_stage_used=False,
            max_capacity_violation_mbps=(
                external_feasible_result.max_capacity_violation_mbps
                if external_feasible_result is not None
                else 0.0
            ),
            resource_usage_mbps=(
                dict(external_feasible_result.resource_usage_mbps)
                if external_feasible_result is not None
                else {}
            ),
            certification_status=effective_status,
            solver_proven_optimal=interpretation.solver_proven_optimal,
            certified_within_tolerance=interpretation.certified_within_tolerance,
            bounded_time_limited=interpretation.bounded_time_limited,
            utility_lower_bound=interpretation.utility_lower_bound,
            utility_upper_bound=interpretation.utility_upper_bound,
            normalized_lower_bound=interpretation.normalized_lower_bound,
            normalized_upper_bound=interpretation.normalized_upper_bound,
            normalization_weight=normalization_weight,
            solver_incumbent_utility=interpretation.solver_incumbent_utility,
            solver_dual_upper_bound=interpretation.solver_dual_upper_bound,
            external_feasible_lower_bound=external_lower_bound,
            lp_relaxation_upper_bound=lp_upper_bound,
            matrix_upper_bound=matrix_upper_bound,
            effective_relative_gap=interpretation.effective_relative_gap,
            bound_valid=interpretation.bound_valid,
            termination_reason=interpretation.termination_reason,
            runtime_seconds=lp_runtime + solve_runtime,
            mip_node_count=(
                int(first.mip_node_count)
                if getattr(first, "mip_node_count", None) is not None
                else None
            ),
            solver_call_performed=True,
            absolute_utility_interval_width=(
                interpretation.utility_upper_bound - interpretation.utility_lower_bound
                if interpretation.bound_valid
                else None
            ),
            normalized_app_interval_width=(
                interpretation.normalized_upper_bound - interpretation.normalized_lower_bound
                if interpretation.bound_valid
                else None
            ),
            **structural_fields,
            decision_threshold_utility=decision_threshold_utility,
            decision_infeasible_proven=decision_infeasible,
            solver_has_incumbent=False,
            external_feasible_LB_available=external_feasible_result is not None,
            effective_LB=interpretation.utility_lower_bound,
            effective_LB_source=external_source,
            constructive_external_utility=(
                constructive_result.objective_utility if constructive_result is not None else None
            ),
        )

    solution = np.asarray(first.x)
    second_stage_used = False
    if (
        int(first.status) == 0
        and deterministic_second_stage
        and not explicit_x_variables
        and not add_cover_inequalities
    ):
        primary_optimum = float(utility_by_variable @ solution)
        primary_row = coo_matrix(utility_by_variable.reshape(1, -1)).tocsr()
        second_matrix = vstack([matrix, primary_row], format="csr")
        second_lower = np.concatenate([lower, [primary_optimum - primary_objective_tolerance]])
        second_upper = np.concatenate([upper, [np.inf]])
        rank_cost = np.arange(1, len(variables) + 1, dtype=float) / (len(variables) + 1)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Unrecognized options detected:.*passed to HiGHS verbatim"
            )
            second = milp(
                c=rank_cost,
                integrality=np.ones(len(variables), dtype=np.uint8),
                bounds=Bounds(np.zeros(len(variables)), np.ones(len(variables))),
                constraints=LinearConstraint(second_matrix, second_lower, second_upper),
                options=options,
            )
        if int(second.status) == 0 and second.x is not None:
            solution = np.asarray(second.x)
            second_stage_used = True

    reconstruction = audit_binary_reconstruction(matrix, upper, solution, len(variables))
    chosen = reconstruction["rounded"].astype(bool)
    admitted = np.zeros(len(poi_ids), dtype=bool)
    selected_chain = [""] * len(poi_ids)
    for variable, is_chosen in zip(variables, chosen):
        if not is_chosen:
            continue
        poi_index = int(variable["poi_index"])
        if admitted[poi_index]:
            raise RuntimeError("MILP returned more than one chain for a POI")
        admitted[poi_index] = True
        selected_chain[poi_index] = str(variable["name"])

    usage_vector = reconstruction["rounded_lhs"]
    violations = np.maximum(usage_vector - upper, 0.0)
    max_violation = float(violations.max(initial=0.0))
    if max_violation > feasibility_tolerance:
        _capture_feasibility_failure(
            variables=variables, resource_names=resource_names, matrix=matrix, upper=upper,
            raw_solution=solution, rounded_solution=chosen.astype(float), solver_result=first,
            feasibility_tolerance=feasibility_tolerance, max_violation=max_violation,
        )
        # The solver may call a near-binary vector an incumbent under its own
        # integrality tolerance.  If deterministic reconstruction is not
        # feasible under the frozen checker, it is not an integer witness and
        # must not contribute to the LB or be labelled a solver incumbent.
        fallback = external_feasible_result
        fallback_admitted = fallback.admitted.copy() if fallback is not None else empty_admitted
        fallback_chains = list(fallback.selected_chain) if fallback is not None else [""] * len(poi_ids)
        fallback_usage = dict(fallback.resource_usage_mbps) if fallback is not None else {}
        fallback_violation = float(fallback.max_capacity_violation_mbps) if fallback is not None else 0.0
        if fallback_violation > feasibility_tolerance:
            raise RuntimeError("External fallback witness failed the frozen capacity checker")
        safe_lower = float(fallback.objective_utility) if fallback is not None else 0.0
        safe_upper = max(safe_lower, float(interpretation.utility_upper_bound))
        safe_gap = (safe_upper - safe_lower) / max(abs(safe_lower), 1e-12)
        return AdmissionResult(
            admitted=fallback_admitted,
            selected_chain=fallback_chains,
            objective_utility=safe_lower,
            admitted_count=int(fallback_admitted.sum()),
            optimal=False,
            solver_status=int(first.status),
            solver_message=str(first.message) + "; raw incumbent rejected after exact 0/1 reconstruction",
            mip_gap=float(first.mip_gap) if getattr(first, "mip_gap", None) is not None else None,
            feasible_chain_count=len(variables),
            deterministic_second_stage_used=second_stage_used,
            max_capacity_violation_mbps=fallback_violation,
            resource_usage_mbps=fallback_usage,
            certification_status="bounded_time_limited",
            solver_proven_optimal=False,
            certified_within_tolerance=False,
            bounded_time_limited=True,
            utility_lower_bound=safe_lower,
            utility_upper_bound=safe_upper,
            normalized_lower_bound=(safe_lower / normalization_weight if normalization_weight > 0 else 0.0),
            normalized_upper_bound=(safe_upper / normalization_weight if normalization_weight > 0 else 0.0),
            normalization_weight=normalization_weight,
            solver_incumbent_utility=None,
            solver_dual_upper_bound=interpretation.solver_dual_upper_bound,
            external_feasible_lower_bound=safe_lower,
            lp_relaxation_upper_bound=lp_upper_bound,
            matrix_upper_bound=matrix_upper_bound,
            effective_relative_gap=safe_gap,
            bound_valid=safe_lower <= safe_upper + feasibility_tolerance,
            termination_reason="solver_raw_incumbent_rejected_after_exact_reconstruction",
            runtime_seconds=float(solve_runtime),
            mip_node_count=(int(first.mip_node_count) if getattr(first, "mip_node_count", None) is not None else None),
            solver_call_performed=True,
            absolute_utility_interval_width=safe_upper-safe_lower,
            normalized_app_interval_width=((safe_upper-safe_lower)/normalization_weight if normalization_weight > 0 else 0.0),
            **structural_fields,
            decision_threshold_utility=decision_threshold_utility,
            decision_infeasible_proven=False,
            solver_has_incumbent=False,
            external_feasible_LB_available=fallback is not None,
            effective_LB=safe_lower,
            effective_LB_source=external_source if fallback is not None else "zero_service_external",
            constructive_external_utility=(constructive_result.objective_utility if constructive_result is not None else None),
        )
    usage = {
        name: float(value)
        for name, value in zip(resource_names, usage_vector)
        if not name.startswith("poi:") and value > feasibility_tolerance
    }
    solver_objective = float(weights[admitted].sum())
    if external_feasible_result is not None and external_lower_bound > solver_objective + feasibility_tolerance:
        admitted = external_feasible_result.admitted.copy()
        selected_chain = list(external_feasible_result.selected_chain)
        usage = dict(external_feasible_result.resource_usage_mbps)
        max_violation = float(external_feasible_result.max_capacity_violation_mbps)
    solver_incumbent_utility = interpretation.solver_incumbent_utility
    if solver_incumbent_utility is None:
        effective_source = external_source
    elif abs(solver_incumbent_utility - external_lower_bound) <= feasibility_tolerance:
        effective_source = "solver_and_external_tie"
    elif solver_incumbent_utility > external_lower_bound:
        effective_source = "solver_incumbent"
    else:
        effective_source = external_source
    return AdmissionResult(
        admitted=admitted,
        selected_chain=selected_chain,
        objective_utility=float(weights[admitted].sum()),
        admitted_count=int(admitted.sum()),
        optimal=interpretation.solver_proven_optimal,
        solver_status=int(first.status),
        solver_message=str(first.message),
        mip_gap=float(first.mip_gap) if getattr(first, "mip_gap", None) is not None else None,
        feasible_chain_count=len(variables),
        deterministic_second_stage_used=second_stage_used,
        max_capacity_violation_mbps=max_violation,
        resource_usage_mbps=usage,
        certification_status=effective_status,
        solver_proven_optimal=interpretation.solver_proven_optimal,
        certified_within_tolerance=interpretation.certified_within_tolerance,
        bounded_time_limited=interpretation.bounded_time_limited,
        utility_lower_bound=interpretation.utility_lower_bound,
        utility_upper_bound=interpretation.utility_upper_bound,
        normalized_lower_bound=interpretation.normalized_lower_bound,
        normalized_upper_bound=interpretation.normalized_upper_bound,
        normalization_weight=normalization_weight,
        solver_incumbent_utility=interpretation.solver_incumbent_utility,
        solver_dual_upper_bound=interpretation.solver_dual_upper_bound,
        external_feasible_lower_bound=external_lower_bound,
        lp_relaxation_upper_bound=lp_upper_bound,
        matrix_upper_bound=matrix_upper_bound,
        effective_relative_gap=interpretation.effective_relative_gap,
        bound_valid=interpretation.bound_valid,
        termination_reason=interpretation.termination_reason,
        runtime_seconds=lp_runtime + solve_runtime,
        mip_node_count=(
            int(first.mip_node_count)
            if getattr(first, "mip_node_count", None) is not None
            else None
        ),
        solver_call_performed=True,
        absolute_utility_interval_width=(
            interpretation.utility_upper_bound - interpretation.utility_lower_bound
            if interpretation.bound_valid
            else None
        ),
        normalized_app_interval_width=(
            interpretation.normalized_upper_bound - interpretation.normalized_lower_bound
            if interpretation.bound_valid
            else None
        ),
        **structural_fields,
        decision_threshold_utility=decision_threshold_utility,
        decision_infeasible_proven=decision_infeasible,
        solver_has_incumbent=True,
        external_feasible_LB_available=external_feasible_result is not None,
        effective_LB=interpretation.utility_lower_bound,
        effective_LB_source=effective_source,
        constructive_external_utility=(
            constructive_result.objective_utility if constructive_result is not None else None
        ),
    )


def greedy_capacity_admission(
    poi_ids: Sequence[str],
    utility_weights: Sequence[float],
    demands_mbps: Sequence[float],
    chains_by_poi: Mapping[str, Sequence[Mapping[str, Any]]],
    edge_capacities_mbps: Mapping[EdgeKey, float],
    uav_capacity_mbps: float,
    gbs_capacity_mbps: float,
) -> AdmissionResult:
    """Deterministic v2 greedy comparator using the same actual-demand constraints."""
    poi_ids = list(map(str, poi_ids))
    weights = np.asarray(utility_weights, dtype=float)
    demands = np.asarray(demands_mbps, dtype=float)
    _validate_inputs(poi_ids, weights, demands)
    edge_capacities_mbps = {
        normalize_edge_key(key): value for key, value in edge_capacities_mbps.items()
    }
    variables = _prepare_variables(
        poi_ids,
        weights,
        demands,
        chains_by_poi,
        edge_capacities_mbps,
        uav_capacity_mbps,
        gbs_capacity_mbps,
    )
    by_poi: dict[int, list[dict[str, Any]]] = {}
    for variable in variables:
        by_poi.setdefault(int(variable["poi_index"]), []).append(variable)
    admitted = np.zeros(len(poi_ids), dtype=bool)
    selected_chain = [""] * len(poi_ids)
    uav_load: dict[str, float] = {}
    gbs_load: dict[str, float] = {}
    edge_load: dict[EdgeKey, float] = {}
    order = sorted(range(len(poi_ids)), key=lambda index: (-weights[index], poi_ids[index]))
    for poi_index in order:
        demand = float(demands[poi_index])
        feasible: list[tuple[float, str, dict[str, Any]]] = []
        for variable in by_poi.get(poi_index, []):
            uavs = [variable["source"]] + ([variable["relay"]] if variable["relay"] else [])
            if any(uav_load.get(uav, 0.0) + demand > uav_capacity_mbps + 1e-10 for uav in uavs):
                continue
            if gbs_load.get(variable["gbs"], 0.0) + demand > gbs_capacity_mbps + 1e-10:
                continue
            if any(edge_load.get(edge, 0.0) + demand > edge_capacities_mbps[edge] + 1e-10 for edge in variable["edges"]):
                continue
            residuals = [uav_capacity_mbps - uav_load.get(uav, 0.0) - demand for uav in uavs]
            residuals.append(gbs_capacity_mbps - gbs_load.get(variable["gbs"], 0.0) - demand)
            residuals.extend(edge_capacities_mbps[edge] - edge_load.get(edge, 0.0) - demand for edge in variable["edges"])
            feasible.append((min(residuals), variable["name"], variable))
        if not feasible:
            continue
        _, _, chosen = max(feasible, key=lambda item: (item[0], item[1]))
        for uav in [chosen["source"]] + ([chosen["relay"]] if chosen["relay"] else []):
            uav_load[uav] = uav_load.get(uav, 0.0) + demand
        gbs_load[chosen["gbs"]] = gbs_load.get(chosen["gbs"], 0.0) + demand
        for edge in chosen["edges"]:
            edge_load[edge] = edge_load.get(edge, 0.0) + demand
        admitted[poi_index] = True
        selected_chain[poi_index] = chosen["name"]
    usage = {_resource_name("uav", key): value for key, value in uav_load.items()}
    usage.update({_resource_name("gbs", key): value for key, value in gbs_load.items()})
    usage.update({_resource_name("edge", *key): value for key, value in edge_load.items()})
    return AdmissionResult(
        admitted=admitted,
        selected_chain=selected_chain,
        objective_utility=float(weights[admitted].sum()),
        admitted_count=int(admitted.sum()),
        optimal=False,
        solver_status=-1,
        solver_message="Deterministic greedy comparator; optimality not certified",
        mip_gap=None,
        feasible_chain_count=len(variables),
        deterministic_second_stage_used=False,
        max_capacity_violation_mbps=0.0,
        resource_usage_mbps=usage,
    )
