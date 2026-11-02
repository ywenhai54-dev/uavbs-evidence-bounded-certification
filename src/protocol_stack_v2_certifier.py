from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from protocol_stack_v2_exact_admission import (
    AdmissionResult,
    EdgeKey,
    exact_capacity_admission,
    greedy_capacity_admission,
    normalize_edge_key,
)


TOLERANCE = 1e-10
FIRST_BLOCKING_LABELS = [
    "fail_nominal",
    "fail_phy",
    "fail_link",
    "fail_net",
    "fail_app",
    "served",
]


@dataclass(frozen=True)
class CertificationResult:
    summary: dict[str, Any]
    poi_attribution: pd.DataFrame
    exact_admission: AdmissionResult
    greedy_admission: AdmissionResult


def weighted_ratio(mask: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(weights.sum())
    return float(weights[mask].sum() / denominator) if denominator > 0 else 0.0


def hierarchy_valid(values: Sequence[float], tolerance: float = TOLERANCE) -> bool:
    return all(left + tolerance >= right for left, right in zip(values, values[1:]))


def first_blocking_reason(
    nominal: np.ndarray,
    physical: np.ndarray,
    link: np.ndarray,
    network: np.ndarray,
    admitted: np.ndarray,
) -> np.ndarray:
    result = np.full(len(nominal), "served", dtype=object)
    result[~nominal] = "fail_nominal"
    result[nominal & ~physical] = "fail_phy"
    result[physical & ~link] = "fail_link"
    result[link & ~network] = "fail_net"
    result[network & ~admitted] = "fail_app"
    return result


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def edge_capacities_for_case(
    link_edges: pd.DataFrame, case_id: str, terrain: str
) -> dict[EdgeKey, float]:
    feasible = link_edges[_truthy(link_edges["link_feasible"])].copy()
    relay = feasible[(feasible["link_type"] == "relay") & (feasible["scope_id"] == terrain)]
    backhaul = feasible[
        (feasible["link_type"] == "backhaul") & (feasible["scope_id"] == case_id)
    ]
    capacities: dict[EdgeKey, float] = {}
    for row in pd.concat([relay, backhaul], ignore_index=True).itertuples(index=False):
        key = normalize_edge_key(
            (str(row.link_type), str(row.source_id), str(row.target_id))
        )
        value = float(row.link_rate_mbps)
        if key in capacities and not np.isclose(capacities[key], value, atol=1e-10, rtol=0.0):
            raise ValueError(f"Conflicting capacities for edge {key}: {capacities[key]} vs {value}")
        capacities[key] = value
    return capacities


def valid_options_for_selection(
    case_options: pd.DataFrame, selected_ids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in case_options.sort_values(
        ["source_uav_id", "option_rank", "path_nodes"], kind="stable"
    ).itertuples(index=False):
        source = str(row.source_uav_id)
        if source not in selected_ids:
            continue
        relay = "" if pd.isna(row.relay_uav_id) else str(row.relay_uav_id)
        path_type = str(row.path_type)
        if path_type not in {"direct", "one_relay"}:
            continue
        if path_type == "direct" and relay:
            continue
        if path_type == "one_relay" and (
            not relay or relay == source or relay not in selected_ids
        ):
            continue
        if int(row.path_edges) > 2 or int(row.relay_uavs) > 1:
            continue
        result.setdefault(source, []).append(
            {
                "source_uav_id": source,
                "relay_uav_id": relay,
                "gbs_id": str(row.gbs_id),
                "path_type": path_type,
                "path_nodes": str(row.path_nodes),
            }
        )
    return result


def certify_deployment_v2(
    *,
    case_id: str,
    method: str,
    terrain: str,
    selected_ids: Sequence[str],
    demand: pd.DataFrame,
    access_phy_edges: pd.DataFrame,
    link_edges: pd.DataFrame,
    backhaul_options: pd.DataFrame,
    uav_capacity_mbps: float,
    gbs_capacity_mbps: float,
    time_limit_seconds: float = 120.0,
    mip_relative_gap: float = 0.0,
    deterministic_second_stage: bool = False,
    gap_tolerance: float = 1e-6,
    lp_time_limit_seconds: float = 30.0,
    run_milp: bool = True,
    solver_threads: int = 1,
    solver_seed: int = 0,
    decision_threshold_normalized: float | None = None,
    safe_prune_below_normalized: float | None = None,
    use_constructive_witness: bool = True,
    explicit_x_variables: bool = False,
    add_cover_inequalities: bool = False,
) -> CertificationResult:
    required_demand = {
        "poi_id",
        "service_demand_weight",
        "service_demand_mbps",
    }
    missing = required_demand - set(demand.columns)
    if missing:
        raise ValueError(f"Demand frame lacks required columns: {sorted(missing)}")
    selected = set(map(str, selected_ids))
    poi_ids = demand["poi_id"].astype(str).tolist()
    weights = demand["service_demand_weight"].to_numpy(float)
    demands_mbps = demand["service_demand_mbps"].to_numpy(float)
    poi_index = {poi_id: index for index, poi_id in enumerate(poi_ids)}

    access = access_phy_edges[
        (access_phy_edges["terrain"] == terrain)
        & access_phy_edges["uav_id"].astype(str).isin(selected)
    ].copy()
    nominal = np.zeros(len(poi_ids), dtype=bool)
    physical = np.zeros(len(poi_ids), dtype=bool)
    physical_pairs: set[tuple[str, str]] = set()
    for row in access.itertuples(index=False):
        poi_id = str(row.poi_id)
        if poi_id not in poi_index:
            continue
        nominal[poi_index[poi_id]] = True
    physical_access = access[_truthy(access["phy_feasible"])]
    for row in physical_access.itertuples(index=False):
        poi_id = str(row.poi_id)
        if poi_id in poi_index:
            physical[poi_index[poi_id]] = True
            physical_pairs.add((str(row.uav_id), poi_id))

    access_links = link_edges[
        (link_edges["scope_id"] == terrain)
        & (link_edges["link_type"] == "access")
        & link_edges["source_id"].astype(str).isin(selected)
        & _truthy(link_edges["link_feasible"])
    ]
    access_rate = {
        (str(row.source_id), str(row.target_id)): float(row.link_rate_mbps)
        for row in access_links.itertuples(index=False)
    }
    access_sources: dict[str, list[tuple[str, float]]] = {}
    link = np.zeros(len(poi_ids), dtype=bool)
    for source_poi in sorted(physical_pairs):
        source, poi_id = source_poi
        rate = access_rate.get(source_poi)
        if rate is None:
            continue
        index = poi_index[poi_id]
        if demands_mbps[index] <= rate + TOLERANCE:
            link[index] = True
            access_sources.setdefault(poi_id, []).append((source, rate))

    case_options = backhaul_options[backhaul_options["case_id"] == case_id]
    options_by_source = valid_options_for_selection(case_options, selected)
    chains_by_poi: dict[str, list[dict[str, Any]]] = {}
    for poi_id, sources in access_sources.items():
        for source, rate in sorted(sources):
            for option in options_by_source.get(source, []):
                chains_by_poi.setdefault(poi_id, []).append(
                    {**option, "access_rate_mbps": rate}
                )
    network = np.array([bool(chains_by_poi.get(poi_id)) for poi_id in poi_ids], dtype=bool)
    edge_capacities = edge_capacities_for_case(link_edges, case_id, terrain)
    greedy = greedy_capacity_admission(
        poi_ids,
        weights,
        demands_mbps,
        chains_by_poi,
        edge_capacities,
        uav_capacity_mbps,
        gbs_capacity_mbps,
    )
    exact = exact_capacity_admission(
        poi_ids,
        weights,
        demands_mbps,
        chains_by_poi,
        edge_capacities,
        uav_capacity_mbps,
        gbs_capacity_mbps,
        time_limit_seconds=time_limit_seconds,
        mip_relative_gap=mip_relative_gap,
        deterministic_second_stage=deterministic_second_stage,
        gap_tolerance=gap_tolerance,
        lp_time_limit_seconds=lp_time_limit_seconds,
        external_feasible_result=greedy,
        run_milp=run_milp,
        solver_threads=solver_threads,
        solver_seed=solver_seed,
        decision_threshold_utility=(
            float(decision_threshold_normalized) * float(weights.sum())
            if decision_threshold_normalized is not None
            else None
        ),
        safe_prune_threshold_utility=(
            float(safe_prune_below_normalized) * float(weights.sum())
            if safe_prune_below_normalized is not None
            else None
        ),
        use_constructive_witness=use_constructive_witness,
        explicit_x_variables=explicit_x_variables,
        add_cover_inequalities=add_cover_inequalities,
    )
    metrics = [
        weighted_ratio(nominal, weights),
        weighted_ratio(physical, weights),
        weighted_ratio(link, weights),
        weighted_ratio(network, weights),
        weighted_ratio(exact.admitted, weights),
    ]
    reasons = first_blocking_reason(nominal, physical, link, network, exact.admitted)
    attribution = pd.DataFrame(
        {
            "case_id": case_id,
            "method": method,
            "poi_id": poi_ids,
            "service_demand_weight": weights,
            "service_demand_mbps": demands_mbps,
            "nominal_pass": nominal,
            "phy_pass": physical,
            "link_pass": link,
            "net_pass": network,
            "app_admitted": exact.admitted,
            "first_blocking_reason": reasons,
            "selected_service_chain": exact.selected_chain,
        }
    )
    summary = {
        "case_id": case_id,
        "method": method,
        "C_nom": metrics[0],
        "PHY": metrics[1],
        "LINK": metrics[2],
        "NET": metrics[3],
        "APP": metrics[4],
        "hierarchy_pass": hierarchy_valid(metrics),
        "poi_count": len(poi_ids),
        "served_poi_count": exact.admitted_count,
        "exact_optimal": exact.optimal,
        "certification_status": exact.certification_status,
        "APP_LB": exact.normalized_lower_bound,
        "APP_UB": exact.normalized_upper_bound,
        "effective_relative_gap": exact.effective_relative_gap,
        "solver_status": exact.solver_status,
        "solver_mip_gap": exact.mip_gap,
        "max_capacity_violation_mbps": exact.max_capacity_violation_mbps,
        "greedy_APP": weighted_ratio(greedy.admitted, weights),
        "exact_minus_greedy_APP": weighted_ratio(exact.admitted, weights)
        - weighted_ratio(greedy.admitted, weights),
    }
    return CertificationResult(summary, attribution, exact, greedy)
