from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from protocol_stack_v2_certifier import (
    CertificationResult,
    _truthy,
    edge_capacities_for_case,
    valid_options_for_selection,
)
from protocol_stack_v2_exact_admission import chain_edges


OPERATORS = ("O1", "O2", "O3")
OPERATOR_INFLUENCE = {
    "O1": {
        "name": "spatial_access_propagation_repair",
        "direct_layers": ("C_nom", "PHY", "LINK"),
        "indirect_layers": ("NET", "APP"),
        "matched_failures": ("fail_nominal", "fail_phy", "fail_link"),
    },
    "O2": {
        "name": "topology_backhaul_repair",
        "direct_layers": ("NET",),
        "indirect_layers": ("APP",),
        "matched_failures": ("fail_net",),
    },
    "O3": {
        "name": "resource_capacity_repair",
        "direct_layers": ("APP",),
        "indirect_layers": (),
        "matched_failures": ("fail_app",),
    },
}
FAILURE_TO_OPERATOR = {
    failure: operator
    for operator, values in OPERATOR_INFLUENCE.items()
    for failure in values["matched_failures"]
}


@dataclass(frozen=True)
class CandidateProposal:
    operator: str
    candidate_out: str
    candidate_in: str
    selected_ids: tuple[str, ...]
    proxy_primary: float
    proxy_secondary: float
    proxy_tertiary: float
    rank: int


def weighted_failure_signature(attribution: pd.DataFrame) -> dict[str, float]:
    failures = attribution[attribution["first_blocking_reason"] != "served"]
    total = float(attribution["service_demand_weight"].sum())
    grouped = failures.groupby("first_blocking_reason")["service_demand_weight"].sum()
    return {
        label: float(grouped.get(label, 0.0) / total) if total > 0 else 0.0
        for label in FAILURE_TO_OPERATOR
    }


def dominant_attribution(attribution: pd.DataFrame) -> tuple[str, str, dict[str, float]]:
    signature = weighted_failure_signature(attribution)
    failure = max(
        FAILURE_TO_OPERATOR,
        key=lambda label: (signature[label], -list(FAILURE_TO_OPERATOR).index(label)),
    )
    return failure, FAILURE_TO_OPERATOR[failure], signature


def _weighted_sum(poi_ids: set[str], weight_map: Mapping[str, float]) -> float:
    return float(sum(float(weight_map.get(poi_id, 0.0)) for poi_id in poi_ids))


def generate_operator_candidates(
    *,
    case_id: str,
    terrain: str,
    selected_ids: Sequence[str],
    candidate_pool: Sequence[str],
    demand: pd.DataFrame,
    baseline: CertificationResult,
    access_phy_edges: pd.DataFrame,
    link_edges: pd.DataFrame,
    backhaul_options: pd.DataFrame,
    uav_capacity_mbps: float,
    gbs_capacity_mbps: float,
    budget_per_operator: int,
) -> dict[str, list[CandidateProposal]]:
    """Rank legal one-for-one swaps independently for O1, O2, and O3.

    The proxies only screen candidates. Every retained state still requires the full
    v2 certifier, and diagnosis never hard-disables another legal operator.
    """
    if budget_per_operator <= 0:
        raise ValueError("budget_per_operator must be positive")
    selected = set(map(str, selected_ids))
    pool = sorted(set(map(str, candidate_pool)))
    weight_map = demand.set_index("poi_id")["service_demand_weight"].astype(float).to_dict()
    demand_map = demand.set_index("poi_id")["service_demand_mbps"].astype(float).to_dict()
    terrain_access = access_phy_edges[access_phy_edges["terrain"] == terrain].copy()
    terrain_access["_phy"] = _truthy(terrain_access["phy_feasible"])
    nominal_by_uav = {
        str(uav): set(group["poi_id"].astype(str))
        for uav, group in terrain_access.groupby("uav_id")
    }
    physical_by_uav = {
        str(uav): set(group.loc[group["_phy"], "poi_id"].astype(str))
        for uav, group in terrain_access.groupby("uav_id")
    }
    access_links = link_edges[
        (link_edges["scope_id"] == terrain)
        & (link_edges["link_type"] == "access")
        & _truthy(link_edges["link_feasible"])
    ]
    access_rate = {
        (str(row.source_id), str(row.target_id)): float(row.link_rate_mbps)
        for row in access_links.itertuples(index=False)
    }
    link_by_uav: dict[str, set[str]] = {}
    for uav, pois in physical_by_uav.items():
        link_by_uav[uav] = {
            poi
            for poi in pois
            if (uav, poi) in access_rate
            and float(demand_map.get(poi, np.inf)) <= access_rate[(uav, poi)] + 1e-10
        }
    case_options = backhaul_options[backhaul_options["case_id"] == case_id]
    edge_caps = edge_capacities_for_case(link_edges, case_id, terrain)
    baseline_usage = baseline.exact_admission.resource_usage_mbps
    failed_app = set(
        baseline.poi_attribution.loc[
            baseline.poi_attribution["first_blocking_reason"] == "fail_app", "poi_id"
        ].astype(str)
    )

    proposals: dict[str, list[CandidateProposal]] = {}
    all_pairs = [(out, incoming) for out in sorted(selected) for incoming in pool if incoming not in selected]
    if len(all_pairs) < budget_per_operator:
        raise ValueError("Not enough legal swaps to fill the operator budget")
    for operator in OPERATORS:
        scored: list[tuple[tuple[float, float, float], str, str, tuple[str, ...]]] = []
        for candidate_out, candidate_in in all_pairs:
            proposal = tuple(sorted((selected - {candidate_out}) | {candidate_in}))
            proposal_set = set(proposal)
            nominal_union = set().union(*(nominal_by_uav.get(uav, set()) for uav in proposal))
            phy_union = set().union(*(physical_by_uav.get(uav, set()) for uav in proposal))
            link_union = set().union(*(link_by_uav.get(uav, set()) for uav in proposal))
            options_by_source = valid_options_for_selection(case_options, proposal_set)
            net_pois: set[str] = set()
            for source in proposal:
                if options_by_source.get(source):
                    net_pois |= link_by_uav.get(source, set())

            if operator == "O1":
                proxy = (
                    _weighted_sum(phy_union, weight_map),
                    _weighted_sum(link_union, weight_map),
                    _weighted_sum(nominal_union, weight_map),
                )
            elif operator == "O2":
                option_count = sum(len(value) for value in options_by_source.values())
                distinct_gbs = len(
                    {
                        str(option["gbs_id"])
                        for values in options_by_source.values()
                        for option in values
                    }
                )
                proxy = (
                    _weighted_sum(net_pois, weight_map),
                    float(option_count),
                    float(distinct_gbs),
                )
            else:
                residual_potential = 0.0
                distinct_gbs: set[str] = set()
                for poi_id in failed_app & net_pois:
                    demand_value = float(demand_map[poi_id])
                    best_residual = 0.0
                    for source in proposal:
                        if poi_id not in link_by_uav.get(source, set()):
                            continue
                        for option in options_by_source.get(source, []):
                            chain = {**option, "source_uav_id": source}
                            resources = [(f"uav:{source}", float(uav_capacity_mbps))]
                            relay = str(option.get("relay_uav_id", ""))
                            if relay:
                                resources.append((f"uav:{relay}", float(uav_capacity_mbps)))
                            gbs = str(option["gbs_id"])
                            resources.append((f"gbs:{gbs}", float(gbs_capacity_mbps)))
                            distinct_gbs.add(gbs)
                            for edge in chain_edges(chain):
                                resources.append(("edge:" + ":".join(edge), float(edge_caps[edge])))
                            residual = min(
                                max(capacity - baseline_usage.get(name, 0.0), 0.0)
                                / max(demand_value, 1e-12)
                                for name, capacity in resources
                            )
                            best_residual = max(best_residual, min(residual, 1.0))
                    residual_potential += float(weight_map.get(poi_id, 0.0)) * best_residual
                proxy = (
                    residual_potential,
                    float(len(distinct_gbs)),
                    _weighted_sum(net_pois, weight_map),
                )
            scored.append((proxy, candidate_out, candidate_in, proposal))
        scored.sort(key=lambda item: (-item[0][0], -item[0][1], -item[0][2], item[1], item[2]))
        proposals[operator] = [
            CandidateProposal(
                operator=operator,
                candidate_out=out,
                candidate_in=incoming,
                selected_ids=state,
                proxy_primary=float(proxy[0]),
                proxy_secondary=float(proxy[1]),
                proxy_tertiary=float(proxy[2]),
                rank=rank,
            )
            for rank, (proxy, out, incoming, state) in enumerate(
                scored[:budget_per_operator], start=1
            )
        ]
    return proposals
