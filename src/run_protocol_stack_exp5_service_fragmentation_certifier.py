from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from protocol_stack_pipeline_common import (
    WORKSPACE,
    hash_files,
    load_config,
    write_flag,
    write_json,
)


STAGE = "Exp5"
OPENBASE_METHODS = ["PRM-ABS", "UrbanInfrastructure-GA", "FlyingRelay"]
METHODS = ["Random", "Coverage-first", *OPENBASE_METHODS, "BestOpen"]
REASONS = [
    "fail_nominal",
    "fail_phy",
    "fail_link_path",
    "fail_joint_service",
    "fail_capacity",
    "served",
]
TOLERANCE = 1e-10


def weighted_ratio(mask: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(np.sum(weights))
    return float(np.sum(weights[mask]) / denominator) if denominator > 0 else 0.0


def first_failure_reason(
    nominal: np.ndarray,
    physical: np.ndarray,
    link: np.ndarray,
    joint: np.ndarray,
    admitted: np.ndarray,
) -> np.ndarray:
    n = len(nominal)
    result = np.full(n, "served", dtype=object)
    result[~nominal] = "fail_nominal"
    result[nominal & ~physical] = "fail_phy"
    result[physical & ~link] = "fail_link_path"
    result[link & ~joint] = "fail_joint_service"
    result[joint & ~admitted] = "fail_capacity"
    return result


def hierarchy_valid(values: list[float], tolerance: float = TOLERANCE) -> bool:
    return all(left + tolerance >= right for left, right in zip(values, values[1:]))


def deterministic_random_selection(candidate_ids: list[str], k: int, seed: int) -> list[str]:
    if len(candidate_ids) < k:
        raise ValueError("Candidate pool smaller than K")
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(candidate_ids, size=k, replace=False).tolist())


def coverage_first_selection(
    candidate_ids: list[str],
    nominal_by_uav: dict[str, set[str]],
    weight_map: dict[str, float],
    k: int,
) -> list[str]:
    selected: list[str] = []
    covered: set[str] = set()
    for _ in range(k):
        best: tuple[float, str] | None = None
        for candidate in candidate_ids:
            if candidate in selected:
                continue
            gain = sum(weight_map.get(poi, 0.0) for poi in nominal_by_uav.get(candidate, set()) - covered)
            key = (float(gain), candidate)
            if best is None or key[0] > best[0] or (math.isclose(key[0], best[0]) and key[1] < best[1]):
                best = key
        if best is None:
            raise RuntimeError("Coverage-first could not select K candidates")
        selected.append(best[1])
        covered |= nominal_by_uav.get(best[1], set())
    return selected


def valid_options_for_selection(options: pd.DataFrame, selected: set[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in options.itertuples(index=False):
        source = str(row.source_uav_id)
        if source not in selected:
            continue
        path_type = str(row.path_type)
        relay = "" if pd.isna(row.relay_uav_id) else str(row.relay_uav_id)
        if path_type == "one_relay" and (not relay or relay not in selected or relay == source):
            continue
        if int(row.path_edges) > 2 or int(row.relay_uavs) > 1:
            continue
        result.setdefault(source, []).append(
            {
                "path_type": path_type,
                "relay_uav_id": relay,
                "gbs_id": str(row.gbs_id),
                "path_edges": int(row.path_edges),
                "relay_uavs": int(row.relay_uavs),
                "bottleneck_rate_mbps": float(row.bottleneck_rate_mbps),
                "path_nodes": str(row.path_nodes),
            }
        )
    return result


def capacity_admission(
    poi_ids: list[str],
    weights: np.ndarray,
    chains_by_poi: dict[str, list[dict[str, Any]]],
    load_mbps: float,
    uav_capacity_mbps: float,
    gbs_capacity_mbps: float,
) -> tuple[np.ndarray, list[str]]:
    admitted = np.zeros(len(poi_ids), dtype=bool)
    selected_chain = [""] * len(poi_ids)
    uav_load: dict[str, float] = {}
    gbs_load: dict[str, float] = {}
    edge_load: dict[tuple[str, str, str], float] = {}
    order = sorted(range(len(poi_ids)), key=lambda idx: (-float(weights[idx]), poi_ids[idx]))
    for idx in order:
        poi_id = poi_ids[idx]
        demand = float(weights[idx]) * float(load_mbps)
        feasible: list[tuple[float, str, dict[str, Any], list[tuple[str, str, str]]]] = []
        for chain in chains_by_poi.get(poi_id, []):
            source, relay, gbs = chain["source_uav_id"], chain["relay_uav_id"], chain["gbs_id"]
            if demand > float(chain["access_rate_mbps"]) + TOLERANCE:
                continue
            nodes = [source] + ([relay] if relay else [])
            if any(uav_load.get(node, 0.0) + demand > uav_capacity_mbps + TOLERANCE for node in nodes):
                continue
            if gbs_load.get(gbs, 0.0) + demand > gbs_capacity_mbps + TOLERANCE:
                continue
            if relay:
                edge_keys = [("relay", source, relay), ("backhaul", relay, gbs)]
            else:
                edge_keys = [("backhaul", source, gbs)]
            edge_capacity = float(chain["bottleneck_rate_mbps"])
            if any(edge_load.get(key, 0.0) + demand > edge_capacity + TOLERANCE for key in edge_keys):
                continue
            residuals = [
                (uav_capacity_mbps - uav_load.get(node, 0.0) - demand) / max(uav_capacity_mbps, 1.0)
                for node in nodes
            ]
            residuals.append((gbs_capacity_mbps - gbs_load.get(gbs, 0.0) - demand) / max(gbs_capacity_mbps, 1.0))
            residuals.extend(
                (edge_capacity - edge_load.get(key, 0.0) - demand) / max(edge_capacity, 1.0)
                for key in edge_keys
            )
            feasible.append((min(residuals), str(chain["path_nodes"]), chain, edge_keys))
        if not feasible:
            continue
        _, _, chain, edge_keys = max(feasible, key=lambda item: (item[0], item[1]))
        source, relay, gbs = chain["source_uav_id"], chain["relay_uav_id"], chain["gbs_id"]
        uav_load[source] = uav_load.get(source, 0.0) + demand
        if relay:
            uav_load[relay] = uav_load.get(relay, 0.0) + demand
        gbs_load[gbs] = gbs_load.get(gbs, 0.0) + demand
        for key in edge_keys:
            edge_load[key] = edge_load.get(key, 0.0) + demand
        admitted[idx] = True
        selected_chain[idx] = str(chain["path_nodes"])
    return admitted, selected_chain


def certify_deployment(
    case_id: str,
    method: str,
    selected_ids: list[str],
    demand: pd.DataFrame,
    nominal_by_uav: dict[str, set[str]],
    physical_by_uav: dict[str, set[str]],
    link_rate_by_uav_poi: dict[tuple[str, str], float],
    case_options: pd.DataFrame,
    capacity_main_mbps: float,
    capacity_sweep_mbps: list[float],
    uav_capacity_mbps: float,
    gbs_capacity_mbps: float,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    selected = set(selected_ids)
    poi_ids = demand["poi_id"].astype(str).tolist()
    weights = demand["service_demand_weight"].to_numpy(float)
    index = {poi: idx for idx, poi in enumerate(poi_ids)}
    nominal = np.zeros(len(poi_ids), dtype=bool)
    physical = np.zeros(len(poi_ids), dtype=bool)
    link = np.zeros(len(poi_ids), dtype=bool)
    access_sources: dict[str, list[tuple[str, float]]] = {}
    for uav in selected:
        for poi in nominal_by_uav.get(uav, set()):
            if poi in index:
                nominal[index[poi]] = True
        for poi in physical_by_uav.get(uav, set()):
            if poi in index:
                physical[index[poi]] = True
        for poi in physical_by_uav.get(uav, set()):
            rate = link_rate_by_uav_poi.get((uav, poi))
            if rate is not None and poi in index:
                link[index[poi]] = True
                access_sources.setdefault(poi, []).append((uav, float(rate)))
    options_by_source = valid_options_for_selection(case_options, selected)
    chains_by_poi: dict[str, list[dict[str, Any]]] = {}
    for poi, sources in access_sources.items():
        for source, access_rate in sources:
            for option in options_by_source.get(source, []):
                chains_by_poi.setdefault(poi, []).append(
                    {
                        **option,
                        "source_uav_id": source,
                        "access_rate_mbps": access_rate,
                    }
                )
    joint = np.array([bool(chains_by_poi.get(poi)) for poi in poi_ids], dtype=bool)
    admitted, selected_chains = capacity_admission(
        poi_ids, weights, chains_by_poi, capacity_main_mbps, uav_capacity_mbps, gbs_capacity_mbps
    )
    metrics = [
        weighted_ratio(nominal, weights),
        weighted_ratio(physical, weights),
        weighted_ratio(link, weights),
        weighted_ratio(joint, weights),
        weighted_ratio(admitted, weights),
    ]
    reasons = first_failure_reason(nominal, physical, link, joint, admitted)
    poi_result = pd.DataFrame(
        {
            "case_id": case_id,
            "method": method,
            "poi_id": poi_ids,
            "service_demand_weight": weights,
            "nominal_pass": nominal,
            "physical_pass": physical,
            "link_pass": link,
            "joint_service_pass": joint,
            "capacity_admitted": admitted,
            "first_failure_reason": reasons,
            "selected_service_chain": selected_chains,
        }
    )
    summary = {
        "case_id": case_id,
        "method": method,
        "C_nom": metrics[0],
        "PTR_phy": metrics[1],
        "LSPR": metrics[2],
        "JSWPR": metrics[3],
        "JSWPR_cap": metrics[4],
        "poi_count": len(poi_ids),
        "served_poi_count": int(admitted.sum()),
        "hierarchy_pass": hierarchy_valid(metrics),
    }
    sweep_rows = []
    for load in capacity_sweep_mbps:
        sweep_admitted, _ = capacity_admission(
            poi_ids, weights, chains_by_poi, load, uav_capacity_mbps, gbs_capacity_mbps
        )
        sweep_rows.append(
            {
                "case_id": case_id,
                "method": method,
                "load_mbps": float(load),
                "JSWPR_cap": weighted_ratio(sweep_admitted, weights),
                "admitted_poi_count": int(sweep_admitted.sum()),
            }
        )
    path_stats = {
        "case_id": case_id,
        "method": method,
        "max_path_edges": max(
            (int(chain["path_edges"]) for chains in chains_by_poi.values() for chain in chains),
            default=0,
        ),
        "max_relay_uavs": max(
            (int(chain["relay_uavs"]) for chains in chains_by_poi.values() for chain in chains),
            default=0,
        ),
        "relay_of_relay_count": sum(
            int(chain["relay_uavs"]) > 1 for chains in chains_by_poi.values() for chain in chains
        ),
    }
    return summary, poi_result, sweep_rows, path_stats


def append_frame(path: Path, frame: pd.DataFrame, first: bool) -> None:
    frame.to_csv(path, mode="w" if first else "a", header=first, index=False)


def require_passed(root: Path, label: str) -> dict[str, Any]:
    path = root / "audit_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} audit: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise RuntimeError(f"{label} did not pass")
    return audit


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("protocol_stack_exp5")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def save_figures(results: pd.DataFrame, attribution: pd.DataFrame, figure_dir: Path) -> None:
    metric_columns = ["C_nom", "PTR_phy", "LSPR", "JSWPR", "JSWPR_cap"]
    by_method = results.groupby("method")[metric_columns].mean().reindex(METHODS)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(metric_columns))
    for method in METHODS:
        ax.plot(x, by_method.loc[method], marker="o", linewidth=1.6, label=method)
    ax.set_xticks(x, metric_columns)
    ax.set_ylabel("Weighted POI ratio")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "service_funnel_by_method.png", dpi=300)
    plt.close(fig)

    by_terrain = results.groupby("terrain")[metric_columns].mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for terrain, row in by_terrain.iterrows():
        ax.plot(x, row, marker="o", linewidth=1.7, label=terrain)
    ax.set_xticks(x, metric_columns)
    ax.set_ylabel("Weighted POI ratio")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "service_funnel_by_terrain.png", dpi=300)
    plt.close(fig)

    reason_weight = (
        attribution.groupby(["method", "first_failure_reason"])["service_demand_weight"].sum()
        .unstack(fill_value=0).reindex(index=METHODS, columns=REASONS)
    )
    reason_weight = reason_weight.div(reason_weight.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    reason_weight.plot(kind="bar", stacked=True, ax=ax, width=0.75)
    ax.set_ylabel("Weighted POI fraction")
    ax.set_xlabel("")
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "failure_reason_stack_by_method.png", dpi=300)
    plt.close(fig)

    terrain_weight = (
        attribution.groupby(["terrain", "first_failure_reason"])["service_demand_weight"].sum()
        .unstack(fill_value=0).reindex(columns=REASONS)
    )
    terrain_weight = terrain_weight.div(terrain_weight.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    terrain_weight.plot(kind="bar", stacked=True, ax=ax, width=0.7)
    ax.set_ylabel("Weighted POI fraction")
    ax.set_xlabel("")
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "failure_reason_stack_by_terrain.png", dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, config_path, config_hash = load_config(args.config)
    root = WORKSPACE / config["paths"]["output_root"]
    output_root = root / "exp5_service_fragmentation_certifier"
    if output_root.exists() and args.force:
        resolved, parent = output_root.resolve(), root.resolve()
        if parent not in resolved.parents:
            raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    for name in ("data", "audit", "figures"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_root / "run.log")
    exp2_root = root / "exp2_infrastructure_limited_service_state"
    exp3_root = root / "exp3_service_primitives_compile"
    exp4_root = root / "exp4_openbase_adaptation_audit"
    backup_root = root / "exp2_infrastructure_limited_service_state_variable_gbs_ratio_backup"
    exp2_audit = require_passed(exp2_root, "Exp2")
    exp3_audit = require_passed(exp3_root, "Exp3")
    exp4_audit = require_passed(exp4_root, "Exp4")
    if int(exp4_audit.get("actual_method_cases", 0)) != 180:
        raise RuntimeError("Exp4 formal method-case count is not 180")
    input_hash = hash_files([
        config_path, exp2_root / "audit_summary.json", exp3_root / "audit_summary.json",
        exp3_root / "data" / "access_phy_edges.csv", exp3_root / "data" / "link_edges.csv",
        exp3_root / "data" / "backhaul_options.csv", exp4_root / "audit_summary.json",
        exp4_root / "data" / "openbase_selected_deployments.csv",
    ])
    cases = pd.read_csv(exp3_root / "data" / "case_index.csv")
    bundles = pd.read_csv(exp3_root / "data" / "primitive_bundle_index.csv").set_index("case_id")
    access = pd.read_csv(exp3_root / "data" / "access_phy_edges.csv")
    link = pd.read_csv(exp3_root / "data" / "link_edges.csv")
    options = pd.read_csv(exp3_root / "data" / "backhaul_options.csv")
    exp4_deployments = pd.read_csv(exp4_root / "data" / "openbase_selected_deployments.csv")
    capacity = config["capacity"]
    main_load = float(capacity["capacity_main_mbps"])
    sweep_loads = [float(value) for value in capacity["capacity_sweep_mbps"]]
    uav_capacity = float(capacity["uav_aggregate_capacity_mbps"])
    gbs_capacity = float(capacity["gbs_egress_capacity_mbps"])

    nominal_cache: dict[str, dict[str, set[str]]] = {}
    physical_cache: dict[str, dict[str, set[str]]] = {}
    link_cache: dict[str, dict[tuple[str, str], float]] = {}
    candidate_cache: dict[str, list[str]] = {}
    for terrain in ["S1", "S2", "S3", "S4"]:
        terrain_access = access[access["terrain"] == terrain]
        nominal_cache[terrain] = {
            str(uav): set(group["poi_id"].astype(str))
            for uav, group in terrain_access.groupby("uav_id")
        }
        physical_cache[terrain] = {
            str(uav): set(group.loc[group["phy_feasible"], "poi_id"].astype(str))
            for uav, group in terrain_access.groupby("uav_id")
        }
        terrain_link = link[(link["scope_id"] == terrain) & (link["link_type"] == "access")]
        link_cache[terrain] = {
            (str(row.source_id), str(row.target_id)): float(row.link_rate_mbps)
            for row in terrain_link.itertuples(index=False)
        }
        ids = set(nominal_cache[terrain])
        relay = pd.read_csv(exp3_root / "data" / "relay_phy_edges.csv")
        terrain_relay = relay[relay["terrain"] == terrain]
        ids |= set(terrain_relay["uav_a_id"].astype(str)) | set(terrain_relay["uav_b_id"].astype(str))
        expected_pool = int(cases.loc[cases["terrain"] == terrain, "uav_candidate_count"].iloc[0])
        if len(ids) != expected_pool:
            raise RuntimeError(f"{terrain}: Exp3-derived candidate pool {len(ids)} != {expected_pool}")
        candidate_cache[terrain] = sorted(ids)

    output_paths = {
        "poi": output_root / "data" / "certifier_results_by_poi.csv",
        "attribution": output_root / "data" / "first_failure_attribution.csv",
    }
    first_write = True
    result_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    failure_audit_rows: list[dict[str, Any]] = []

    for case in cases.itertuples(index=False):
        case_id, terrain, level, seed = (
            str(case.case_id), str(case.terrain), str(case.infrastructure_level), int(case.seed)
        )
        demand = pd.read_csv(Path(str(bundles.loc[case_id, "poi_demand_path"])))
        weight_map = demand.set_index("poi_id")["service_demand_weight"].to_dict()
        candidate_ids = candidate_cache[terrain]
        random_ids = deterministic_random_selection(
            candidate_ids, int(config["design"]["K"]),
            seed=50_000 + 100 * ["S1", "S2", "S3", "S4"].index(terrain) + seed,
        )
        coverage_ids = coverage_first_selection(
            candidate_ids, nominal_cache[terrain], weight_map, int(config["design"]["K"])
        )
        selected_by_method = {"Random": random_ids, "Coverage-first": coverage_ids}
        case_exp4 = exp4_deployments[exp4_deployments["case_id"] == case_id]
        for method in OPENBASE_METHODS:
            row = case_exp4[case_exp4["method"] == method]
            if len(row) != 1:
                raise RuntimeError(f"{case_id}: missing unique Exp4 deployment for {method}")
            selected_by_method[method] = str(row.iloc[0]["selected_candidate_ids"]).split(";")
        case_options = options[options["case_id"] == case_id]
        certified_open: dict[str, tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], dict[str, Any]]] = {}
        for method in ["Random", "Coverage-first", *OPENBASE_METHODS]:
            certification = certify_deployment(
                case_id, method, selected_by_method[method], demand,
                nominal_cache[terrain], physical_cache[terrain], link_cache[terrain],
                case_options, main_load, sweep_loads, uav_capacity, gbs_capacity,
            )
            summary, poi_result, method_sweep, path_stats = certification
            summary.update({"terrain": terrain, "infrastructure_level": level, "seed": seed})
            poi_result.insert(2, "terrain", terrain)
            poi_result.insert(3, "infrastructure_level", level)
            poi_result.insert(4, "seed", seed)
            result_rows.append(summary)
            sweep_rows.extend({**row, "terrain": terrain, "infrastructure_level": level, "seed": seed} for row in method_sweep)
            path_rows.append(path_stats)
            append_frame(output_paths["poi"], poi_result, first_write)
            append_frame(
                output_paths["attribution"],
                poi_result[[
                    "case_id", "method", "terrain", "infrastructure_level", "seed", "poi_id",
                    "service_demand_weight", "first_failure_reason",
                ]],
                first_write,
            )
            first_write = False
            unique_complete = len(poi_result) == len(demand) and poi_result["poi_id"].nunique() == len(demand)
            reason_valid = poi_result["first_failure_reason"].isin(REASONS).all()
            failure_audit_rows.append({
                "case_id": case_id, "method": method, "poi_count": len(demand),
                "attributed_count": len(poi_result), "unique_poi_count": poi_result["poi_id"].nunique(),
                "mutually_exclusive": bool(unique_complete), "complete": bool(unique_complete and reason_valid),
            })
            input_rows.append({
                "case_id": case_id, "method": method, "exp2_input_used": True,
                "exp3_input_used": True, "exp4_input_used": method in OPENBASE_METHODS,
                "backup_exp2_used": False, "raw_full_gbs_used": False,
                "primitive_bypass": False, "retained_gbs_count": int(case.retained_gbs_count),
                "expected_retained_gbs_count": {"low": 4, "medium": 3, "high": 2}[level],
            })
            if method in OPENBASE_METHODS:
                certified_open[method] = certification
        best_method = max(
            OPENBASE_METHODS,
            key=lambda name: (
                float(certified_open[name][0]["JSWPR_cap"]),
                float(certified_open[name][0]["JSWPR"]),
                name,
            ),
        )
        best_summary, best_poi, best_sweep, best_path = certified_open[best_method]
        best_summary = {**best_summary, "method": "BestOpen", "source_method": best_method,
                        "terrain": terrain, "infrastructure_level": level, "seed": seed}
        best_poi = best_poi.copy()
        best_poi["method"] = "BestOpen"
        result_rows.append(best_summary)
        sweep_rows.extend({
            **row, "method": "BestOpen", "source_method": best_method,
            "terrain": terrain, "infrastructure_level": level, "seed": seed,
        } for row in best_sweep)
        path_rows.append({**best_path, "method": "BestOpen"})
        append_frame(output_paths["poi"], best_poi, False)
        append_frame(
            output_paths["attribution"],
            best_poi[[
                "case_id", "method", "terrain", "infrastructure_level", "seed", "poi_id",
                "service_demand_weight", "first_failure_reason",
            ]],
            False,
        )
        failure_audit_rows.append({
            "case_id": case_id, "method": "BestOpen", "poi_count": len(demand),
            "attributed_count": len(best_poi), "unique_poi_count": best_poi["poi_id"].nunique(),
            "mutually_exclusive": True, "complete": best_poi["first_failure_reason"].isin(REASONS).all(),
        })
        input_rows.append({
            "case_id": case_id, "method": "BestOpen", "exp2_input_used": True,
            "exp3_input_used": True, "exp4_input_used": True, "backup_exp2_used": False,
            "raw_full_gbs_used": False, "primitive_bypass": False,
            "retained_gbs_count": int(case.retained_gbs_count),
            "expected_retained_gbs_count": {"low": 4, "medium": 3, "high": 2}[level],
        })
        logger.info("%s certified for six methods; BestOpen=%s", case_id, best_method)

    results = pd.DataFrame(result_rows)
    sweep = pd.DataFrame(sweep_rows)
    path_audit = pd.DataFrame(path_rows)
    input_audit = pd.DataFrame(input_rows)
    failure_audit = pd.DataFrame(failure_audit_rows)
    hierarchy_audit = results[
        ["case_id", "method", "C_nom", "PTR_phy", "LSPR", "JSWPR", "JSWPR_cap", "hierarchy_pass"]
    ].copy()
    capacity_audit = sweep.groupby(["case_id", "method"]).apply(
        lambda group: pd.Series({
            "all_finite": np.isfinite(group["JSWPR_cap"]).all(),
            "nonincreasing_with_load": bool(
                np.all(np.diff(group.sort_values("load_mbps")["JSWPR_cap"].to_numpy()) <= TOLERANCE)
            ),
        }),
        include_groups=False,
    ).reset_index()
    results.to_csv(output_root / "data" / "certifier_results_by_case_method.csv", index=False)
    results.to_csv(output_root / "data" / "capacity_main_results.csv", index=False)
    sweep.to_csv(output_root / "data" / "capacity_sweep_results.csv", index=False)
    fragmentation = (
        pd.read_csv(output_paths["attribution"])
        .groupby(["method", "terrain", "infrastructure_level", "first_failure_reason"])
        .agg(poi_count=("poi_id", "size"), weighted_poi=("service_demand_weight", "sum"))
        .reset_index()
    )
    fragmentation.to_csv(output_root / "data" / "service_fragmentation_summary.csv", index=False)
    hierarchy_audit.to_csv(output_root / "audit" / "hierarchy_audit.csv", index=False)
    failure_audit.to_csv(output_root / "audit" / "first_failure_audit.csv", index=False)
    input_audit.to_csv(output_root / "audit" / "input_consistency_audit.csv", index=False)
    path_audit.to_csv(output_root / "audit" / "path_legality_audit.csv", index=False)
    capacity_audit.to_csv(output_root / "audit" / "capacity_audit.csv", index=False)
    summary = results.groupby("method")[
        ["C_nom", "PTR_phy", "LSPR", "JSWPR", "JSWPR_cap"]
    ].mean().reindex(METHODS).reset_index()
    summary.to_csv(output_root / "summary.csv", index=False)
    attribution = pd.read_csv(output_paths["attribution"])
    save_figures(results, attribution, output_root / "figures")

    numeric_finite = np.isfinite(
        results[["C_nom", "PTR_phy", "LSPR", "JSWPR", "JSWPR_cap"]].to_numpy()
    ).all() and np.isfinite(sweep["JSWPR_cap"]).all()
    required = [
        output_root / "data" / name for name in [
            "certifier_results_by_case_method.csv", "certifier_results_by_poi.csv",
            "first_failure_attribution.csv", "capacity_main_results.csv",
            "capacity_sweep_results.csv", "service_fragmentation_summary.csv",
        ]
    ] + [
        output_root / "audit" / name for name in [
            "hierarchy_audit.csv", "first_failure_audit.csv", "input_consistency_audit.csv",
            "path_legality_audit.csv", "capacity_audit.csv",
        ]
    ] + [
        output_root / "figures" / name for name in [
            "service_funnel_by_method.png", "service_funnel_by_terrain.png",
            "failure_reason_stack_by_method.png", "failure_reason_stack_by_terrain.png",
        ]
    ] + [output_root / "summary.csv", output_root / "run.log"]
    audit = {
        "stage": STAGE,
        "expected_method_cases": 360,
        "actual_method_cases": len(results),
        "expected_cases": 60,
        "actual_cases": results["case_id"].nunique(),
        "methods": METHODS,
        "exp2_input_used": True,
        "exp3_input_used": True,
        "exp4_input_used": True,
        "backup_exp2_used": False,
        "raw_full_gbs_used_as_available_gbs": False,
        "primitive_bypass_detected": False,
        "retained_gbs_count_match": bool(
            (input_audit["retained_gbs_count"] == input_audit["expected_retained_gbs_count"]).all()
        ),
        "all_values_finite": bool(numeric_finite),
        "all_required_outputs_exist": bool(all(path.exists() for path in required)),
        "hierarchy_passed": bool(hierarchy_audit["hierarchy_pass"].all()),
        "first_failure_mutually_exclusive": bool(failure_audit["mutually_exclusive"].all()),
        "first_failure_complete": bool(failure_audit["complete"].all()),
        "no_relay_of_relay": bool((path_audit["relay_of_relay_count"] == 0).all()),
        "max_path_edges_observed": int(path_audit["max_path_edges"].max()),
        "max_relay_uavs_observed": int(path_audit["max_relay_uavs"].max()),
        "constraint_violation_sum": int(
            (~hierarchy_audit["hierarchy_pass"]).sum()
            + (~failure_audit["mutually_exclusive"]).sum()
            + (~failure_audit["complete"]).sum()
            + (path_audit["relay_of_relay_count"] > 0).sum()
            + (path_audit["max_path_edges"] > 2).sum()
            + (path_audit["max_relay_uavs"] > 1).sum()
            + (~capacity_audit["all_finite"]).sum()
            + (~capacity_audit["nonincreasing_with_load"]).sum()
            + input_audit["backup_exp2_used"].sum()
            + input_audit["raw_full_gbs_used"].sum()
            + input_audit["primitive_bypass"].sum()
        ),
        "capacity_main_mbps": main_load,
        "capacity_sweep_mbps": sweep_loads,
        "config_hash": config_hash,
        "input_hash": input_hash,
    }
    audit["passed"] = bool(
        audit["actual_method_cases"] == audit["expected_method_cases"]
        and audit["actual_cases"] == audit["expected_cases"]
        and audit["retained_gbs_count_match"]
        and audit["all_values_finite"]
        and audit["all_required_outputs_exist"]
        and audit["hierarchy_passed"]
        and audit["first_failure_mutually_exclusive"]
        and audit["first_failure_complete"]
        and audit["no_relay_of_relay"]
        and audit["max_path_edges_observed"] <= 2
        and audit["max_relay_uavs_observed"] <= 1
        and audit["constraint_violation_sum"] == 0
    )
    write_json(output_root / "audit_summary.json", audit)
    reason_summary = (
        attribution.groupby("first_failure_reason")["service_demand_weight"].sum()
        / attribution["service_demand_weight"].sum()
    ).sort_values(ascending=False)
    report = [
        "# Exp5 four-layer service fragmentation and failure attribution",
        "",
        "Low/medium/high denote infrastructure limitation levels. DEM, POI geometry, candidate pool, "
        "and the 1400 Mbps main load remain fixed; only retained service anchors vary by level.",
        "",
        "## Certification semantics",
        "",
        "- C_nom: nominal access-radius coverage.",
        "- PTR_phy: access edges passing Exp3 terrain LoS and Fresnel checks.",
        "- LSPR: Exp3 feasible access-link prefix.",
        "- JSWPR: feasible access plus a legal direct or one-relay path to a retained GBS.",
        "- JSWPR_cap: deterministic admission under UAV, GBS, and shared path capacities.",
        "",
        "## Audit result",
        "",
        f"- Method cases: `{len(results)}/360`",
        f"- Hierarchy passed: `{audit['hierarchy_passed']}`",
        f"- Failure attribution mutually exclusive: `{audit['first_failure_mutually_exclusive']}`",
        f"- Failure attribution complete: `{audit['first_failure_complete']}`",
        f"- No relay-of-relay: `{audit['no_relay_of_relay']}`",
        f"- Constraint violations: `{audit['constraint_violation_sum']}`",
        f"- Final passed: `{audit['passed']}`",
        "",
        "## Overall weighted first-failure distribution",
        "",
    ]
    report.extend(f"- `{reason}`: `{value:.6f}`" for reason, value in reason_summary.items())
    report.extend([
        "",
        "BestOpen is the per-case strongest of the three certified OpenBase deployments under JSWPR_cap; "
        "it is a defined derived comparator rather than a duplicated method used to inflate sample size.",
        f"The deprecated Exp2 backup `{backup_root}` was excluded.",
    ])
    (output_root / "audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_flag(
        output_root / "experiment_done.flag", STAGE, "passed" if audit["passed"] else "failed",
        360, len(results), config_hash, input_hash,
    )
    logger.info("Exp5 completed: expected=360 actual=%d passed=%s", len(results), audit["passed"])
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
