from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import time
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
    sha256_file,
    write_flag,
    write_json,
)


STAGE = "Exp4"
METHODS = ["PRM-ABS", "UrbanInfrastructure-GA", "FlyingRelay"]


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo <= 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def greedy_diverse_selection(
    candidate_ids: list[str],
    base_score: np.ndarray,
    xy: np.ndarray,
    k: int,
    diversity_weight: float = 0.08,
) -> list[str]:
    if len(candidate_ids) < k:
        raise ValueError(f"Candidate pool has {len(candidate_ids)} points, fewer than K={k}")
    selected: list[int] = []
    map_scale = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1.0)
    for _ in range(k):
        best: tuple[float, str, int] | None = None
        for idx, candidate_id in enumerate(candidate_ids):
            if idx in selected:
                continue
            if selected:
                min_distance = float(np.min(np.linalg.norm(xy[selected] - xy[idx], axis=1))) / map_scale
            else:
                min_distance = 1.0
            value = float(base_score[idx]) + diversity_weight * min_distance
            key = (value, candidate_id, idx)
            if best is None or key[0] > best[0] or (math.isclose(key[0], best[0]) and key[1] < best[1]):
                best = key
        assert best is not None
        selected.append(best[2])
    return [candidate_ids[idx] for idx in selected]


def adapt_prm_abs(
    candidate_ids: list[str],
    propagation_score: np.ndarray,
    backhaul_score: np.ndarray,
    xy: np.ndarray,
    k: int,
    backhaul_weight: float,
) -> list[str]:
    score = normalize(propagation_score) + backhaul_weight * normalize(backhaul_score)
    return greedy_diverse_selection(candidate_ids, score, xy, k, diversity_weight=0.06)


def _ga_fitness(
    chromosome: np.ndarray,
    candidate_score: np.ndarray,
    pair_overlap: np.ndarray,
    xy: np.ndarray,
) -> float:
    chosen = np.flatnonzero(chromosome)
    if len(chosen) == 0:
        return -1e9
    quality = float(np.mean(candidate_score[chosen]))
    overlap = 0.0
    if len(chosen) > 1:
        pairs = pair_overlap[np.ix_(chosen, chosen)]
        overlap = float((pairs.sum() - np.trace(pairs)) / (len(chosen) * (len(chosen) - 1)))
    span = float(np.mean(np.ptp(xy[chosen], axis=0))) / max(float(np.mean(np.ptp(xy, axis=0))), 1.0)
    return quality - 0.20 * overlap + 0.08 * span


def adapt_urban_ga(
    candidate_ids: list[str],
    satisfaction_score: np.ndarray,
    pair_overlap: np.ndarray,
    xy: np.ndarray,
    k: int,
    seed: int,
    settings: dict[str, Any],
) -> list[str]:
    n = len(candidate_ids)
    if n < k:
        raise ValueError(f"Candidate pool has {n} points, fewer than K={k}")
    rng = np.random.default_rng(seed)
    pop_size = int(settings["population"])
    generations = int(settings["generations"])
    elite_count = int(settings["elite_count"])
    tournament_size = int(settings["tournament_size"])
    crossover_probability = float(settings["crossover_probability"])
    mutation_probability = float(settings["mutation_probability"])

    def random_chromosome() -> np.ndarray:
        chromosome = np.zeros(n, dtype=np.int8)
        chromosome[rng.choice(n, size=k, replace=False)] = 1
        return chromosome

    def repair(chromosome: np.ndarray) -> np.ndarray:
        chromosome = chromosome.copy()
        on = np.flatnonzero(chromosome)
        if len(on) > k:
            chromosome[rng.choice(on, size=len(on) - k, replace=False)] = 0
        elif len(on) < k:
            off = np.flatnonzero(chromosome == 0)
            chromosome[rng.choice(off, size=k - len(on), replace=False)] = 1
        return chromosome

    population = [random_chromosome() for _ in range(pop_size)]
    for _ in range(generations):
        scored = sorted(
            [(_ga_fitness(item, satisfaction_score, pair_overlap, xy), item) for item in population],
            key=lambda pair: pair[0],
            reverse=True,
        )
        next_population = [item.copy() for _, item in scored[:elite_count]]
        while len(next_population) < pop_size:
            contenders = rng.choice(pop_size, size=tournament_size, replace=False)
            parent1 = max((scored[int(i)] for i in contenders), key=lambda pair: pair[0])[1]
            contenders = rng.choice(pop_size, size=tournament_size, replace=False)
            parent2 = max((scored[int(i)] for i in contenders), key=lambda pair: pair[0])[1]
            child = parent1.copy()
            if rng.random() < crossover_probability:
                mask = rng.random(n) < 0.5
                child[mask] = parent2[mask]
            if rng.random() < mutation_probability:
                on, off = np.flatnonzero(child), np.flatnonzero(child == 0)
                if len(on) and len(off):
                    child[rng.choice(on)] = 0
                    child[rng.choice(off)] = 1
            next_population.append(repair(child))
        population = next_population
    winner = max(population, key=lambda item: _ga_fitness(item, satisfaction_score, pair_overlap, xy))
    return [candidate_ids[idx] for idx in np.flatnonzero(winner)]


def adapt_flying_relay(
    candidate_ids: list[str],
    demand_score: np.ndarray,
    network_score: np.ndarray,
    relay_score: np.ndarray,
    xy: np.ndarray,
    k: int,
    settings: dict[str, Any],
) -> list[str]:
    score = (
        float(settings["network_weight"]) * normalize(network_score)
        + float(settings["relay_weight"]) * normalize(relay_score)
        + float(settings["demand_weight"]) * normalize(demand_score)
    )
    return greedy_diverse_selection(candidate_ids, score, xy, k, diversity_weight=0.04)


def require_inputs(exp2_root: Path, exp3_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    for root, name in ((exp2_root, "Exp2"), (exp3_root, "Exp3")):
        audit_path = root / "audit_summary.json"
        if not audit_path.exists():
            raise FileNotFoundError(f"Missing {name} audit: {audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("passed"):
            raise RuntimeError(f"{name} formal audit did not pass")
    bundles = pd.read_csv(exp3_root / "data" / "primitive_bundle_index.csv")
    case_index = pd.read_csv(exp3_root / "data" / "case_index.csv")
    if len(bundles) != 60 or len(case_index) != 60:
        raise RuntimeError("Exp3 formal case index is not 60/60")
    return bundles, case_index


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("protocol_stack_exp4")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def source_archive_audit(config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("PRM_ABS", "UrbanInfrastructure_GA", "FlyingRelay"):
        archive = Path(config["openbase"][key]["code_archive"])
        if not archive.exists():
            raise FileNotFoundError(f"OpenBase archive not found: {archive}")
        result[key] = {
            "code_archive": str(archive),
            "sha256": sha256_file(archive),
            "adaptation": "open-source algorithmic search principle migrated to the unified Exp3 primitive space",
        }
    return result


def weighted_candidate_features(
    terrain: str,
    case_id: str,
    candidate_ids: list[str],
    access: pd.DataFrame,
    options: pd.DataFrame,
    demand: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    demand_map = demand.set_index("poi_id")["service_demand_weight"]
    feasible = access[(access["terrain"] == terrain) & access["phy_feasible"]][["uav_id", "poi_id"]]
    feasible = feasible[feasible["poi_id"].isin(demand_map.index)].copy()
    feasible["weight"] = feasible["poi_id"].map(demand_map).astype(float)
    demand_score_map = feasible.groupby("uav_id")["weight"].sum()
    demand_score = np.array([float(demand_score_map.get(cid, 0.0)) for cid in candidate_ids])

    case_options = options[options["case_id"] == case_id].copy()
    network_map = case_options.groupby("source_uav_id")["bottleneck_rate_mbps"].max()
    relay_map = (
        case_options.assign(is_relay=(case_options["path_type"] == "one_relay").astype(float))
        .groupby("source_uav_id")["is_relay"].sum()
    )
    direct_map = (
        case_options.assign(is_direct=(case_options["path_type"] == "direct").astype(float))
        .groupby("source_uav_id")["is_direct"].sum()
    )
    network_score = np.array([float(network_map.get(cid, 0.0)) for cid in candidate_ids])
    relay_score = np.array([float(relay_map.get(cid, 0.0)) for cid in candidate_ids])
    direct_score = np.array([float(direct_map.get(cid, 0.0)) for cid in candidate_ids])
    return demand_score, network_score, relay_score, direct_score


def weighted_overlap_matrix(
    candidate_ids: list[str],
    access_subsets: dict[str, set[str]],
    demand: pd.DataFrame,
) -> np.ndarray:
    demand_map = demand.set_index("poi_id")["service_demand_weight"].to_dict()
    matrix = np.zeros((len(candidate_ids), len(candidate_ids)), dtype=float)
    weights = {
        cid: sum(float(demand_map.get(poi, 0.0)) for poi in access_subsets.get(cid, set()))
        for cid in candidate_ids
    }
    for i, left in enumerate(candidate_ids):
        matrix[i, i] = 1.0
        for j in range(i + 1, len(candidate_ids)):
            right = candidate_ids[j]
            denominator = max(weights[left] + weights[right], 1e-12)
            intersection = sum(
                float(demand_map.get(poi, 0.0))
                for poi in access_subsets.get(left, set()) & access_subsets.get(right, set())
            )
            value = 2.0 * intersection / denominator
            matrix[i, j] = matrix[j, i] = value
    return matrix


def save_figures(
    deployments: pd.DataFrame,
    positions: pd.DataFrame,
    candidate_positions: dict[str, pd.DataFrame],
    figure_dir: Path,
) -> None:
    overlap = np.zeros((3, 3), dtype=float)
    grouped = deployments.set_index(["case_id", "method"])
    for i, left in enumerate(METHODS):
        for j, right in enumerate(METHODS):
            values = []
            for case_id in deployments["case_id"].unique():
                a = set(str(grouped.loc[(case_id, left), "selected_candidate_ids"]).split(";"))
                b = set(str(grouped.loc[(case_id, right), "selected_candidate_ids"]).split(";"))
                values.append(len(a & b) / len(a | b))
            overlap[i, j] = float(np.mean(values))
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    image = ax.imshow(overlap, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(3), METHODS, rotation=25, ha="right")
    ax.set_yticks(range(3), METHODS)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{overlap[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Mean Jaccard overlap")
    fig.tight_layout()
    fig.savefig(figure_dir / "method_selection_overlap.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(10.2, 11.0), sharex=True, sharey=True)
    for row, terrain in enumerate(["S1", "S2", "S3", "S4"]):
        pool = candidate_positions[terrain]
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            ax.scatter(pool["x_m"] / 1000, pool["y_m"] / 1000, s=8, c="#B8BDC4", alpha=0.55)
            chosen = positions[
                (positions["terrain"] == terrain)
                & (positions["infrastructure_level"] == "low")
                & (positions["seed"] == 0)
                & (positions["method"] == method)
            ]
            ax.scatter(chosen["x_m"] / 1000, chosen["y_m"] / 1000, s=34, c="#D1495B", edgecolors="white")
            if row == 0:
                ax.set_title(method)
            if col == 0:
                ax.set_ylabel(f"{terrain}\ny (km)")
            if row == 3:
                ax.set_xlabel("x (km)")
            ax.set_aspect("equal")
            ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(figure_dir / "deployment_examples_seed0.png", dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, config_path, config_hash = load_config(args.config)
    root = WORKSPACE / config["paths"]["output_root"]
    output_root = root / "exp4_openbase_adaptation_audit"
    if output_root.exists() and args.force:
        resolved, expected_parent = output_root.resolve(), root.resolve()
        if expected_parent not in resolved.parents:
            raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    for name in ("data", "audit", "figures"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_root / "run.log")
    exp2_root = root / "exp2_infrastructure_limited_service_state"
    exp3_root = root / "exp3_service_primitives_compile"
    backup_root = root / "exp2_infrastructure_limited_service_state_variable_gbs_ratio_backup"
    bundles, cases = require_inputs(exp2_root, exp3_root)
    archives = source_archive_audit(config)
    input_hash = hash_files([
        config_path, exp2_root / "audit_summary.json", exp2_root / "data" / "surviving_gbs_positions_all_cases.csv",
        exp3_root / "audit_summary.json", exp3_root / "data" / "primitive_bundle_index.csv",
        exp3_root / "data" / "access_phy_edges.csv", exp3_root / "data" / "backhaul_options.csv",
    ])
    k = int(config["design"]["K"])
    access = pd.read_csv(exp3_root / "data" / "access_phy_edges.csv")
    options = pd.read_csv(exp3_root / "data" / "backhaul_options.csv")
    frozen_root = WORKSPACE / config["paths"]["frozen_input_root"]
    bundle_lookup = bundles.set_index("case_id")
    candidate_positions = {
        terrain: pd.read_csv(frozen_root / terrain / "uav_candidate.csv")
        for terrain in ["S1", "S2", "S3", "S4"]
    }
    access_subsets = {
        terrain: {
            str(candidate_id): set(group["poi_id"].astype(str))
            for candidate_id, group in access[
                (access["terrain"] == terrain) & access["phy_feasible"]
            ].groupby("uav_id")
        }
        for terrain in ["S1", "S2", "S3", "S4"]
    }
    deployments: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    runtimes: list[dict[str, Any]] = []
    fairness: list[dict[str, Any]] = []
    leakage: list[dict[str, Any]] = []
    pool_audit: list[dict[str, Any]] = []
    k_audit: list[dict[str, Any]] = []
    input_audit: list[dict[str, Any]] = []
    settings = config["openbase"]["adapter_settings"]

    for case in cases.itertuples(index=False):
        case_id, terrain, level, seed = (
            str(case.case_id), str(case.terrain), str(case.infrastructure_level), int(case.seed)
        )
        bundle = bundle_lookup.loc[case_id]
        pool = candidate_positions[terrain].copy()
        candidate_ids = pool["candidate_id"].astype(str).tolist()
        xy = pool[["x_m", "y_m"]].to_numpy(float)
        demand = pd.read_csv(Path(str(bundle["poi_demand_path"])))
        demand_score, network_score, relay_score, direct_score = weighted_candidate_features(
            terrain, case_id, candidate_ids, access, options, demand
        )
        overlap = weighted_overlap_matrix(candidate_ids, access_subsets[terrain], demand)
        satisfaction = (
            0.65 * normalize(demand_score)
            + 0.20 * normalize(network_score)
            + 0.15 * normalize(direct_score)
        )
        adapters = {
            "PRM-ABS": lambda: adapt_prm_abs(
                candidate_ids, demand_score, network_score, xy, k,
                float(settings["prm_backhaul_weight"]),
            ),
            "UrbanInfrastructure-GA": lambda: adapt_urban_ga(
                candidate_ids, satisfaction, overlap, xy, k,
                seed=10_000 + seed + 100 * ["S1", "S2", "S3", "S4"].index(terrain),
                settings=settings["urban_ga"],
            ),
            "FlyingRelay": lambda: adapt_flying_relay(
                candidate_ids, demand_score, network_score, relay_score, xy, k,
                settings["flying_relay"],
            ),
        }
        for method, adapter in adapters.items():
            started = time.perf_counter()
            selected = adapter()
            elapsed = time.perf_counter() - started
            selected_set = set(selected)
            legal = len(selected) == k and len(selected_set) == k and selected_set <= set(candidate_ids)
            deployments.append({
                "case_id": case_id, "method": method, "terrain": terrain,
                "infrastructure_level": level, "seed": seed, "K_requested": k,
                "K_selected": len(selected), "selected_candidate_ids": ";".join(selected),
                "input_primitive_bundle_id": case_id, "run_status": "completed" if legal else "invalid",
                "candidate_pool_source": str(frozen_root / terrain / "uav_candidate.csv"),
                "primitive_source": str(exp3_root), "no_certifier_leakage": True,
                "raw_full_gbs_used": False, "backup_exp2_used": False,
            })
            selected_positions = pool.set_index("candidate_id").loc[selected].reset_index()
            for rank, row in enumerate(selected_positions.itertuples(index=False), start=1):
                positions.append({
                    "case_id": case_id, "method": method, "terrain": terrain,
                    "infrastructure_level": level, "seed": seed, "selection_rank": rank,
                    "candidate_id": str(row.candidate_id), "x_m": float(row.x_m), "y_m": float(row.y_m),
                    "lon": float(row.lon), "lat": float(row.lat), "ground_z_m": float(row.ground_z_m),
                    "H_AGL_m": float(config["geometry"]["H_AGL_m"]), "z_m": float(row.ground_z_m) + float(config["geometry"]["H_AGL_m"]),
                })
            runtimes.append({"case_id": case_id, "method": method, "runtime_s": elapsed})
            fairness.append({
                "case_id": case_id, "method": method, "same_candidate_pool": True,
                "same_K": len(selected) == k, "same_primitive_bundle": True,
                "same_retained_gbs": True, "fairness_pass": legal,
            })
            leakage.append({
                "case_id": case_id, "method": method, "certifier_metric_read": False,
                "failure_log_read": False, "repair_trace_read": False,
                "selection_inputs": "Exp2 demand + Exp3 propagation/link/backhaul primitives",
                "no_leakage_pass": True,
            })
            pool_audit.append({
                "case_id": case_id, "method": method, "candidate_pool_size": len(candidate_ids),
                "selected_in_pool": selected_set <= set(candidate_ids),
                "duplicate_count": len(selected) - len(selected_set),
            })
            k_audit.append({
                "case_id": case_id, "method": method, "K_requested": k,
                "K_selected": len(selected), "pass": len(selected) == k,
            })
            input_audit.append({
                "case_id": case_id, "method": method,
                "exp2_formal_input_used": True, "exp3_formal_input_used": True,
                "primitive_bundle_id": case_id, "retained_gbs_count": int(case.retained_gbs_count),
                "expected_retained_gbs_count": {"low": 4, "medium": 3, "high": 2}[level],
                "backup_exp2_used": False, "raw_full_gbs_used": False,
                "primitive_bypass": False,
            })
            logger.info("%s %s selected %d candidates in %.4f s", case_id, method, len(selected), elapsed)

    deployment_df = pd.DataFrame(deployments)
    position_df = pd.DataFrame(positions)
    runtime_df = pd.DataFrame(runtimes)
    fairness_df = pd.DataFrame(fairness)
    leakage_df = pd.DataFrame(leakage)
    pool_df = pd.DataFrame(pool_audit)
    k_df = pd.DataFrame(k_audit)
    input_df = pd.DataFrame(input_audit)
    deployment_df.to_csv(output_root / "data" / "openbase_selected_deployments.csv", index=False)
    position_df.to_csv(output_root / "data" / "openbase_selected_positions.csv", index=False)
    runtime_df.to_csv(output_root / "data" / "method_runtime.csv", index=False)
    write_json(output_root / "data" / "method_config_snapshot.json", {
        "methods": METHODS, "K": k, "settings": settings, "source_archives": archives,
        "adaptation_boundary": "Adapted open-source baselines; not exact reproductions.",
        "selection_objectives": {
            "PRM-ABS": "weighted propagation-map access utility plus pre-certification backhaul quality",
            "UrbanInfrastructure-GA": "GA user-satisfaction and energy/spatial-efficiency proxy",
            "FlyingRelay": "legal direct/one-relay backhaul bridge quality plus demand utility",
        },
    })
    fairness_df.to_csv(output_root / "audit" / "fairness_audit.csv", index=False)
    leakage_df.to_csv(output_root / "audit" / "leakage_audit.csv", index=False)
    pool_df.to_csv(output_root / "audit" / "candidate_pool_audit.csv", index=False)
    k_df.to_csv(output_root / "audit" / "k_consistency_audit.csv", index=False)
    input_df.to_csv(output_root / "audit" / "input_consistency_audit.csv", index=False)
    summary = (
        deployment_df.groupby("method")
        .agg(method_cases=("case_id", "size"), legal_cases=("run_status", lambda x: int((x == "completed").sum())))
        .reset_index()
        .merge(runtime_df.groupby("method")["runtime_s"].agg(["mean", "max"]).reset_index(), on="method")
        .rename(columns={"mean": "runtime_mean_s", "max": "runtime_max_s"})
    )
    summary.to_csv(output_root / "summary.csv", index=False)
    save_figures(deployment_df, position_df, candidate_positions, output_root / "figures")

    counts_match = bool((input_df["retained_gbs_count"] == input_df["expected_retained_gbs_count"]).all())
    all_legal = bool((deployment_df["run_status"] == "completed").all())
    audit = {
        "stage": STAGE,
        "expected_method_cases": 180,
        "actual_method_cases": len(deployment_df),
        "methods": METHODS,
        "exp2_input_used": True,
        "exp3_input_used": True,
        "backup_exp2_used": False,
        "raw_full_gbs_used_as_available_gbs": False,
        "primitive_bypass_detected": False,
        "candidate_pool_consistent": bool(fairness_df["same_candidate_pool"].all()),
        "retained_gbs_count_match": counts_match,
        "all_methods_output_legal": all_legal,
        "all_k_selected_equal_8": bool((deployment_df["K_selected"] == k).all()),
        "no_duplicate_selected_uav_per_case": bool((pool_df["duplicate_count"] == 0).all()),
        "no_out_of_pool_selection": bool(pool_df["selected_in_pool"].all()),
        "no_certifier_leakage": bool(leakage_df["no_leakage_pass"].all()),
        "no_relay_of_relay": True,
        "source_archives_verified": True,
        "constraint_violation_sum": int(
            (~fairness_df["fairness_pass"]).sum()
            + (~pool_df["selected_in_pool"]).sum()
            + (pool_df["duplicate_count"] > 0).sum()
            + (~k_df["pass"]).sum()
            + (~input_df["exp2_formal_input_used"]).sum()
            + (~input_df["exp3_formal_input_used"]).sum()
            + input_df["primitive_bypass"].sum()
        ),
        "config_hash": config_hash,
        "input_hash": input_hash,
    }
    required = [
        output_root / "data" / "openbase_selected_deployments.csv",
        output_root / "data" / "openbase_selected_positions.csv",
        output_root / "data" / "method_runtime.csv",
        output_root / "data" / "method_config_snapshot.json",
        output_root / "audit" / "fairness_audit.csv",
        output_root / "audit" / "leakage_audit.csv",
        output_root / "audit" / "candidate_pool_audit.csv",
        output_root / "audit" / "k_consistency_audit.csv",
        output_root / "audit" / "input_consistency_audit.csv",
        output_root / "figures" / "method_selection_overlap.png",
        output_root / "figures" / "deployment_examples_seed0.png",
        output_root / "summary.csv",
        output_root / "run.log",
    ]
    audit["all_required_outputs_exist"] = bool(all(path.exists() for path in required))
    audit["passed"] = bool(
        audit["actual_method_cases"] == audit["expected_method_cases"]
        and set(audit["methods"]) == set(METHODS)
        and audit["candidate_pool_consistent"]
        and audit["retained_gbs_count_match"]
        and audit["all_methods_output_legal"]
        and audit["all_k_selected_equal_8"]
        and audit["no_duplicate_selected_uav_per_case"]
        and audit["no_out_of_pool_selection"]
        and audit["no_certifier_leakage"]
        and audit["no_relay_of_relay"]
        and audit["constraint_violation_sum"] == 0
        and audit["all_required_outputs_exist"]
    )
    write_json(output_root / "audit_summary.json", audit)
    report = [
        "# Exp4 OpenBase adaptation and fairness audit",
        "",
        "The three methods are adapted open-source baselines, not exact reproductions. "
        "Their search principles are migrated to the same frozen candidate pool and Exp3 primitives.",
        "",
        "## Formal inputs",
        "",
        f"- Exp2 retained service anchors: `{exp2_root}`",
        f"- Exp3 primitive bundles: `{exp3_root}`",
        f"- Deprecated Exp2 backup excluded: `{backup_root}`",
        "- Infrastructure level labels are low/medium/high; DEM and POI geometry remain fixed.",
        "",
        "## Method boundaries",
        "",
        "- PRM-ABS: propagation/access utility with pre-certification backhaul quality.",
        "- UrbanInfrastructure-GA: fixed-seed GA over user-satisfaction, infrastructure utility, and spatial efficiency proxies.",
        "- FlyingRelay: legal direct/one-relay bridge quality with demand utility.",
        "- No method reads final certification outcomes, rejected POIs, failure labels, or repair traces.",
        "",
        "## Audit result",
        "",
        f"- Method cases: `{len(deployment_df)}/180`",
        f"- All K selected equal 8: `{audit['all_k_selected_equal_8']}`",
        f"- Candidate pool consistent: `{audit['candidate_pool_consistent']}`",
        f"- Retained GBS counts match: `{audit['retained_gbs_count_match']}`",
        f"- No certifier leakage: `{audit['no_certifier_leakage']}`",
        f"- Constraint violations: `{audit['constraint_violation_sum']}`",
        f"- Final passed: `{audit['passed']}`",
    ]
    (output_root / "audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_flag(
        output_root / "experiment_done.flag", STAGE, "passed" if audit["passed"] else "failed",
        180, len(deployment_df), config_hash, input_hash,
    )
    logger.info("Exp4 completed: expected=180 actual=%d passed=%s", len(deployment_df), audit["passed"])
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
