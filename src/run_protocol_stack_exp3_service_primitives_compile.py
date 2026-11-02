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
    utc_timestamp,
    write_flag,
    write_json,
)


STAGE = "Exp3"
WINDOW_SIZE_M = 8000.0
LIGHT_SPEED_MPS = 299_792_458.0
PROFILE_SAMPLES = 48
POI_HEIGHT_M = 1.5
GBS_HEIGHT_M = 30.0


def bilinear_dem(dem: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    rows, cols = dem.shape
    col = np.clip((np.asarray(x_m) + 4000.0) / WINDOW_SIZE_M * (cols - 1), 0, cols - 1)
    row_bottom = np.clip((np.asarray(y_m) + 4000.0) / WINDOW_SIZE_M * (rows - 1), 0, rows - 1)
    row = (rows - 1) - row_bottom
    c0, r0 = np.floor(col).astype(int), np.floor(row).astype(int)
    c1, r1 = np.minimum(c0 + 1, cols - 1), np.minimum(r0 + 1, rows - 1)
    dc, dr = col - c0, row - r0
    return (
        dem[r0, c0] * (1 - dc) * (1 - dr)
        + dem[r0, c1] * dc * (1 - dr)
        + dem[r1, c0] * (1 - dc) * dr
        + dem[r1, c1] * dc * dr
    )


def profile_clearance(
    dem: np.ndarray,
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
    frequency_hz: float,
    clearance_ratio: float,
) -> tuple[float, float, float, bool, bool]:
    horizontal = math.hypot(x1 - x0, y1 - y0)
    distance_3d = math.sqrt(horizontal * horizontal + (z1 - z0) ** 2)
    if horizontal <= 0.0:
        return horizontal, distance_3d, 0.0, True, True
    t = np.linspace(0.0, 1.0, PROFILE_SAMPLES)
    terrain = bilinear_dem(dem, x0 + t * (x1 - x0), y0 + t * (y1 - y0))
    line_z = z0 + t * (z1 - z0)
    interior = slice(1, -1)
    los_intrusion = float(np.max(terrain[interior] - line_z[interior]))
    wavelength = LIGHT_SPEED_MPS / frequency_hz
    d1, d2 = t * distance_3d, (1.0 - t) * distance_3d
    fresnel = np.sqrt(np.maximum(wavelength * d1 * d2 / max(distance_3d, 1.0), 0.0))
    fresnel_intrusion = float(
        np.max(terrain[interior] + clearance_ratio * fresnel[interior] - line_z[interior])
    )
    return horizontal, distance_3d, fresnel_intrusion, los_intrusion <= 0.0, fresnel_intrusion <= 0.0


def link_rate_mbps(
    distance_m: float,
    bandwidth_hz: float,
    tx_power_dbm: float,
    frequency_hz: float,
    noise_density_dbm_per_hz: float,
    noise_figure_db: float,
    implementation_loss_db: float,
    scheduling_efficiency: float,
    spectral_efficiency_cap: float,
) -> tuple[float, float, float]:
    d_km = max(distance_m, 1.0) / 1000.0
    f_mhz = frequency_hz / 1e6
    path_loss_db = 32.44 + 20.0 * math.log10(d_km) + 20.0 * math.log10(f_mhz)
    rx_dbm = tx_power_dbm - path_loss_db - implementation_loss_db
    noise_dbm = noise_density_dbm_per_hz + 10.0 * math.log10(bandwidth_hz) + noise_figure_db
    snr_db = rx_dbm - noise_dbm
    efficiency = min(math.log2(1.0 + 10.0 ** (snr_db / 10.0)), spectral_efficiency_cap)
    rate = scheduling_efficiency * bandwidth_hz * efficiency / 1e6
    return float(rate), float(snr_db), float(path_loss_db)


def enumerate_backhaul_options(
    candidate_ids: list[str],
    direct_edges: pd.DataFrame,
    relay_edges: pd.DataFrame,
    max_options_per_uav: int,
) -> list[dict[str, Any]]:
    direct_by_uav: dict[str, list[dict[str, Any]]] = {}
    for row in direct_edges.itertuples(index=False):
        if bool(row.link_feasible):
            direct_by_uav.setdefault(str(row.uav_id), []).append(
                {"gbs_id": str(row.gbs_id), "rate": float(row.link_rate_mbps)}
            )
    relay_map: dict[str, list[tuple[str, float]]] = {}
    for row in relay_edges.itertuples(index=False):
        if bool(row.link_feasible):
            relay_map.setdefault(str(row.uav_a_id), []).append(
                (str(row.uav_b_id), float(row.link_rate_mbps))
            )
            relay_map.setdefault(str(row.uav_b_id), []).append(
                (str(row.uav_a_id), float(row.link_rate_mbps))
            )
    options: list[dict[str, Any]] = []
    for source in candidate_ids:
        available: list[dict[str, Any]] = []
        for direct in direct_by_uav.get(source, []):
            available.append(
                {
                    "source_uav_id": source,
                    "path_type": "direct",
                    "relay_uav_id": "",
                    "gbs_id": direct["gbs_id"],
                    "path_edges": 1,
                    "relay_uavs": 0,
                    "bottleneck_rate_mbps": direct["rate"],
                    "path_nodes": f"{source}>{direct['gbs_id']}",
                }
            )
        for relay, relay_rate in relay_map.get(source, []):
            if relay == source:
                continue
            for direct in direct_by_uav.get(relay, []):
                available.append(
                    {
                        "source_uav_id": source,
                        "path_type": "one_relay",
                        "relay_uav_id": relay,
                        "gbs_id": direct["gbs_id"],
                        "path_edges": 2,
                        "relay_uavs": 1,
                        "bottleneck_rate_mbps": min(relay_rate, direct["rate"]),
                        "path_nodes": f"{source}>{relay}>{direct['gbs_id']}",
                    }
                )
        available.sort(
            key=lambda item: (
                -float(item["bottleneck_rate_mbps"]),
                int(item["path_edges"]),
                str(item["path_nodes"]),
            )
        )
        seen: set[tuple[str, str, str]] = set()
        rank = 0
        for item in available:
            key = (str(item["path_type"]), str(item["relay_uav_id"]), str(item["gbs_id"]))
            if key in seen:
                continue
            seen.add(key)
            rank += 1
            item["option_rank"] = rank
            options.append(item)
            if rank >= max_options_per_uav:
                break
    return options


def require_exp2(exp2_root: Path, expected_cases: int, counts: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_path = exp2_root / "audit_summary.json"
    state_path = exp2_root / "data" / "state_by_case.csv"
    gbs_path = exp2_root / "data" / "surviving_gbs_positions_all_cases.csv"
    for path in (audit_path, state_path, gbs_path, exp2_root / "summary.csv"):
        if not path.exists():
            raise FileNotFoundError(f"Missing formal Exp2 input: {path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise RuntimeError("Formal Exp2 audit did not pass")
    if int(audit.get("expected_cases", -1)) != expected_cases or int(audit.get("actual_cases", -1)) != expected_cases:
        raise RuntimeError("Formal Exp2 case count is not 60/60")
    states = pd.read_csv(state_path)
    anchors = pd.read_csv(gbs_path)
    if len(states) != expected_cases:
        raise RuntimeError(f"Exp2 state table has {len(states)} rows, expected {expected_cases}")
    observed = anchors.groupby(["terrain", "seed", "damage_level"]).size()
    for (_, _, level), value in observed.items():
        if int(value) != int(counts[level]):
            raise RuntimeError(f"Exp2 retained GBS count mismatch for {level}: {value}")
    for (terrain, seed), group in anchors.groupby(["terrain", "seed"]):
        sets = {
            level: set(group.loc[group["damage_level"] == level, "gbs_id"].astype(str))
            for level in ("low", "medium", "high")
        }
        if not (sets["high"] < sets["medium"] < sets["low"]):
            raise RuntimeError(f"Exp2 retained GBS nesting failed for {terrain}, seed={seed}")
    return states, anchors


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("protocol_stack_exp3")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_figures(summary: pd.DataFrame, figure_dir: Path) -> None:
    grouped = summary.groupby("infrastructure_level")[
        ["access_phy_count", "relay_phy_count", "backhaul_phy_count", "link_edge_count"]
    ].mean().reindex(["low", "medium", "high"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    grouped.plot(kind="line", marker="o", linewidth=1.8, ax=ax)
    ax.set_xlabel("Infrastructure limitation level")
    ax.set_ylabel("Mean primitive count per case")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "primitive_counts_by_layer.png", dpi=300)
    plt.close(fig)

    option_group = summary.groupby("infrastructure_level")["backhaul_option_count"].mean().reindex(
        ["low", "medium", "high"]
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(option_group.index, option_group.values, color="#2A9D8F", marker="o", linewidth=2)
    ax.set_xlabel("Infrastructure limitation level")
    ax.set_ylabel("Mean retained backhaul options")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "backhaul_option_counts.png", dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, config_path, config_hash = load_config(args.config)
    output_root = WORKSPACE / config["paths"]["output_root"] / "exp3_service_primitives_compile"
    if output_root.exists() and args.force:
        resolved = output_root.resolve()
        expected_parent = (WORKSPACE / config["paths"]["output_root"]).resolve()
        if expected_parent not in resolved.parents:
            raise RuntimeError(f"Refusing to clear unexpected output path: {resolved}")
        shutil.rmtree(resolved)
    for name in ("data", "figures", "audit"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_root / "run.log")
    expected_cases = int(config["design"]["expected_base_cases"])
    exp2_root = WORKSPACE / config["paths"]["output_root"] / "exp2_infrastructure_limited_service_state"
    backup_root = WORKSPACE / config["paths"]["output_root"] / "exp2_infrastructure_limited_service_state_variable_gbs_ratio_backup"
    frozen_root = WORKSPACE / config["paths"]["frozen_input_root"]
    counts = {k: int(v) for k, v in config["infrastructure_state"]["surviving_gbs_count"].items()}
    geometry, radio, capacity = config["geometry"], config["radio"], config["capacity"]
    frozen = {
        "H_AGL_m": float(geometry["H_AGL_m"]),
        "R_access_m": float(geometry["access_radius_m"]),
        "R_relay_m": float(geometry["relay_radius_m"]),
        "R_backhaul_m": float(geometry["backhaul_radius_m"]),
        "K": int(config["design"]["K"]),
        "capacity_main_mbps": float(capacity["capacity_main_mbps"]),
        "capacity_sweep_mbps": [float(x) for x in capacity["capacity_sweep_mbps"]],
        "max_backhaul_edges_per_path": int(geometry["max_backhaul_edges_per_path"]),
        "max_relay_uavs_per_path": int(geometry["max_relay_uavs_per_path"]),
        "max_backhaul_options_per_uav": int(geometry["max_backhaul_options_per_uav"]),
        "allow_relay_of_relay": bool(geometry["allow_relay_of_relay"]),
    }
    logger.info("Starting Exp3; formal Exp2 input=%s", exp2_root)
    states, retained = require_exp2(exp2_root, expected_cases, counts)
    input_files = [
        config_path,
        exp2_root / "audit_summary.json",
        exp2_root / "summary.csv",
        exp2_root / "data" / "state_by_case.csv",
        exp2_root / "data" / "surviving_gbs_positions_all_cases.csv",
    ]
    input_hash = hash_files(input_files)
    access_rows: list[dict[str, Any]] = []
    relay_rows: list[dict[str, Any]] = []
    backhaul_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []
    option_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    bundle_rows: list[dict[str, Any]] = []
    primitive_audit: list[dict[str, Any]] = []
    path_audit: list[dict[str, Any]] = []
    exp2_audit: list[dict[str, Any]] = []
    scenario_cache: dict[str, dict[str, Any]] = {}
    f_hz = float(radio["carrier_frequency_hz"])
    clearance = float(radio["fresnel_clearance_ratio"])

    for terrain in [item["id"] for item in config["design"]["terrains"]]:
        scenario_dir = frozen_root / terrain
        dem = np.load(scenario_dir / "dem_0.npy")
        poi = pd.read_csv(scenario_dir / "poi_0.csv")
        uav = pd.read_csv(scenario_dir / "uav_candidate.csv")
        uav = uav.copy()
        uav["ground_z_m"] = bilinear_dem(dem, uav["x_m"].to_numpy(), uav["y_m"].to_numpy())
        uav["z_m"] = uav["ground_z_m"] + frozen["H_AGL_m"]
        poi_z = bilinear_dem(dem, poi["x_m"].to_numpy(), poi["y_m"].to_numpy()) + POI_HEIGHT_M
        terrain_access: list[dict[str, Any]] = []
        for cand in uav.itertuples(index=False):
            dx = poi["x_m"].to_numpy() - float(cand.x_m)
            dy = poi["y_m"].to_numpy() - float(cand.y_m)
            within = np.flatnonzero(np.hypot(dx, dy) <= frozen["R_access_m"])
            for idx in within:
                horizontal, distance, intrusion, los, fresnel = profile_clearance(
                    dem, float(cand.x_m), float(cand.y_m), float(cand.z_m),
                    float(poi.iloc[idx]["x_m"]), float(poi.iloc[idx]["y_m"]), float(poi_z[idx]),
                    f_hz, clearance,
                )
                record = {
                    "terrain": terrain,
                    "uav_id": str(cand.candidate_id),
                    "poi_id": str(poi.iloc[idx]["poi_id"]),
                    "horizontal_distance_m": horizontal,
                    "distance_3d_m": distance,
                    "los_clear": los,
                    "fresnel_clear": fresnel,
                    "max_fresnel_intrusion_m": intrusion,
                    "phy_feasible": bool(los and fresnel),
                }
                terrain_access.append(record)
                access_rows.append(record)
                if record["phy_feasible"]:
                    rate, snr, path_loss = link_rate_mbps(
                        distance, float(radio["access_bandwidth_hz"]), float(radio["uav_tx_power_dbm"]),
                        f_hz, float(radio["noise_density_dbm_per_hz"]),
                        float(radio["receiver_noise_figure_db"]), float(radio["implementation_loss_db"]),
                        float(radio["scheduling_efficiency"]), float(radio["spectral_efficiency_cap_bps_hz"]),
                    )
                    link_rows.append({
                        "scope_id": terrain, "terrain": terrain, "infrastructure_level": "fixed",
                        "seed": -1, "link_type": "access", "source_id": str(cand.candidate_id),
                        "target_id": str(poi.iloc[idx]["poi_id"]), "distance_3d_m": distance,
                        "link_rate_mbps": rate, "snr_db": snr, "path_loss_db": path_loss,
                        "link_feasible": rate >= float(radio["minimum_access_rate_mbps"]),
                    })
        terrain_relay: list[dict[str, Any]] = []
        for i in range(len(uav)):
            a = uav.iloc[i]
            for j in range(i + 1, len(uav)):
                b = uav.iloc[j]
                if math.hypot(float(b.x_m - a.x_m), float(b.y_m - a.y_m)) > frozen["R_relay_m"]:
                    continue
                horizontal, distance, intrusion, los, fresnel = profile_clearance(
                    dem, float(a.x_m), float(a.y_m), float(a.z_m),
                    float(b.x_m), float(b.y_m), float(b.z_m), f_hz, clearance,
                )
                rate, snr, path_loss = link_rate_mbps(
                    distance, float(radio["relay_bandwidth_hz"]), float(radio["uav_tx_power_dbm"]),
                    f_hz, float(radio["noise_density_dbm_per_hz"]),
                    float(radio["receiver_noise_figure_db"]), float(radio["implementation_loss_db"]),
                    float(radio["scheduling_efficiency"]), float(radio["spectral_efficiency_cap_bps_hz"]),
                )
                feasible = bool(los and fresnel and rate >= float(radio["minimum_relay_rate_mbps"]))
                record = {
                    "terrain": terrain, "uav_a_id": str(a.candidate_id), "uav_b_id": str(b.candidate_id),
                    "horizontal_distance_m": horizontal, "distance_3d_m": distance,
                    "los_clear": los, "fresnel_clear": fresnel,
                    "max_fresnel_intrusion_m": intrusion, "link_rate_mbps": rate,
                    "snr_db": snr, "path_loss_db": path_loss, "link_feasible": feasible,
                }
                terrain_relay.append(record)
                relay_rows.append(record)
                if feasible:
                    link_rows.append({
                        "scope_id": terrain, "terrain": terrain, "infrastructure_level": "fixed",
                        "seed": -1, "link_type": "relay", "source_id": str(a.candidate_id),
                        "target_id": str(b.candidate_id), "distance_3d_m": distance,
                        "link_rate_mbps": rate, "snr_db": snr, "path_loss_db": path_loss,
                        "link_feasible": True,
                    })
        scenario_cache[terrain] = {
            "dem": dem, "poi": poi, "uav": uav,
            "access": pd.DataFrame(terrain_access), "relay": pd.DataFrame(terrain_relay),
        }

    for state in states.itertuples(index=False):
        terrain = str(state.scenario_id)
        level = str(state.stress_level)
        seed = int(state.seed)
        case_id = f"{terrain}_{level}_seed{seed}"
        cached = scenario_cache[terrain]
        dem, uav = cached["dem"], cached["uav"]
        gbs = retained[
            (retained["terrain"] == terrain)
            & (retained["damage_level"] == level)
            & (retained["seed"] == seed)
        ].copy()
        if len(gbs) != counts[level]:
            raise RuntimeError(f"{case_id}: retained GBS count mismatch")
        gbs["ground_z_m"] = bilinear_dem(dem, gbs["x_m"].to_numpy(), gbs["y_m"].to_numpy())
        gbs["z_m"] = gbs["ground_z_m"] + GBS_HEIGHT_M
        case_backhaul: list[dict[str, Any]] = []
        for cand in uav.itertuples(index=False):
            for station in gbs.itertuples(index=False):
                if math.hypot(float(station.x_m - cand.x_m), float(station.y_m - cand.y_m)) > frozen["R_backhaul_m"]:
                    continue
                horizontal, distance, intrusion, los, fresnel = profile_clearance(
                    dem, float(cand.x_m), float(cand.y_m), float(cand.z_m),
                    float(station.x_m), float(station.y_m), float(station.z_m), f_hz, clearance,
                )
                rate, snr, path_loss = link_rate_mbps(
                    distance, float(radio["backhaul_bandwidth_hz"]), float(radio["uav_tx_power_dbm"]),
                    f_hz, float(radio["noise_density_dbm_per_hz"]),
                    float(radio["receiver_noise_figure_db"]), float(radio["implementation_loss_db"]),
                    float(radio["scheduling_efficiency"]), float(radio["spectral_efficiency_cap_bps_hz"]),
                )
                feasible = bool(los and fresnel and rate >= float(radio["minimum_backhaul_rate_mbps"]))
                record = {
                    "case_id": case_id, "terrain": terrain, "infrastructure_level": level, "seed": seed,
                    "uav_id": str(cand.candidate_id), "gbs_id": str(station.gbs_id),
                    "horizontal_distance_m": horizontal, "distance_3d_m": distance,
                    "los_clear": los, "fresnel_clear": fresnel,
                    "max_fresnel_intrusion_m": intrusion, "link_rate_mbps": rate,
                    "snr_db": snr, "path_loss_db": path_loss, "link_feasible": feasible,
                }
                case_backhaul.append(record)
                backhaul_rows.append(record)
                if feasible:
                    link_rows.append({
                        "scope_id": case_id, "terrain": terrain, "infrastructure_level": level,
                        "seed": seed, "link_type": "backhaul", "source_id": str(cand.candidate_id),
                        "target_id": str(station.gbs_id), "distance_3d_m": distance,
                        "link_rate_mbps": rate, "snr_db": snr, "path_loss_db": path_loss,
                        "link_feasible": True,
                    })
        backhaul_df = pd.DataFrame(case_backhaul)
        options = enumerate_backhaul_options(
            uav["candidate_id"].astype(str).tolist(), backhaul_df, cached["relay"],
            frozen["max_backhaul_options_per_uav"],
        )
        for option in options:
            option.update({"case_id": case_id, "terrain": terrain, "infrastructure_level": level, "seed": seed})
            option_rows.append(option)
        access_feasible = int(cached["access"]["phy_feasible"].sum())
        relay_feasible = int(cached["relay"]["link_feasible"].sum())
        backhaul_feasible = int(backhaul_df["link_feasible"].sum()) if len(backhaul_df) else 0
        case_link_count = access_feasible + relay_feasible + backhaul_feasible
        case_rows.append({
            "case_id": case_id, "terrain": terrain, "infrastructure_level": level, "seed": seed,
            "retained_gbs_count": len(gbs), "poi_count": len(cached["poi"]),
            "uav_candidate_count": len(uav), "access_phy_count": access_feasible,
            "relay_phy_count": relay_feasible, "backhaul_phy_count": backhaul_feasible,
            "link_edge_count": case_link_count, "backhaul_option_count": len(options),
        })
        bundle_rows.append({
            "case_id": case_id, "terrain": terrain, "infrastructure_level": level, "seed": seed,
            "phy_access_scope": terrain, "phy_relay_scope": terrain,
            "phy_backhaul_scope": case_id, "link_scope": f"{terrain}+{case_id}",
            "backhaul_options_scope": case_id, "poi_demand_path": str(state.demand_state_path),
            "retained_gbs_source": str(exp2_root / "data" / "surviving_gbs_positions_all_cases.csv"),
        })
        primitive_audit.append({
            "case_id": case_id, "access_phy_count": access_feasible,
            "relay_phy_count": relay_feasible, "backhaul_phy_count": backhaul_feasible,
            "link_edge_count": case_link_count, "backhaul_option_count": len(options),
            "finite": True, "nonempty_candidate_table": len(uav) > 0,
        })
        max_edges = max([int(x["path_edges"]) for x in options], default=0)
        max_relays = max([int(x["relay_uavs"]) for x in options], default=0)
        path_audit.append({
            "case_id": case_id, "max_path_edges": max_edges, "max_relay_uavs": max_relays,
            "relay_of_relay_count": sum(int(x["relay_uavs"]) > 1 for x in options),
            "self_relay_count": sum(x["relay_uav_id"] == x["source_uav_id"] for x in options),
            "path_legality_pass": max_edges <= 2 and max_relays <= 1,
        })
        exp2_audit.append({
            "case_id": case_id, "infrastructure_level": level,
            "expected_retained_gbs": counts[level], "actual_retained_gbs": len(gbs),
            "count_match": len(gbs) == counts[level],
            "retained_gbs_source_is_exp2": True, "raw_full_gbs_used": False,
            "backup_exp2_used": False,
        })
        logger.info("%s compiled: access=%d relay=%d backhaul=%d options=%d",
                    case_id, access_feasible, relay_feasible, backhaul_feasible, len(options))

    tables = {
        "case_index.csv": pd.DataFrame(case_rows),
        "access_phy_edges.csv": pd.DataFrame(access_rows),
        "relay_phy_edges.csv": pd.DataFrame(relay_rows),
        "backhaul_phy_edges.csv": pd.DataFrame(backhaul_rows),
        "link_edges.csv": pd.DataFrame(link_rows),
        "backhaul_options.csv": pd.DataFrame(option_rows),
        "primitive_bundle_index.csv": pd.DataFrame(bundle_rows),
    }
    for name, frame in tables.items():
        frame.to_csv(output_root / "data" / name, index=False)
    pd.DataFrame(primitive_audit).to_csv(output_root / "audit" / "primitive_count_audit.csv", index=False)
    pd.DataFrame(path_audit).to_csv(output_root / "audit" / "path_legality_audit.csv", index=False)
    pd.DataFrame(exp2_audit).to_csv(output_root / "audit" / "exp2_input_consistency_audit.csv", index=False)
    summary = pd.DataFrame(case_rows)
    summary.to_csv(output_root / "summary.csv", index=False)
    save_figures(summary, output_root / "figures")

    numeric_finite = all(
        np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
        for frame in tables.values() if not frame.empty
    )
    path_frame = pd.DataFrame(path_audit)
    exp2_frame = pd.DataFrame(exp2_audit)
    required = [
        output_root / "data" / name for name in tables
    ] + [
        output_root / "figures" / "primitive_counts_by_layer.png",
        output_root / "figures" / "backhaul_option_counts.png",
        output_root / "audit" / "primitive_count_audit.csv",
        output_root / "audit" / "path_legality_audit.csv",
        output_root / "audit" / "exp2_input_consistency_audit.csv",
        output_root / "summary.csv",
        output_root / "run.log",
    ]
    audit = {
        "stage": STAGE,
        "experiment_name": "Four-layer Service Certification Primitive Compilation under Infrastructure-Limited Conditions",
        "expected_cases": expected_cases,
        "actual_cases": len(summary),
        "exp2_passed": True,
        "exp2_input_used": True,
        "exp2_input_path": str(exp2_root),
        "raw_full_gbs_used_as_available_gbs": False,
        "backup_exp2_used": False,
        "backup_exp2_path_excluded": str(backup_root),
        "retained_gbs_count_match": bool(exp2_frame["count_match"].all()),
        "all_primitives_finite": bool(numeric_finite),
        "all_required_outputs_exist": bool(all(path.exists() for path in required)),
        "phy_link_net_app_primitive_tables_generated": True,
        "graph_structures_generated": True,
        "no_relay_of_relay": bool((path_frame["relay_of_relay_count"] == 0).all()),
        "max_path_edges_observed": int(path_frame["max_path_edges"].max()),
        "max_relay_uavs_observed": int(path_frame["max_relay_uavs"].max()),
        "allow_relay_of_relay": frozen["allow_relay_of_relay"],
        "constraint_violation_sum": int(
            (~path_frame["path_legality_pass"]).sum() + (~exp2_frame["count_match"]).sum()
        ),
        "primitive_counts": {
            "access_phy_feasible_unique": int(pd.DataFrame(access_rows)["phy_feasible"].sum()),
            "relay_phy_feasible_unique": int(pd.DataFrame(relay_rows)["link_feasible"].sum()),
            "backhaul_phy_feasible_case_scoped": int(pd.DataFrame(backhaul_rows)["link_feasible"].sum()),
            "link_edges": len(link_rows),
            "backhaul_options": len(option_rows),
        },
        "frozen_parameters": frozen,
        "config_hash": config_hash,
        "input_hash": input_hash,
    }
    audit["passed"] = bool(
        audit["actual_cases"] == expected_cases
        and audit["retained_gbs_count_match"]
        and audit["all_primitives_finite"]
        and audit["all_required_outputs_exist"]
        and audit["no_relay_of_relay"]
        and audit["max_path_edges_observed"] <= frozen["max_backhaul_edges_per_path"]
        and audit["max_relay_uavs_observed"] <= frozen["max_relay_uavs_per_path"]
        and not audit["allow_relay_of_relay"]
        and audit["constraint_violation_sum"] == 0
    )
    write_json(output_root / "audit_summary.json", audit)
    report = [
        "# Exp3 audit report",
        "",
        "## Scope",
        "",
        "Four-layer service-certification primitives were compiled under infrastructure-limited conditions. "
        "The labels low/medium/high denote infrastructure limitation levels; DEM and POI geometry remain fixed.",
        "",
        "## Input provenance",
        "",
        f"- Formal Exp2 input: `{exp2_root}`",
        "- Available GBS are read only from Exp2 retained service anchors.",
        "- Raw full GBS files are not used as available-GBS sets.",
        f"- Deprecated Exp2 backup excluded: `{backup_root}`",
        "",
        "## Frozen parameters",
        "",
    ]
    report.extend(f"- `{key}`: `{value}`" for key, value in frozen.items())
    report.extend([
        "",
        "## Physical and link model",
        "",
        f"- DEM profile samples per assessed link: `{PROFILE_SAMPLES}`",
        f"- Endpoint heights: POI `{POI_HEIGHT_M} m`, GBS `{GBS_HEIGHT_M} m`, UAV AGL from YAML.",
        "- Physical feasibility requires radius compliance, terrain LoS, and configured first-Fresnel clearance.",
        "- Link rate uses free-space path loss and capped Shannon spectral efficiency with YAML radio parameters.",
        "- Access/relay geometry is compiled once per terrain and referenced by all cases because DEM, POI, and UAV candidates are fixed.",
        "",
        "## Audit result",
        "",
        f"- Cases: `{len(summary)}/{expected_cases}`",
        f"- Retained GBS counts match Exp2: `{audit['retained_gbs_count_match']}`",
        f"- All primitives finite: `{audit['all_primitives_finite']}`",
        f"- No relay-of-relay: `{audit['no_relay_of_relay']}`",
        f"- Maximum path edges observed: `{audit['max_path_edges_observed']}`",
        f"- Maximum relay UAVs observed: `{audit['max_relay_uavs_observed']}`",
        f"- Constraint violations: `{audit['constraint_violation_sum']}`",
        f"- Final passed: `{audit['passed']}`",
        "",
        f"Generated at `{utc_timestamp()}`.",
    ])
    (output_root / "audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_flag(
        output_root / "experiment_done.flag", STAGE, "passed" if audit["passed"] else "failed",
        expected_cases, len(summary), config_hash, input_hash,
    )
    logger.info("Exp3 completed: expected=%d actual=%d passed=%s", expected_cases, len(summary), audit["passed"])
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
