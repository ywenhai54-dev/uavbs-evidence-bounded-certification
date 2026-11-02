"""Strict-budget certification policies for the CT-KS mechanism comparison.

This module is experiment-only.  It preserves the frozen production outputs and
uses the same SciPy/HiGHS model and checker as the existing CT-KS implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

from ctks_service_chain_kernel import (
    PolicyResult,
    PreparedKernelModel,
    prepare_kernel_model,
    run_policy,
)


TOTAL_BUDGET_SECONDS = 5.0
CTKS_STAGE_BUDGET_SECONDS = 1.0
RELAXED_MIP_GAP = 1.0e-3

METHOD_FULL = "full_domain"
METHOD_THRESHOLD = "threshold_only"
METHOD_CTKS = "ctks"
METHOD_RELAXED = "threshold_relaxed_gap_1e-3"
METHODS = (METHOD_FULL, METHOD_THRESHOLD, METHOD_CTKS, METHOD_RELAXED)


@dataclass(frozen=True)
class FairCertification:
    model: PreparedKernelModel
    policy: PolicyResult
    row: dict[str, Any]


def external_decision(lower: float, upper: float, tau: float) -> str:
    if lower >= tau:
        return "accept"
    if upper < tau:
        return "reject"
    return "unresolved"


def _decision_stage(method: str, policy: PolicyResult, decision: str) -> str:
    if method == METHOD_FULL:
        return "full_domain" if policy.full_solves else "unresolved"
    if policy.safely_rejected:
        return "LP"
    phases = [str(item["phase"]) for item in policy.solve_history]
    if method in {METHOD_THRESHOLD, METHOD_RELAXED}:
        return "full_domain" if phases else "unresolved"
    if "full_domain_fallback" in phases:
        return "recovery"
    if "expanded_kernel" in phases:
        return "K1"
    if "initial_kernel" in phases:
        return "K0"
    return "unresolved" if decision == "unresolved" else "LP"


def certify_built_candidate(
    *,
    built: tuple[Any, ...],
    uav_capacity_mbps: float,
    gbs_capacity_mbps: float,
    tau: float,
    method: str,
    build_seconds: float = 0.0,
    total_budget_seconds: float = TOTAL_BUDGET_SECONDS,
    stage_budget_seconds: float = CTKS_STAGE_BUDGET_SECONDS,
) -> FairCertification:
    """Certify one fixed candidate under a strict, shared wall-clock budget.

    ``build_seconds`` is the already measured path/service-chain construction
    time.  It is charged before LP, witness, kernel, and MILP work.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown fair-comparison method: {method}")
    if total_budget_seconds <= 0.0:
        raise ValueError("total_budget_seconds must be positive")
    if build_seconds < 0.0:
        raise ValueError("build_seconds cannot be negative")

    started = perf_counter()
    include_lp_witness = method != METHOD_FULL
    remaining_before_prepare = max(total_budget_seconds - build_seconds, 0.01)
    model = prepare_kernel_model(
        poi_ids=built[0],
        weights=built[1],
        demands=built[2],
        chains_by_poi=built[3],
        edge_capacities=built[4],
        uav_capacity_mbps=uav_capacity_mbps,
        gbs_capacity_mbps=gbs_capacity_mbps,
        lp_time_limit_seconds=remaining_before_prepare,
        include_lp_witness=include_lp_witness,
        wall_clock_budget_seconds=remaining_before_prepare,
    )
    elapsed_before_policy = build_seconds + (perf_counter() - started)
    remaining = max(total_budget_seconds - elapsed_before_policy, 0.0)

    policy_name = "Full-5s" if method == METHOD_FULL else (
        "CT-KS" if method == METHOD_CTKS else "Threshold-only"
    )
    mip_gap = RELAXED_MIP_GAP if method == METHOD_RELAXED else 0.0
    policy = run_policy(
        model,
        policy=policy_name,
        threshold_normalized=float(tau),
        total_milp_budget_seconds=remaining,
        stage_budget_seconds=min(stage_budget_seconds, max(remaining, 0.01)),
        bucket_fraction=0.10,
        mip_relative_gap=mip_gap,
        strict_wall_clock_budget=True,
        threshold_already_tolerance_adjusted=True,
    )

    runtime_total = build_seconds + (perf_counter() - started)
    runtime_lp = float(model.lp_seconds)
    runtime_witness = float(model.witness_seconds)
    runtime_kernel = float(policy.kernel_seconds)
    runtime_milp = float(policy.milp_seconds)
    runtime_prepare = max(
        0.0,
        runtime_total - runtime_lp - runtime_witness - runtime_kernel - runtime_milp,
    )
    policy = replace(
        policy,
        total_accounted_seconds=float(runtime_total),
        preparation_seconds=float(runtime_prepare + runtime_lp + runtime_witness),
    )
    lower = float(policy.lower_bound)
    upper = float(policy.full_domain_upper_bound)
    decision = external_decision(lower, upper, float(tau))
    stage = _decision_stage(method, policy, decision)
    statuses = [str(item["solver_status"]) for item in policy.solve_history]
    row = {
        "method": method,
        "runtime_prepare": runtime_prepare,
        "runtime_lp": runtime_lp,
        "runtime_witness": runtime_witness,
        "runtime_kernel": runtime_kernel,
        "runtime_milp": runtime_milp,
        "runtime_total": runtime_total,
        "runtime_component_sum": (
            runtime_prepare + runtime_lp + runtime_witness + runtime_kernel + runtime_milp
        ),
        "nominal_budget": float(total_budget_seconds),
        "budget_overrun": max(0.0, runtime_total - total_budget_seconds),
        "budget_overrun_seconds": max(0.0, runtime_total - total_budget_seconds),
        "overrun_flag": bool(runtime_total > total_budget_seconds),
        "overrun_ratio": max(0.0, runtime_total - total_budget_seconds)
        / total_budget_seconds,
        "L": lower,
        "U": upper,
        "tau": float(tau),
        "decision": decision,
        "decision_stage": stage,
        "solver_status": ";".join(statuses) if statuses else "not_run",
        "termination_status": policy.status,
        "mip_rel_gap": mip_gap,
        "time_limit": float(total_budget_seconds),
        "full_domain_calls": int(policy.full_solves),
        "kernel_calls": int(policy.restricted_solves),
        "recovery_calls": int(policy.full_domain_fallback),
        "safe_reject": int(decision == "reject"),
        "accepted": int(decision == "accept"),
        "unresolved": int(decision == "unresolved"),
        "kernel_constructed": bool(policy.kernel_seconds > 0.0),
        "checker_pass": bool(policy.exact_checker_pass),
        "solve_history": policy.solve_history,
    }
    return FairCertification(model=model, policy=policy, row=row)
