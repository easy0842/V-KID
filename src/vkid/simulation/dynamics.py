"""3-DOF bicycle model and Magic Formula tire dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from vkid.simulation.parameters import DynamicsCondition

GRAVITY = 9.81


@dataclass(frozen=True)
class TireDiagnostics:
    alpha_f_rad: np.ndarray
    alpha_r_rad: np.ndarray
    fyf_n: np.ndarray
    fyr_n: np.ndarray


@dataclass(frozen=True)
class SimulationResult:
    time_s: np.ndarray
    state: np.ndarray
    action: np.ndarray
    pose: np.ndarray
    diagnostics: TireDiagnostics


def magic_formula_lateral_force(alpha_rad: np.ndarray | float, fz_n: float, condition: DynamicsCondition) -> np.ndarray:
    """Compute lateral tire force with a load-scaled Magic Formula."""

    tire = condition.tire
    alpha = np.clip(alpha_rad, -0.8, 0.8)
    shape = np.sin(tire.c * np.arctan(tire.b * alpha - tire.e * (tire.b * alpha - np.arctan(tire.b * alpha))))
    return tire.d * fz_n * shape


def tire_diagnostics(
    vx_mps: np.ndarray,
    vy_mps: np.ndarray,
    wz_radps: np.ndarray,
    delta_rad: np.ndarray,
    condition: DynamicsCondition,
) -> TireDiagnostics:
    vehicle = condition.vehicle
    vx_safe = np.maximum(vx_mps, 1.0)
    alpha_f = delta_rad - np.arctan2(vy_mps + vehicle.lf_m * wz_radps, vx_safe)
    alpha_r = -np.arctan2(vy_mps - vehicle.lr_m * wz_radps, vx_safe)

    fzf = vehicle.mass_kg * GRAVITY * vehicle.lr_m / vehicle.wheelbase_m
    fzr = vehicle.mass_kg * GRAVITY * vehicle.lf_m / vehicle.wheelbase_m
    fyf = magic_formula_lateral_force(alpha_f, fzf, condition)
    fyr = magic_formula_lateral_force(alpha_r, fzr, condition)
    return TireDiagnostics(alpha_f, alpha_r, fyf, fyr)


def _interp_action(time_s: float, sample_times_s: np.ndarray, actions: np.ndarray) -> tuple[float, float]:
    delta = float(np.interp(time_s, sample_times_s, actions[:, 0]))
    fx = float(np.interp(time_s, sample_times_s, actions[:, 1]))
    return delta, fx


def _derivative(
    time_s: float,
    ode_state: np.ndarray,
    sample_times_s: np.ndarray,
    actions: np.ndarray,
    condition: DynamicsCondition,
) -> np.ndarray:
    vx, vy, wz, pos_x, pos_y, yaw = ode_state
    del pos_x, pos_y

    vehicle = condition.vehicle
    delta, fx_rear = _interp_action(time_s, sample_times_s, actions)
    vx_safe = max(float(vx), 1.0)

    diag = tire_diagnostics(
        vx_mps=np.array([vx_safe]),
        vy_mps=np.array([vy]),
        wz_radps=np.array([wz]),
        delta_rad=np.array([delta]),
        condition=condition,
    )
    fyf = float(diag.fyf_n[0])
    fyr = float(diag.fyr_n[0])

    fxf = 0.0
    fxr = fx_rear
    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)

    dvx = (fxf * cos_delta - fyf * sin_delta + fxr) / vehicle.mass_kg + vy * wz
    dvy = (fyf * cos_delta + fxf * sin_delta + fyr) / vehicle.mass_kg - vx * wz
    dwz = (vehicle.lf_m * (fyf * cos_delta + fxf * sin_delta) - vehicle.lr_m * fyr) / vehicle.iz_kgm2

    # Prevent numerical excursions into reverse driving for this first simulator.
    if vx < 0.5 and dvx < 0.0:
        dvx = 0.0

    dpos_x = vx * np.cos(yaw) - vy * np.sin(yaw)
    dpos_y = vx * np.sin(yaw) + vy * np.cos(yaw)
    dyaw = wz
    return np.array([dvx, dvy, dwz, dpos_x, dpos_y, dyaw], dtype=np.float64)


def simulate_sequence(
    condition: DynamicsCondition,
    actions: np.ndarray,
    sample_rate_hz: float,
    vx0_mps: float,
) -> SimulationResult:
    """Roll out one sequence for a sampled dynamics condition."""

    n_steps = actions.shape[0]
    time_s = np.arange(n_steps, dtype=np.float64) / sample_rate_hz
    initial_state = np.array([vx0_mps, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    solution = solve_ivp(
        fun=lambda t, y: _derivative(t, y, time_s, actions, condition),
        t_span=(float(time_s[0]), float(time_s[-1])),
        y0=initial_state,
        t_eval=time_s,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
        max_step=1.0 / sample_rate_hz,
    )
    if not solution.success:
        raise RuntimeError(f"Simulation failed for condition {condition.condition_id}: {solution.message}")

    state = solution.y[:3].T.astype(np.float64)
    pose = solution.y[3:6].T.astype(np.float64)
    diag = tire_diagnostics(
        vx_mps=state[:, 0],
        vy_mps=state[:, 1],
        wz_radps=state[:, 2],
        delta_rad=actions[:, 0],
        condition=condition,
    )
    return SimulationResult(time_s=time_s, state=state, action=actions, pose=pose, diagnostics=diag)
