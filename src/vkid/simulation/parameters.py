"""Parameter containers and sampling for vehicle dynamics simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class TireParams:
    """Magic Formula parameters for a shared front/rear tire model."""

    b: float
    c: float
    d: float
    e: float


@dataclass(frozen=True)
class VehicleParams:
    """Vehicle parameters for the 3-DOF bicycle model."""

    mass_kg: float
    iz_kgm2: float
    lf_m: float
    lr_m: float

    @property
    def wheelbase_m(self) -> float:
        return self.lf_m + self.lr_m


@dataclass(frozen=True)
class DynamicsCondition:
    """One sampled vehicle+tire dynamics condition."""

    condition_id: int
    tire: TireParams
    vehicle: VehicleParams

    def as_array(self) -> np.ndarray:
        lf_ratio = self.vehicle.lf_m / self.vehicle.wheelbase_m
        return np.array(
            [
                self.tire.b,
                self.tire.c,
                self.tire.d,
                self.tire.e,
                self.vehicle.mass_kg,
                self.vehicle.iz_kgm2,
                lf_ratio,
            ],
            dtype=np.float64,
        )


PARAMETER_NAMES = np.array(["B", "C", "D", "E", "m", "Iz", "lf_ratio"])


def _scale_unit_samples(samples: np.ndarray, ranges: list[tuple[float, float]]) -> np.ndarray:
    lower = np.array([item[0] for item in ranges], dtype=np.float64)
    upper = np.array([item[1] for item in ranges], dtype=np.float64)
    return qmc.scale(samples, lower, upper)


def sample_conditions(config: dict, seed: int) -> list[DynamicsCondition]:
    """Sample dynamics conditions with Latin Hypercube Sampling."""

    n_conditions = int(config["dataset"]["conditions"])
    wheelbase_m = float(config["vehicle"]["wheelbase_m"])
    ranges = [
        tuple(config["tire"]["b_range"]),
        tuple(config["tire"]["c_range"]),
        tuple(config["tire"]["d_range"]),
        tuple(config["tire"]["e_range"]),
        tuple(config["vehicle"]["mass_kg_range"]),
        tuple(config["vehicle"]["iz_kgm2_range"]),
        tuple(config["vehicle"]["lf_ratio_range"]),
    ]

    sampler = qmc.LatinHypercube(d=7, seed=seed)
    samples = _scale_unit_samples(sampler.random(n_conditions), ranges)

    conditions: list[DynamicsCondition] = []
    for condition_id, row in enumerate(samples):
        b, c, d, e, mass_kg, iz_kgm2, lf_ratio = row
        lf_m = wheelbase_m * lf_ratio
        lr_m = wheelbase_m - lf_m
        conditions.append(
            DynamicsCondition(
                condition_id=condition_id,
                tire=TireParams(b=b, c=c, d=d, e=e),
                vehicle=VehicleParams(
                    mass_kg=mass_kg,
                    iz_kgm2=iz_kgm2,
                    lf_m=lf_m,
                    lr_m=lr_m,
                ),
            )
        )
    return conditions


def conditions_to_array(conditions: list[DynamicsCondition]) -> np.ndarray:
    """Convert sampled conditions to a dense metadata matrix."""

    return np.stack([condition.as_array() for condition in conditions], axis=0)
