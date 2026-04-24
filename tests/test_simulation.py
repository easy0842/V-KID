import numpy as np

from vkid.simulation.actions import generate_actions
from vkid.simulation.dynamics import magic_formula_lateral_force, simulate_sequence
from vkid.simulation.parameters import DynamicsCondition, TireParams, VehicleParams


def _condition() -> DynamicsCondition:
    return DynamicsCondition(
        condition_id=0,
        tire=TireParams(b=10.0, c=1.3, d=0.9, e=-0.5),
        vehicle=VehicleParams(mass_kg=1500.0, iz_kgm2=2400.0, lf_m=1.25, lr_m=1.45),
    )


def test_magic_formula_is_odd_around_zero() -> None:
    condition = _condition()
    alpha = np.array([-0.05, 0.0, 0.05])
    force = magic_formula_lateral_force(alpha, fz_n=4000.0, condition=condition)
    np.testing.assert_allclose(force[0], -force[2], rtol=1e-6, atol=1e-6)
    assert force[1] == 0.0


def test_simulate_sequence_shapes_and_finite_values() -> None:
    config = {
        "action": {
            "steer_cutoff_hz": 0.8,
            "steer_amplitude_deg_range": [3.0, 4.0],
            "fx_cutoff_hz": 0.4,
            "fx_amplitude_n_range": [1000.0, 1200.0],
            "event_probability": 0.0,
        }
    }
    rng = np.random.default_rng(123)
    actions = generate_actions(config, rng, n_steps=30, sample_rate_hz=20.0)
    result = simulate_sequence(_condition(), actions, sample_rate_hz=20.0, vx0_mps=20.0)

    assert result.state.shape == (30, 3)
    assert result.action.shape == (30, 2)
    assert result.pose.shape == (30, 3)
    assert np.isfinite(result.state).all()
    assert np.min(result.state[:, 0]) > 0.0
