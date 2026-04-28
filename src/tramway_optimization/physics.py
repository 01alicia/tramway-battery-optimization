"""Physical equations used by the tramway simulation.

This module implements the simplified physical models used to estimate:

- train kinematics: speed and acceleration;
- electrical power demand;
- equivalent electrical network seen by the train;
- resulting train voltage.

The models are intentionally simplified to remain computationally efficient
and suitable for repeated evaluation within optimization loops.
"""

from __future__ import annotations

import numpy as np

from .models import ElectricalNetwork, TrainConfig


def compute_kinematics(time_s: np.ndarray, position_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute train speed and acceleration from position samples.

    The derivatives are approximated using NumPy's gradient function, which
    applies central differences in the interior and first-order differences
    at the boundaries.

    Args:
        time_s (np.ndarray): Time samples, in seconds.
        position_m (np.ndarray): Train position samples, in meters.

    Returns:
        tuple[np.ndarray, np.ndarray]: Speed in meters per second and acceleration
        in meters per second squared.

    Notes:
        This numerical differentiation is sufficient for estimating power demand,
        but it may introduce noise if the position signal is not smooth.
    """
    speed_m_per_s = np.gradient(position_m, time_s)
    acceleration_m_per_s2 = np.gradient(speed_m_per_s, time_s)
    return speed_m_per_s, acceleration_m_per_s2


def compute_train_electrical_power(
    time_s: np.ndarray,
    position_m: np.ndarray,
    train: TrainConfig,
) -> np.ndarray:
    """Estimate the train electrical power demand over time.

    The computation follows three steps: compute speed and acceleration from
    the position profile, estimate traction force from inertia, slope and
    resistive forces, then convert mechanical power to electrical power.

    Args:
        time_s (np.ndarray): Time samples, in seconds.
        position_m (np.ndarray): Train position samples, in meters.
        train (TrainConfig): Train configuration containing mass, efficiency and
            resistance coefficients.

    Returns:
        np.ndarray: Electrical power demand in watts.

    Notes:
        Positive values correspond to traction demand. Negative values correspond
        to regenerative braking power. A constant auxiliary power is added to
        account for onboard systems.

        The resistive force follows a Davis-like model:

        `F = A + Bv + Cv²`
    """
    speed_m_per_s, acceleration_m_per_s2 = compute_kinematics(time_s, position_m)
    speed_kmh = speed_m_per_s * 3.6
    mass_ton = train.mass_kg / 1e3

    resistive_force_n = (
        (train.a0_n + train.a1_n_per_ton * mass_ton)
        + (train.b0_n_per_kmh + train.b1_n_per_ton_per_kmh * mass_ton) * speed_kmh
        + (train.c0_n_per_kmh2 + train.c1_n_per_ton_per_kmh2 * mass_ton) * speed_kmh**2
    )

    traction_force_n = (
        train.mass_kg * acceleration_m_per_s2
        + train.mass_kg * train.gravity_m_per_s2 * np.sin(train.slope_rad)
        + resistive_force_n
    )

    mechanical_power_w = traction_force_n * speed_m_per_s

    electrical_power_w = np.where(
        mechanical_power_w >= 0,
        mechanical_power_w / train.motor_efficiency,
        mechanical_power_w * train.motor_efficiency,
    )
    return electrical_power_w + train.onboard_auxiliary_power_w


def equivalent_resistance(
    position_m: np.ndarray | float,
    line_length_m: float,
    network: ElectricalNetwork,
) -> np.ndarray:
    """Compute the equivalent electrical resistance seen by the train.

    The tramway is supplied by two substations located at both ends of the line.
    Each substation is connected to the train through the overhead line and rails.
    The two branches are combined into a Thevenin equivalent resistance.

    Args:
        position_m (np.ndarray | float): Train position along the line, in meters.
        line_length_m (float): Total length of the line, in meters.
        network (ElectricalNetwork): Electrical network parameters.

    Returns:
        np.ndarray: Equivalent resistance in ohms.

    Notes:
        The equivalent resistance is computed as:

        `R_eq = (R_left * R_right) / (R_left + R_right)`

        The resistance varies with train position because the lengths of the
        left and right supply branches change over time.
    """
    position = np.asarray(position_m, dtype=float)

    left_branch_ohm = (
        network.substation_resistance_ohm
        + (network.overhead_line_resistance_ohm_per_m + network.rail_resistance_ohm_per_m) * position
    )
    right_branch_ohm = (
        network.substation_resistance_ohm
        + (network.overhead_line_resistance_ohm_per_m + network.rail_resistance_ohm_per_m)
        * (line_length_m - position)
    )

    return (left_branch_ohm * right_branch_ohm) / (left_branch_ohm + right_branch_ohm)


def train_voltage_from_line_power(
    line_power_w: np.ndarray | float,
    position_m: np.ndarray | float,
    line_length_m: float,
    network: ElectricalNetwork,
    fallback_voltage_v: float = 500.0,
) -> np.ndarray:
    """Compute train voltage from line power using a Thevenin model.

    The electrical network is modeled as a voltage source with an equivalent
    series resistance. This leads to a quadratic equation in train voltage.

    Args:
        line_power_w (np.ndarray | float): Power drawn from the line, in watts.
        position_m (np.ndarray | float): Train position along the line, in meters.
        line_length_m (float): Total length of the line, in meters.
        network (ElectricalNetwork): Electrical network parameters.
        fallback_voltage_v (float): Voltage returned when the quadratic equation
            has no real solution.

    Returns:
        np.ndarray: Train voltage in volts.

    Notes:
        The voltage is obtained from:

        `V_train² - V_sst * V_train + R_eq * P = 0`

        Only the physically meaningful high-voltage root is used. If the
        discriminant is negative, the requested power exceeds what the network
        can supply at that position, and `fallback_voltage_v` is returned to
        avoid numerical failure.
    """
    power = np.asarray(line_power_w, dtype=float)
    resistance_ohm = equivalent_resistance(position_m, line_length_m, network)
    discriminant = network.substation_voltage_v**2 - 4 * resistance_ohm * power

    voltage = np.where(
        discriminant >= 0,
        0.5 * (network.substation_voltage_v + np.sqrt(np.maximum(discriminant, 0))),
        fallback_voltage_v,
    )
    return voltage