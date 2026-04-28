"""Time-domain tramway simulations with and without onboard battery storage.

This module contains the main simulation routines used to evaluate a battery
design. The simulation is performed sample by sample over a train trajectory.

Two configurations are supported:

- baseline operation without battery;
- hybrid operation with onboard battery storage and rule-based energy
  management.
"""

from __future__ import annotations

import numpy as np

from .models import BatteryConfig, ElectricalNetwork, SimulationResult, TrainConfig
from .physics import compute_train_electrical_power, train_voltage_from_line_power


def simulate_without_battery(
    time_s: np.ndarray,
    position_m: np.ndarray,
    network: ElectricalNetwork | None = None,
    train: TrainConfig | None = None,
) -> SimulationResult:
    """Simulate the tramway supply system without onboard battery storage.

    In this baseline configuration, positive train power is fully supplied by
    the DC line. Negative train power, corresponding to regenerative braking,
    is not sent back to the grid and is therefore ignored from the line-power
    point of view.

    Args:
        time_s (np.ndarray): Simulation time samples, in seconds.
        position_m (np.ndarray): Train position at each time sample, in meters.
        network (ElectricalNetwork | None): Electrical network configuration.
            If `None`, default parameters are used.
        train (TrainConfig | None): Train configuration. If `None`, default
            parameters are used.

    Returns:
        SimulationResult: Time-domain simulation outputs, including train power,
        line power, train voltage and voltage drop.

    Notes:
        This function provides the reference case used to evaluate the benefit of
        adding onboard battery storage.
    """
    network = network or ElectricalNetwork()
    train = train or TrainConfig()

    train_power_w = compute_train_electrical_power(time_s, position_m, train)
    line_power_w = np.maximum(train_power_w, 0.0)

    line_length_m = float(np.max(position_m))
    train_voltage_v = train_voltage_from_line_power(line_power_w, position_m, line_length_m, network)
    voltage_drop_v = network.substation_voltage_v - train_voltage_v

    return SimulationResult(
        time_s=time_s,
        position_m=position_m,
        train_power_w=train_power_w,
        line_power_w=line_power_w,
        train_voltage_v=train_voltage_v,
        voltage_drop_v=voltage_drop_v,
    )


def apply_battery_rule(
    train_power_w: float,
    battery_energy_j: float,
    battery: BatteryConfig,
    dt_s: float,
) -> tuple[float, float, float, float]:
    """Apply the rule-based battery controller for one time step.

    The controller follows a simple energy-management strategy:

    - during braking, regenerative power is stored in the battery when free
      capacity is available;
    - when traction demand is below the threshold, the line supplies all power;
    - when traction demand is above the threshold, the battery supplies the
      excess power if enough energy is available;
    - energy that cannot be recovered during braking is dissipated in the
      rheostat.

    Args:
        train_power_w (float): Electrical power requested by the train, in watts.
            Positive values represent traction demand, while negative values
            represent braking.
        battery_energy_j (float): Battery energy available at the beginning of
            the time step, in joules.
        battery (BatteryConfig): Battery configuration containing capacity,
            threshold and efficiency.
        dt_s (float): Time-step duration, in seconds.

    Returns:
        tuple[float, float, float, float]: Line power in watts, battery power in
        watts, rheostat power in watts, and updated battery energy in joules.

    Raises:
        ValueError: If `dt_s` is not strictly positive.

    Notes:
        Battery power follows the project sign convention:

        - positive battery power means discharge into the train;
        - negative battery power means charging during regenerative braking.

        Battery efficiency is applied differently during charge and discharge:
        during charge, only a fraction of recovered braking energy is stored;
        during discharge, more stored energy is consumed than the power delivered
        to the train because of losses.
    """
    if dt_s <= 0:
        raise ValueError("Time step must be positive.")

    battery_energy_j = float(np.clip(battery_energy_j, 0.0, battery.capacity_j))

    if battery.capacity_j == 0:
        line_power_w = max(train_power_w, 0.0)
        rheostat_power_w = max(-train_power_w, 0.0)
        return line_power_w, 0.0, rheostat_power_w, 0.0

    # Regenerative braking: store as much as possible.
    if train_power_w <= 0:
        available_storage_j = battery.capacity_j - battery_energy_j
        requested_charge_power_w = -train_power_w
        max_charge_power_w = (
            available_storage_j / (battery.efficiency * dt_s)
            if available_storage_j > 0
            else 0.0
        )

        actual_charge_power_w = min(requested_charge_power_w, max_charge_power_w)
        battery_power_w = -actual_charge_power_w
        battery_energy_j += actual_charge_power_w * battery.efficiency * dt_s

        rheostat_power_w = requested_charge_power_w - actual_charge_power_w
        return 0.0, battery_power_w, rheostat_power_w, battery_energy_j

    # Traction below threshold: line supplies all power.
    if train_power_w <= battery.power_threshold_w:
        return train_power_w, 0.0, 0.0, battery_energy_j

    # Traction above threshold: battery supplies the excess if possible.
    requested_discharge_power_w = train_power_w - battery.power_threshold_w
    max_discharge_power_w = battery_energy_j * battery.efficiency / dt_s
    actual_discharge_power_w = min(requested_discharge_power_w, max_discharge_power_w)

    battery_power_w = actual_discharge_power_w
    battery_energy_j -= actual_discharge_power_w * dt_s / battery.efficiency
    line_power_w = train_power_w - actual_discharge_power_w

    return line_power_w, battery_power_w, 0.0, battery_energy_j


def simulate_with_battery(
    time_s: np.ndarray,
    position_m: np.ndarray,
    battery: BatteryConfig,
    network: ElectricalNetwork | None = None,
    train: TrainConfig | None = None,
) -> SimulationResult:
    """Simulate the tramway supply system with onboard battery storage.

    This simulation evaluates the complete hybrid system over the train trip.
    At each time step, train power demand is computed, the battery rule is
    applied, and the resulting line power is used to compute train voltage.

    Args:
        time_s (np.ndarray): Simulation time samples, in seconds.
        position_m (np.ndarray): Train position at each time sample, in meters.
        battery (BatteryConfig): Battery configuration and control parameters.
        network (ElectricalNetwork | None): Electrical network configuration.
            If `None`, default parameters are used.
        train (TrainConfig | None): Train configuration. If `None`, default
            parameters are used.

    Returns:
        SimulationResult: Time-domain simulation outputs, including line power,
        battery power, rheostat power, battery energy, train voltage and voltage
        drop.

    Notes:
        The battery state is propagated sequentially, so the result depends on
        the full history of traction and braking events. This is why each battery
        design must be evaluated through a complete time-domain simulation during
        optimization.
    """
    network = network or ElectricalNetwork()
    train = train or TrainConfig()

    train_power_w = compute_train_electrical_power(time_s, position_m, train)

    line_power = np.zeros_like(train_power_w)
    battery_power = np.zeros_like(train_power_w)
    rheostat_power = np.zeros_like(train_power_w)
    battery_energy = np.zeros_like(train_power_w)

    current_energy_j = battery.initial_state_of_charge * battery.capacity_j

    for i, power_w in enumerate(train_power_w):
        dt_s = float(time_s[i] - time_s[i - 1]) if i > 0 else float(time_s[1] - time_s[0])
        line_power[i], battery_power[i], rheostat_power[i], current_energy_j = apply_battery_rule(
            train_power_w=float(power_w),
            battery_energy_j=current_energy_j,
            battery=battery,
            dt_s=dt_s,
        )
        battery_energy[i] = current_energy_j

    line_length_m = float(np.max(position_m))
    train_voltage_v = train_voltage_from_line_power(line_power, position_m, line_length_m, network)
    voltage_drop_v = network.substation_voltage_v - train_voltage_v

    return SimulationResult(
        time_s=time_s,
        position_m=position_m,
        train_power_w=train_power_w,
        line_power_w=line_power,
        train_voltage_v=train_voltage_v,
        voltage_drop_v=voltage_drop_v,
        battery_power_w=battery_power,
        battery_energy_j=battery_energy,
        rheostat_power_w=rheostat_power,
    )