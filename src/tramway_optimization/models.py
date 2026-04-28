"""Core data models for the tramway power-supply simulation.

This module defines immutable configuration and result containers used across
the simulation and optimization pipeline. The classes are intentionally kept
lightweight: they validate physical parameters, expose convenient unit
conversions, and centralize the data exchanged between modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ElectricalNetwork:
    """Electrical parameters of the DC tramway supply network.

    The network is modeled as two DC substations feeding the train through
    the overhead contact line and the rails. The parameters defined here are
    used to compute the Thevenin equivalent resistance seen by the train.

    Args:
        substation_voltage_v (float): No-load voltage of each substation, in volts.
        substation_resistance_ohm (float): Internal resistance of each substation, in ohms.
        overhead_line_resistance_ohm_per_m (float): Linear resistance of the overhead
            contact line, in ohms per meter.
        rail_resistance_ohm_per_m (float): Linear resistance of the rails, in ohms per meter.

    Notes:
        The default resistance values correspond to the corrected values used in
        the project statement to avoid physically impossible operating points.
    """

    substation_voltage_v: float = 790.0
    substation_resistance_ohm: float = 33e-3
    overhead_line_resistance_ohm_per_m: float = 95e-6
    rail_resistance_ohm_per_m: float = 10e-6

    def __post_init__(self) -> None:
        """Validate electrical network parameters."""
        if self.substation_voltage_v <= 0:
            raise ValueError("Substation voltage must be positive.")
        if self.substation_resistance_ohm < 0:
            raise ValueError("Substation resistance cannot be negative.")
        if self.overhead_line_resistance_ohm_per_m < 0:
            raise ValueError("Overhead line resistance cannot be negative.")
        if self.rail_resistance_ohm_per_m < 0:
            raise ValueError("Rail resistance cannot be negative.")


@dataclass(frozen=True)
class TrainConfig:
    """Mechanical and onboard electrical parameters of the train.

    This configuration groups the parameters required to estimate the train
    traction power from its position profile. The resistive force is described
    by a Davis-like polynomial model depending on speed and train mass.

    Args:
        mass_kg (float): Train mass, in kilograms.
        motor_efficiency (float): Constant efficiency of the motor and power conversion chain.
        onboard_auxiliary_power_w (float): Constant onboard auxiliary power demand, in watts.
        slope_rad (float): Track slope angle, in radians. A positive value represents an uphill slope.
        gravity_m_per_s2 (float): Gravitational acceleration, in meters per second squared.
        a0_n (float): Constant rolling-resistance coefficient, in newtons.
        a1_n_per_ton (float): Mass-dependent rolling-resistance coefficient.
        b0_n_per_kmh (float): Linear speed-dependent resistance coefficient.
        b1_n_per_ton_per_kmh (float): Mass-dependent linear speed-resistance coefficient.
        c0_n_per_kmh2 (float): Quadratic speed-dependent aerodynamic resistance coefficient.
        c1_n_per_ton_per_kmh2 (float): Mass-dependent quadratic speed-resistance coefficient.

    Notes:
        The model is intentionally simple because the project focuses on power
        supply simulation and multi-objective battery sizing, not detailed train
        dynamics.
    """

    mass_kg: float = 70e3
    motor_efficiency: float = 0.80
    onboard_auxiliary_power_w: float = 35e3
    slope_rad: float = 0.0
    gravity_m_per_s2: float = 9.81

    # Davis-like resistance coefficients.
    a0_n: float = 780.0
    a1_n_per_ton: float = 6.4
    b0_n_per_kmh: float = 0.0
    b1_n_per_ton_per_kmh: float = 0.14
    c0_n_per_kmh2: float = 0.3634
    c1_n_per_ton_per_kmh2: float = 0.0

    def __post_init__(self) -> None:
        """Validate train parameters."""
        if self.mass_kg <= 0:
            raise ValueError("Train mass must be positive.")
        if not 0 < self.motor_efficiency <= 1:
            raise ValueError("Motor efficiency must be in ]0, 1].")
        if self.onboard_auxiliary_power_w < 0:
            raise ValueError("Auxiliary power cannot be negative.")


@dataclass(frozen=True)
class BatteryConfig:
    """Battery sizing and control parameters.

    The battery is described by its energy capacity, efficiency, initial state
    of charge, and the power threshold used by the rule-based controller.

    Args:
        capacity_j (float): Maximum usable battery energy, in joules.
        power_threshold_w (float): Line power threshold, in watts. When the train demand
            exceeds this value, the controller attempts to supply the excess using the battery.
        efficiency (float): Battery charge/discharge efficiency.
        initial_state_of_charge (float): Initial battery state of charge, between 0 and 1.

    Notes:
        Internally, energy is stored in joules to remain consistent with SI units.
        The `from_kwh` constructor and `capacity_kwh` property are provided for
        user-facing battery sizing.
    """

    capacity_j: float
    power_threshold_w: float
    efficiency: float = 0.90
    initial_state_of_charge: float = 1.0

    @classmethod
    def from_kwh(
        cls,
        capacity_kwh: float,
        power_threshold_kw: float,
        efficiency: float = 0.90,
        initial_state_of_charge: float = 1.0,
    ) -> "BatteryConfig":
        """Create a battery configuration from engineering units.

        Args:
            capacity_kwh (float): Battery capacity, in kilowatt-hours.
            power_threshold_kw (float): Line power threshold, in kilowatts.
            efficiency (float): Battery charge/discharge efficiency.
            initial_state_of_charge (float): Initial battery state of charge, between 0 and 1.

        Returns:
            BatteryConfig: Battery configuration converted to SI units.
        """
        return cls(
            capacity_j=capacity_kwh * 3.6e6,
            power_threshold_w=power_threshold_kw * 1e3,
            efficiency=efficiency,
            initial_state_of_charge=initial_state_of_charge,
        )

    @property
    def capacity_kwh(self) -> float:
        """Battery capacity expressed in kilowatt-hours."""
        return self.capacity_j / 3.6e6

    @property
    def power_threshold_kw(self) -> float:
        """Line power threshold expressed in kilowatts."""
        return self.power_threshold_w / 1e3

    def __post_init__(self) -> None:
        """Validate battery sizing and control parameters."""
        if self.capacity_j < 0:
            raise ValueError("Battery capacity cannot be negative.")
        if self.power_threshold_w < 0:
            raise ValueError("Power threshold cannot be negative.")
        if not 0 < self.efficiency <= 1:
            raise ValueError("Battery efficiency must be in ]0, 1].")
        if not 0 <= self.initial_state_of_charge <= 1:
            raise ValueError("Initial state of charge must be in [0, 1].")


@dataclass(frozen=True)
class SimulationResult:
    """Time-domain outputs of a tramway power-supply simulation.

    This container stores the main time series produced by a simulation.
    Battery-related fields are optional so that the same result type can be
    used for simulations with and without onboard storage.

    Args:
        time_s (np.ndarray): Simulation time samples, in seconds.
        position_m (np.ndarray): Train position at each time sample, in meters.
        train_power_w (np.ndarray): Electrical power requested by the train, in watts.
            Positive values correspond to traction demand, while negative values correspond
            to regenerative braking.
        line_power_w (np.ndarray): Power supplied by the DC line, in watts.
        train_voltage_v (np.ndarray): Voltage at the train terminals, in volts.
        voltage_drop_v (np.ndarray): Difference between substation voltage and train voltage, in volts.
        battery_power_w (np.ndarray | None): Battery power, in watts. Positive values indicate
            discharge into the train, while negative values indicate charging.
        battery_energy_j (np.ndarray | None): Battery stored energy over time, in joules.
        rheostat_power_w (np.ndarray | None): Power dissipated in the rheostat during braking, in watts.
    """

    time_s: np.ndarray
    position_m: np.ndarray
    train_power_w: np.ndarray
    line_power_w: np.ndarray
    train_voltage_v: np.ndarray
    voltage_drop_v: np.ndarray
    battery_power_w: np.ndarray | None = None
    battery_energy_j: np.ndarray | None = None
    rheostat_power_w: np.ndarray | None = None

    @property
    def max_voltage_drop_v(self) -> float:
        """Maximum voltage drop observed during the simulation, in volts."""
        return float(np.max(self.voltage_drop_v))


@dataclass(frozen=True)
class OptimizationResult:
    """Result of a bi-objective battery optimization run.

    The optimization problem minimizes battery capacity and maximum voltage
    drop. This container stores all evaluated designs and the indices of the
    non-dominated solutions.

    Args:
        capacities_kwh (np.ndarray): Battery capacities evaluated during optimization, in kilowatt-hours.
        power_thresholds_kw (np.ndarray): Associated power thresholds, in kilowatts.
        max_voltage_drops_v (np.ndarray): Maximum voltage drop obtained for each evaluated design, in volts.
        pareto_indices (np.ndarray): Indices of non-dominated solutions in the evaluated design arrays.
    """

    capacities_kwh: np.ndarray
    power_thresholds_kw: np.ndarray
    max_voltage_drops_v: np.ndarray
    pareto_indices: np.ndarray

    @property
    def pareto_capacities_kwh(self) -> np.ndarray:
        """Battery capacities of the Pareto-optimal designs, in kWh."""
        return self.capacities_kwh[self.pareto_indices]

    @property
    def pareto_power_thresholds_kw(self) -> np.ndarray:
        """Power thresholds of the Pareto-optimal designs, in kW."""
        return self.power_thresholds_kw[self.pareto_indices]

    @property
    def pareto_voltage_drops_v(self) -> np.ndarray:
        """Maximum voltage drops of the Pareto-optimal designs, in volts."""
        return self.max_voltage_drops_v[self.pareto_indices]