"""Plotting helpers for analysis notebooks.

This module provides convenience functions to visualize simulation results
and optimization outcomes. The goal is to quickly generate standard figures
used in exploratory analysis and reporting.

The plotting functions are intentionally lightweight and do not enforce a
specific visual style, allowing customization at the notebook level.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from .models import OptimizationResult, SimulationResult


def plot_simulation(result: SimulationResult) -> None:
    """Plot key time-domain signals from a simulation.

    The function generates a series of standard plots:

    - train position over time,
    - electrical power profiles (train demand and line supply),
    - train voltage,
    - battery energy (if available).

    Args:
        result (SimulationResult): Simulation output containing time series data.

    Notes:
        Battery-related plots are only displayed when battery data is present.
        Units are converted for readability (km, MW, kWh). This function is
        primarily intended for quick analysis in notebooks, not for
        publication-quality figures.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(result.time_s, result.position_m / 1000)
    plt.xlabel("Time [s]")
    plt.ylabel("Position [km]")
    plt.title("Train position")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(result.time_s, result.train_power_w / 1e6, label="Train")
    plt.plot(result.time_s, result.line_power_w / 1e6, label="Line")
    plt.xlabel("Time [s]")
    plt.ylabel("Power [MW]")
    plt.title("Power profiles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(result.time_s, result.train_voltage_v)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title("Train voltage")
    plt.grid(True)
    plt.tight_layout()

    if result.battery_energy_j is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(result.time_s, result.battery_energy_j / 3.6e6)
        plt.xlabel("Time [s]")
        plt.ylabel("Battery energy [kWh]")
        plt.title("Battery state of energy")
        plt.grid(True)
        plt.tight_layout()


def plot_optimization(result: OptimizationResult) -> None:
    """Visualize optimization results and Pareto front.

    Two complementary views are generated:

    1. Objective space: battery capacity vs. maximum voltage drop.
       This highlights the trade-off between cost and electrical performance.
    2. Decision space: battery capacity vs. power threshold.
       This shows how optimal solutions are distributed in the design space.

    Args:
        result (OptimizationResult): Optimization output containing evaluated
            designs and Pareto indices.

    Notes:
        Pareto-optimal solutions are highlighted for clarity. These plots are
        useful for identifying trends and selecting a compromise design
        (e.g. near the "elbow" of the Pareto front).
    """
    plt.figure(figsize=(7, 5))
    plt.scatter(result.capacities_kwh, result.max_voltage_drops_v, s=12, label="Evaluated designs")
    plt.scatter(result.pareto_capacities_kwh, result.pareto_voltage_drops_v, s=30, label="Pareto front")
    plt.xlabel("Battery capacity [kWh]")
    plt.ylabel("Maximum voltage drop [V]")
    plt.title("Objective space")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(7, 5))
    plt.scatter(result.capacities_kwh, result.power_thresholds_kw, s=12, label="Evaluated designs")
    plt.scatter(result.pareto_capacities_kwh, result.pareto_power_thresholds_kw, s=30, label="Pareto front")
    plt.xlabel("Battery capacity [kWh]")
    plt.ylabel("Power threshold [kW]")
    plt.title("Decision space")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()