"""Tramway battery storage optimization package.

This package provides a complete pipeline for the simulation and
bi-objective optimization of onboard battery storage in DC tramway
power-supply systems.

It includes:

- data loading utilities,
- physical modeling of train dynamics and electrical networks,
- time-domain simulation with and without onboard battery storage,
- Pareto-front computation,
- multi-objective optimization methods.

The main objective is to evaluate battery sizing and control strategies
by minimizing:

1. battery capacity (cost proxy),
2. maximum voltage drop at the train terminals.

Typical usage:

    >>> from tramway_optimization import load_train_run, simulate_without_battery
    >>> time_s, position_m = load_train_run("data/marche.txt")
    >>> result = simulate_without_battery(time_s, position_m)

Modules are designed to remain lightweight, readable, and suitable for
repeated evaluations within optimization workflows.
"""

# from .data import load_train_run
# from .models import (
#     BatteryConfig,
#     ElectricalNetwork,
#     TrainConfig,
#     SimulationResult,
#     OptimizationResult,
# )
# from .simulation import simulate_without_battery, simulate_with_battery
# from .pareto import pareto_mask, pareto_front
# from .optimization import monte_carlo_search, simplified_nsga2_search
#
# __all__ = [
#     "load_train_run",
#     "BatteryConfig",
#     "ElectricalNetwork",
#     "TrainConfig",
#     "SimulationResult",
#     "OptimizationResult",
#     "simulate_without_battery",
#     "simulate_with_battery",
#     "pareto_mask",
#     "pareto_front",
#     "monte_carlo_search",
#     "simplified_nsga2_search",
# ]