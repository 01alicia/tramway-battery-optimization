"""Bi-objective optimization methods for battery sizing and control.

This module provides two optimization approaches for the tramway onboard
battery design problem:

- Monte Carlo random sampling of the decision space.
- A simplified NSGA-II-inspired genetic search.

The optimization problem minimizes two objectives:

1. battery capacity, used as a proxy for cost;
2. maximum voltage drop at the train terminals, used as an electrical
   performance indicator.
"""

from __future__ import annotations

import numpy as np

from .models import BatteryConfig, ElectricalNetwork, OptimizationResult, TrainConfig
from .pareto import pareto_front
from .simulation import simulate_with_battery


def evaluate_design(
    time_s: np.ndarray,
    position_m: np.ndarray,
    capacity_kwh: float,
    power_threshold_kw: float,
    network: ElectricalNetwork | None = None,
    train: TrainConfig | None = None,
    battery_efficiency: float = 0.90,
) -> float:
    """Evaluate a battery design through a full time-domain simulation.

    A design is defined by a battery capacity and a line power threshold.
    The battery is simulated over the whole trip using the rule-based energy
    management strategy, then the maximum voltage drop is returned.

    Args:
        time_s (np.ndarray): Simulation time samples, in seconds.
        position_m (np.ndarray): Train position at each time sample, in meters.
        capacity_kwh (float): Battery capacity, in kilowatt-hours.
        power_threshold_kw (float): Line power threshold, in kilowatts.
        network (ElectricalNetwork | None): Electrical network configuration.
            If `None`, default parameters are used.
        train (TrainConfig | None): Train configuration. If `None`, default parameters are used.
        battery_efficiency (float): Battery charge/discharge efficiency.

    Returns:
        float: Maximum voltage drop over the trip, in volts.
    """
    battery = BatteryConfig.from_kwh(
        capacity_kwh=capacity_kwh,
        power_threshold_kw=power_threshold_kw,
        efficiency=battery_efficiency,
    )
    result = simulate_with_battery(time_s, position_m, battery, network=network, train=train)
    return result.max_voltage_drop_v


def monte_carlo_search(
    time_s: np.ndarray,
    position_m: np.ndarray,
    n_samples: int = 1000,
    capacity_bounds_kwh: tuple[float, float] = (0.0, 14.0),
    threshold_bounds_kw: tuple[float, float] = (0.0, 1000.0),
    random_seed: int | None = 42,
    network: ElectricalNetwork | None = None,
    train: TrainConfig | None = None,
) -> OptimizationResult:
    """Explore the design space by random sampling.

    This method samples battery capacities and power thresholds uniformly
    within the provided bounds. Each sampled design is evaluated independently
    through a complete simulation. The Pareto front is then extracted from the
    objective space using battery capacity and maximum voltage drop.

    Args:
        time_s (np.ndarray): Simulation time samples, in seconds.
        position_m (np.ndarray): Train position at each time sample, in meters.
        n_samples (int): Number of random designs to evaluate.
        capacity_bounds_kwh (tuple[float, float]): Lower and upper bounds for
            battery capacity, in kilowatt-hours.
        threshold_bounds_kw (tuple[float, float]): Lower and upper bounds for the
            line power threshold, in kilowatts.
        random_seed (int | None): Seed used by NumPy's random generator. Use `None`
            for non-reproducible sampling.
        network (ElectricalNetwork | None): Electrical network configuration.
            If `None`, default parameters are used.
        train (TrainConfig | None): Train configuration. If `None`, default parameters are used.

    Returns:
        OptimizationResult: Evaluated designs, objective values, and indices of
        non-dominated solutions.

    Notes:
        Monte Carlo sampling is simple and robust, but inefficient: many evaluated
        designs may be far from the Pareto front.
    """
    rng = np.random.default_rng(random_seed)

    capacities_kwh = rng.uniform(*capacity_bounds_kwh, size=n_samples)
    thresholds_kw = rng.uniform(*threshold_bounds_kw, size=n_samples)

    voltage_drops = np.array([
        evaluate_design(time_s, position_m, c, p, network=network, train=train)
        for c, p in zip(capacities_kwh, thresholds_kw)
    ])

    objectives = np.column_stack([capacities_kwh, voltage_drops])
    indices = pareto_front(objectives)

    return OptimizationResult(
        capacities_kwh=capacities_kwh,
        power_thresholds_kw=thresholds_kw,
        max_voltage_drops_v=voltage_drops,
        pareto_indices=indices,
    )


def _rank_by_dominance(objectives: np.ndarray) -> np.ndarray:
    """Compute a simple dominance count for each solution.

    The returned rank is the number of other solutions that dominate the
    current solution. Lower values therefore correspond to better solutions.

    Args:
        objectives (np.ndarray): Objective matrix of shape
            `(n_solutions, n_objectives)`. All objectives are minimized.

    Returns:
        np.ndarray: Dominance count for each solution.

    Notes:
        This is not the full non-dominated sorting procedure used in NSGA-II.
        It is a lightweight approximation suitable for a compact academic project.
    """
    ranks = np.zeros(len(objectives), dtype=int)
    for i, point in enumerate(objectives):
        dominated_by = np.all(objectives <= point, axis=1) & np.any(objectives < point, axis=1)
        ranks[i] = int(np.sum(dominated_by))
    return ranks


def _crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Create one child design by linear crossover.

    Each gene of the child is obtained as a random linear combination of the
    corresponding genes of two parents. The random weight is sampled in
    `[-0.1, 1.1]` to allow slight exploration outside the interval defined by
    the parents.

    Args:
        parent_a (np.ndarray): First parent genome.
        parent_b (np.ndarray): Second parent genome.
        rng (np.random.Generator): NumPy random number generator.

    Returns:
        np.ndarray: Child genome.
    """
    weights = rng.uniform(-0.1, 1.1, size=parent_a.shape)
    return weights * parent_a + (1.0 - weights) * parent_b


def simplified_nsga2_search(
    time_s: np.ndarray,
    position_m: np.ndarray,
    population_size: int = 100,
    n_generations: int = 10,
    capacity_bounds_kwh: tuple[float, float] = (0.0, 14.0),
    threshold_bounds_kw: tuple[float, float] = (0.0, 1000.0),
    random_seed: int | None = 42,
    network: ElectricalNetwork | None = None,
    train: TrainConfig | None = None,
) -> OptimizationResult:
    """Approximate the Pareto front with a simplified genetic search.

    The algorithm starts from a random population of battery designs. At each
    generation, all individuals are evaluated, ranked using a dominance count,
    and the best half of the population is selected as parents. New individuals
    are created by crossover until the population size is restored.

    Args:
        time_s (np.ndarray): Simulation time samples, in seconds.
        position_m (np.ndarray): Train position at each time sample, in meters.
        population_size (int): Number of individuals maintained at each generation.
        n_generations (int): Number of generations to run.
        capacity_bounds_kwh (tuple[float, float]): Lower and upper bounds for
            battery capacity, in kilowatt-hours.
        threshold_bounds_kw (tuple[float, float]): Lower and upper bounds for the
            line power threshold, in kilowatts.
        random_seed (int | None): Seed used by NumPy's random generator. Use `None`
            for stochastic runs.
        network (ElectricalNetwork | None): Electrical network configuration.
            If `None`, default parameters are used.
        train (TrainConfig | None): Train configuration. If `None`, default parameters are used.

    Returns:
        OptimizationResult: All evaluated designs across generations, their
        objective values, and the extracted Pareto front.

    Notes:
        This function is intentionally not a full NSGA-II implementation. It keeps
        the main educational ideas: population-based search, dominance-based
        selection, crossover, and repeated improvement over generations.
    """
    rng = np.random.default_rng(random_seed)

    population = np.column_stack([
        rng.uniform(*capacity_bounds_kwh, size=population_size),
        rng.uniform(*threshold_bounds_kw, size=population_size),
    ])

    all_designs = []

    for _ in range(n_generations):
        voltage_drops = np.array([
            evaluate_design(time_s, position_m, c, p, network=network, train=train)
            for c, p in population
        ])
        all_designs.append(np.column_stack([population, voltage_drops]))

        objectives = np.column_stack([population[:, 0], voltage_drops])
        ranks = _rank_by_dominance(objectives)
        selected_indices = np.argsort(ranks)[: population_size // 2]
        parents = population[selected_indices]

        children = []
        while len(children) < population_size - len(parents):
            i, j = rng.choice(len(parents), size=2, replace=False)
            child = _crossover(parents[i], parents[j], rng)
            child[0] = np.clip(child[0], *capacity_bounds_kwh)
            child[1] = np.clip(child[1], *threshold_bounds_kw)
            children.append(child)

        population = np.vstack([parents, np.asarray(children)])

    history = np.vstack(all_designs)
    capacities_kwh = history[:, 0]
    thresholds_kw = history[:, 1]
    voltage_drops = history[:, 2]

    objectives = np.column_stack([capacities_kwh, voltage_drops])
    indices = pareto_front(objectives)

    return OptimizationResult(
        capacities_kwh=capacities_kwh,
        power_thresholds_kw=thresholds_kw,
        max_voltage_drops_v=voltage_drops,
        pareto_indices=indices,
    )