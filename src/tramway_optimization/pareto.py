"""Pareto-front utilities for bi-objective minimization.

This module provides basic tools to identify non-dominated solutions
(Pareto-optimal points) in a multi-objective optimization problem.

All objectives are assumed to be minimized.
"""

from __future__ import annotations

import numpy as np


def pareto_mask(objectives: np.ndarray) -> np.ndarray:
    """Identify non-dominated solutions using pairwise comparisons.

    A solution is considered non-dominated if no other solution is strictly
    better in at least one objective while being no worse in all others.

    Args:
        objectives (np.ndarray): Array of shape `(n_points, n_objectives)`
            containing objective values. Each row corresponds to one solution,
            and all objectives are assumed to be minimized.

    Returns:
        np.ndarray: Boolean array of shape `(n_points,)` where `True` indicates
        that the corresponding solution is non-dominated (Pareto-efficient).

    Raises:
        ValueError: If `objectives` is not a 2D array.

    Notes:
        This implementation performs a quadratic number of comparisons (O(n²)),
        which is sufficient for moderate-sized datasets typical of this project.

        The dominance relation used is:

        - solution A dominates solution B if:
          - A is no worse than B in all objectives, and
          - A is strictly better in at least one objective.
    """
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2:
        raise ValueError("Objectives must be a 2D array.")

    n_points = values.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        if not is_efficient[i]:
            continue

        dominated_by_i = np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1)
        if np.any(dominated_by_i):
            is_efficient[i] = False
            continue

        i_dominates = np.all(values[i] <= values, axis=1) & np.any(values[i] < values, axis=1)
        is_efficient[i_dominates] = False
        is_efficient[i] = True

    return is_efficient


def pareto_front(objectives: np.ndarray) -> np.ndarray:
    """Return indices of Pareto-optimal solutions.

    This function extracts the non-dominated solutions and sorts them
    according to the first objective for easier visualization (e.g. plotting
    a Pareto curve).

    Args:
        objectives (np.ndarray): Array of shape `(n_points, n_objectives)`
            containing objective values. All objectives are assumed to be minimized.

    Returns:
        np.ndarray: Indices of non-dominated solutions, sorted in ascending order
        of the first objective.

    Notes:
        Sorting by the first objective is convenient for visualization but does
        not affect the dominance relation itself.
    """
    indices = np.where(pareto_mask(objectives))[0]
    return indices[np.argsort(objectives[indices, 0])]