"""Data loading utilities for tramway simulations.

This module provides helper functions to load input data used in the
simulation pipeline. In particular, it focuses on loading time-position
profiles describing a train run.

The expected input format is intentionally simple to facilitate integration
with external tools (e.g. MATLAB, Excel, or measurement systems).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


def load_train_run(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a train trajectory from a text file.

    The input file is expected to contain at least two columns:

    - time in seconds,
    - position in meters.

    Additional columns are ignored. The function validates the data to ensure
    consistency with the assumptions made in the simulation pipeline.

    Args:
        path (str | Path): Path to a text file containing the train trajectory.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two arrays:

            - time vector in seconds,
            - position vector in meters.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid, contains insufficient data,
            or if time is not strictly increasing.

    Notes:
        - Time must be strictly increasing because it is used to compute
          derivatives (speed and acceleration) via numerical differentiation.
        - At least two samples are required to perform gradient-based
          computations.
        - The function relies on `numpy.loadtxt`, so the file must contain
          numeric values only.

    Examples:
        >>> time_s, position_m = load_train_run("data/train_run.txt")
        >>> time_s.shape
        (1000,)
        >>> position_m.shape
        (1000,)
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Train run file not found: {data_path}")

    data = np.loadtxt(data_path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Expected a text file with at least two columns: time and position.")

    time_s = data[:, 0].astype(float)
    position_m = data[:, 1].astype(float)

    if len(time_s) < 2:
        raise ValueError("At least two time samples are required.")

    if np.any(np.diff(time_s) <= 0):
        raise ValueError("Time values must be strictly increasing.")

    return time_s, position_m