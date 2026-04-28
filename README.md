# 🚋 Tramway Battery Storage Optimization

![Python](https://img.shields.io/badge/Python-3.10-blue)
![NumPy](https://img.shields.io/badge/NumPy-Scientific-lightgrey)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## Overview

A Python project for the **simulation and bi-objective optimization of onboard battery storage** in a DC tramway power-supply system.

The objective is to determine optimal battery configurations by minimizing two conflicting criteria:

- **Battery capacity** (cost proxy)
- **Maximum voltage drop** (electrical performance)

The project focuses on **physical modeling, numerical simulation, and multi-objective optimization**.

---

## Documentation

Full documentation is available here:

👉 https://01alicia.github.io/tramway-battery-optimization/

---

## Key Features

- Time-domain tramway simulation
- Train kinematics from trajectory data
- Electrical power modeling (traction & regenerative braking)
- Thevenin-based voltage computation
- Rule-based battery management system
- Monte Carlo design-space exploration
- Simplified NSGA-II-inspired optimization
- Pareto-front extraction and analysis
- Modular and well-documented codebase

---

## System Overview

The system models a tramway line powered by two DC substations.

- During **traction**, the line supplies power up to a threshold  
- The **battery supplies additional power** when needed  
- During **braking**, regenerative energy is:
  - stored in the battery if possible
  - otherwise dissipated in a rheostat  

---

## Architecture

The project follows a modular scientific architecture:

```bash
src/
├── data.py         # Data loading utilities
├── models.py       # Core data structures
├── physics.py      # Physical models
├── simulation.py   # Time-domain simulation
├── pareto.py       # Pareto front computation
├── optimization.py # Optimization algorithms
└── plotting.py     # Visualization helpers
```

---

## Design Principles

* Clear separation of concerns
* Reproducible simulations
* Vectorized numerical computation (NumPy)
* Lightweight and readable optimization methods

---

## Technical Highlights

### Physical Modeling

* Davis-like resistance model
* Traction force decomposition
* Power conversion with efficiency

### Electrical Modeling

* Thevenin equivalent network
* Position-dependent resistance
* Voltage feasibility handling

### Optimization

* Bi-objective formulation
* Monte Carlo sampling
* Genetic search (NSGA-II inspired)
* Pareto-front analysis

---

## Tech Stack

* Python 3.10
* NumPy
* Matplotlib
* MkDocs (documentation)

---

## 🛠 Installation

```bash
git clone https://github.com/01alicia/tramway-battery-optimization.git
cd tramway-battery-optimization

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/macOS

pip install -e ".[dev]"
```

---

## Running the Project

Basic simulation example:

```python
from tramway_optimization.data import load_train_run
from tramway_optimization.simulation import simulate_without_battery

time_s, position_m = load_train_run("data/marche.txt")
result = simulate_without_battery(time_s, position_m)

print(result.max_voltage_drop_v)
```

---

## Analysis Notebook

A complete analysis (simulation, optimization, plots) is available here:

👉 [`notebooks/tramway_battery_optimization_analysis.ipynb`](notebooks/tramway_battery_optimization_analysis.ipynb)

---

## Units

All internal computations use SI units:

| Quantity | Unit |
| -------- | ---- |
| Time     | s    |
| Position | m    |
| Power    | W    |
| Energy   | J    |
| Voltage  | V    |

Battery capacity is expressed in kWh at the interface level.

---

## Limitations

* Simplified physical models
* No detailed train dynamics
* Genetic algorithm is not a full NSGA-II implementation

---

## Future Work

* More realistic battery models
* Full NSGA-II implementation
* Multi-train interaction
* Real-world dataset integration

---

## Contributors

Developed as part of an academic project.

* Melissa Zina Belguitar
* Seydina Ba