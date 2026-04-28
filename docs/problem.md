# Problem Formulation

The system consists of a tramway supplied by two substations.

The train moves along the line and consumes or regenerates power.

## Objectives

We aim to minimize:

1. Battery capacity (cost proxy)
2. Maximum voltage drop:

$$\Delta V_{max} = \max_t (V_{sst} - V_{train}(t))$$

## Decision Variables

- Battery capacity
- Power threshold

## Constraints

- No power feedback to the grid
- Battery limits (capacity, efficiency)

## Approach

- Simulation-based evaluation
- Pareto optimization