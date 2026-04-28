# Simulation

The system is simulated in time.

At each step:

1. Compute train power
2. Apply battery control
3. Compute voltage


| Without Battery     | With Battery              |
|---------------------|---------------------------|
| All power from line | Charge during braking     |
| Regeneration lost   | Discharge above threshold |