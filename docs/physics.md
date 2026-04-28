# Physical Model

## Train Dynamics

$$M \frac{dv}{dt} = F_{traction} - F_{resistance} - Mg \sin(\alpha)$$

## Resistive Forces

$$F_{resistive} = A + Bv + Cv^2$$



## Electrical Power

$$P_{electrical} =
\begin{cases}
\frac{P_{mechanical}}{\eta} & \text{if traction} \\
P_{mechanical} \cdot \eta & \text{if braking}
\end{cases}$$

## Electrical Network

Using Thévenin equivalent:

$$V_{train}^2 - V_{sst} V_{train} + R_{eq} P = 0$$