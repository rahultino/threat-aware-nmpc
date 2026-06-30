# Threat-Prioritized Nonlinear Multi-Agent Autonomous Driving via Time-to-Collision Cost Shaping


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework: CasADi](https://img.shields.io/badge/Framework-CasADi-orange.svg)](https://web.casadi.org/)

This repository contains the official implementation and simulation environment for the paper: **"Threat-Prioritized Nonlinear Multi-Agent Autonomous Driving via Time-to-Collision Cost Shaping"**. 

This project addresses the limitations of standard Nonlinear Model Predictive Control (NMPC) in dense multi-agent traffic environments. By introducing a temporally-aware cost shaping framework, it integrates **Time-to-Collision (TTC)** urgency directly into the NMPC objective function. This allows the ego vehicle to prioritize imminent threats dynamically without increasing the computational complexity, state/control dimensions, or constraint boundaries of the underlying optimization problem.

---

## 📌 Overview & Key Contributions

Standard NMPC obstacle avoidance relies heavily on spatial distance-based penalties (e.g., exponential or reciprocal costs). In dense, multi-agent scenarios, this spatial-only approach distributes control resources uniformly, failing to separate an immediate high-speed threat from a nearby non-threatening vehicle moving in parallel.

### Core Contributions:
1. **Temporally-Aware Cost Shaping:** Integrates TTC-based urgency modeling into NMPC without altering the optimization dimensionality or constraint structure.
2. **Multi-Component Threat Score:** Formulates a smooth, bounded threat metric $T_i \in [0, 1]$ aggregating spatial proximity, TTC, relative velocity, and semantic road-user class weighting (e.g., pedestrian, cyclist, vehicle).
3. **Smooth TTC Exposure Relaxation:** Formally interprets cumulative proximity-weighted risk exposure as a continuously differentiable relaxation of a discontinuous TTC threshold violation metric.
4. **Computational Preservation:** Retains identical asymptotic complexity $\mathcal{O}(MN)$ and solver characteristics (Jacobian dimension, Hessian sparsity pattern) compared to baseline NMPC.

---

## 🛠️ System Architecture & Framework

The vehicle's ego state is modeled using a nonlinear kinematic bicycle model:
* **States ($x$):** Global positions $(x, y)$, yaw angle $(\psi)$, and longitudinal velocity $(v)$.
* **Control Inputs ($u$):** Longitudinal acceleration ($a$) and front-wheel steering angle ($\delta$).

### Mathematical Core: Cost Reshaping

The modified NMPC objective modulates the exponential obstacle penalty utilizing the computed threat score $T_i$:

$$J_{\text{threat}} = \sum_{k=0}^{N-1} \left( \|x_{k} - x_{k}^{\text{ref}}\|_{Q}^{2} + \|u_{k}\|_{R}^{2} + \sum_{i=1}^{M} T_{i} \exp(-d_{i,k}) \right)$$

Where $T_i$ is evaluated at the control-update level and held constant over the prediction horizon to maintain structural efficiency and prevent dynamic coupling.

### Threat Score Breakdown:
* **Distance ($P_{d,i}$):** Spatial decay field using exponential mapping.
* **TTC ($P_{\text{ttc},i}$):** Smooth sigmoid mapping ensuring continuous differentiability for gradient-based solvers.
* **Velocity ($P_{v,i}$):** Normalized relative motion intensity tracking closing speed dynamics.
* **Class ($P_{c,i}$):** Predefined semantic importance factors prioritizing vulnerable road users.

---

