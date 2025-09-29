# Linear Model Predictive Control – Racetrack Assignment

This project is part of the **EL2700 – Model Predictive Control** course at KTH. It focuses on implementing a **Linear Model Predictive Controller (MPC)** for vehicle path tracking on a racetrack. Building on the previous LQR assignment, this project extends to constrained optimization, horizon tuning, and stability enhancements with terminal costs and dual-mode horizons.

## Project Overview

* **Problem Type:** Linear MPC with constraints
* **Context:** Vehicle dynamics & racing applications
* **Key Features:**

  * Discrete-time linearized vehicle model
  * Feedforward reference trajectory `(xff, uff)`
  * MPC design for state & input error stabilization
  * Constraint handling on steering, lateral velocity, and torque
  * Stability analysis via terminal cost scaling and dual-mode horizon
  * Performance comparison across horizons, solvers, and warm-start strategies

## Tasks

1. **MPC Formulation**

   * Define state & input error dynamics
   * Set up optimization problem in CVXPY
   * Include cost, dynamics, state/input constraints, and initial condition

2. **MPC Implementation**

   * Implement `setup_mpc_problem` (define variables, parameters, constraints)
   * Implement `solve_mpc` (return optimal first control input)

3. **LQR vs MPC Comparison**

   * Compare MPC optimal inputs with LQR baseline
   * Validate asymptotic stability and cost decay

4. **Controller Tuning**

   * Select cost matrices `(Q, R)` using Bryson’s rule or heuristics
   * Evaluate response trade-offs: tracking accuracy vs. control effort

5. **Constraints**

   * Add realistic limits:

     * Steering angle: |δ| ≤ 35°
     * Lateral velocity: |vy| ≤ 3.1 m/s
     * Steering rate: |uδ| ≤ 50 rad/s
     * Torque: |ur| ≤ 60 Nm

6. **Terminal Cost Tuning**

   * Scale terminal cost `QT = λP` with λ ∈ {2, 10, 50, 100}
   * Analyze how λ affects stability and performance

7. **Performance Experiments**

   * Vary horizon length N ∈ {20, 40, 80}
   * Compare solvers: MOSEK vs CLARABEL
   * Test warm-start on/off for speedup

8. **Bonus: Dual-Mode Horizon**

   * Implement extended horizon with dual states
   * Compare stability guarantees with standard MPC

## Results

* MPC matches LQR when unconstrained, but outperforms it under constraints.
* Larger horizons improve performance but increase computational time.
* Terminal cost scaling (λ) critical for ensuring closed-loop stability.
* Warm-starting and solver choice significantly affect runtime feasibility.

## Repository Structure

```
.
├── src/
│   ├── task4.py              # Entry point script
│   ├── mpc_controller.py     # MPC and MPCdual classes
│   ├── linear_car_model.py   # Vehicle model
│   ├── sim.py                # Simulation routines
│   ├── road.py               # Racetrack object
│   ├── u_ff.npy              # Feedforward input
│   ├── x_ff.npy              # Feedforward states
│   └── raceline.npy          # Racetrack reference
│
├── report/
│   └── Assignment4_GroupX.pdf
│
├── results/
│   ├── mpc_vs_lqr.png
│   ├── constraint_effects.png
│   ├── horizon_analysis.txt
│
└── README.md
```

## Usage Instructions

### Requirements

* Python 3.10+
* [CVXPY](https://www.cvxpy.org/) with solvers (CLARABEL, MOSEK)
* NumPy / SciPy

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Project

1. Run the main script:

   ```bash
   python task4.py
   ```

2. The script will:

   * Compute the MPC control sequence
   * Simulate vehicle tracking under constraints
   * Generate plots & logs in the `results/` folder

## Technologies

* **Python 3**
* **CVXPY** for MPC optimization
* **LQR and MPC theory** (stabilizing controllers)
* **Simulation-based evaluation**

## Authors

* Group X (EL2700 – Model Predictive Control, 2025)

---

📄 This repository contains the code, report, and results for Assignment 4 of the MPC course, focusing on Linear MPC with constraints, horizons, and stability guarantees.
