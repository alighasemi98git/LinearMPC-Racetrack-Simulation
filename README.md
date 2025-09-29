# EL2700 - Linear Model Predictive Control (Assignment 4)  

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)  
[![License](https://img.shields.io/badge/License-Educational-green)](LICENSE)  
[![CVXPY](https://img.shields.io/badge/CVXPY-Optimization-orange)](https://www.cvxpy.org/)  

---

## Overview
This repository contains the implementation of a **Linear Model Predictive Control (MPC)** controller for vehicle trajectory tracking on a racetrack as part of EL2700 at KTH. The project demonstrates:

- Standard MPC and dual-mode MPC controller design.  
- Input and state constraints handling.  
- Simulation of MPC vs LQR performance.  
- Exploration of different horizons, solvers, and terminal cost parameters.  

---

## Repository Structure
EL2700-LinearMPC-Assignment4/
│
├── linear_car_model.py # Vehicle model parameters
├── road.py # Racetrack data
├── sim.py # Simulation routines
├── mpc_controller.py # MPC & dual-mode MPC implementation
├── task4.py # Main script for simulation
├── u_ff.npy # Feedforward input
├── x_ff.npy # Feedforward states
├── raceline.npy # Racetrack reference points
└── Assignment4_<STUDENT1>_<STUDENT2>.pdf # Report with answers, plots


---

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/EL2700-LinearMPC-Assignment4.git
pip install numpy scipy matplotlib cvxpy
python task4.py

## Features

Linear MPC for vehicle trajectory tracking.
Dual-mode MPC for enhanced stability.
Input and state constraints enforcement.
LQR vs MPC comparison.
Configurable horizon (N) and solver (CLARABEL or MOSEK).

## References

EL2700 Course Materials, KTH
Bryson Rule for controller tuning
CVXPY Documentation
