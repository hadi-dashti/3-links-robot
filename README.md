# Three-Link Planar Robot

A project for simulating and validating the dynamics of a 3-link planar robot arm using both manual symbolic dynamics and PyBullet physics simulation.

## Project Structure

```
three_link_arm/
  README.md
  requirements.txt
  .gitignore
  LICENSE

  src/
    manual.py              # Manual symbolic dynamics simulator
    pybullet_sim.py        # PyBullet-based simulator

  assets/
    three_link_planar.urdf # Robot URDF model

  data/
    torque_dataset_100.csv # Full torque dataset
    torque_dataset_10.csv  # Sample dataset (10 rows)

  scripts/
    compare_and_report.py  # Validation and comparison script

  docs/
    validation_report.md   # Generated validation report

  results/                 # Generated results (gitignored)
    manual_results_full.csv
    pybullet_results_full.csv
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Manual Dynamics Simulation

```bash
python src/manual.py --input data/torque_dataset_100.csv --output results/manual_results_full.csv
```

### PyBullet Simulation

```bash
python src/pybullet_sim.py --input data/torque_dataset_100.csv --output results/pybullet_results_full.csv
```

### Compare Results and Generate Report

```bash
python scripts/compare_and_report.py \
  --manual results/manual_results_full.csv \
  --pybullet results/pybullet_results_full.csv \
  --out_md docs/validation_report.md
```

## Description

This project implements two simulation approaches for a 3-link planar robot:

1. **Manual Symbolic Dynamics**: Uses SymPy to derive equations of motion via Lagrangian mechanics
2. **PyBullet Simulation**: Uses PyBullet physics engine for validation

Both simulators run scenario-based simulations where each scenario:
- Starts from rest (q=0, dq=0)
- Applies a constant torque vector for duration T
- Records final state and dynamic terms (M, G, Cqdot)

The comparison script generates a detailed validation report analyzing differences between the two approaches.

