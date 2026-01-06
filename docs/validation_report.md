# Validation Report: Manual vs PyBullet (Scenario-Based, Final-Only)

This report compares a symbolic manual dynamics simulator against PyBullet for independent torque scenarios. Each scenario starts from q0=[0,0,0], dq0=[0,0,0], applies a constant torque vector for duration T, and records the final state and dynamic terms.

- Scenarios: **100**
- dt: **0.001 s**
- T: **1.0 s**

## 1) Input consistency

- Torque MAE (manual vs PyBullet): **0.000e+00**
- ✅ Inputs are identical; differences are due to modeling/solver differences rather than different torques.

## 2) Error metrics summary

| Quantity | MAE | RMSE | Rel. MAE (vs mean |manual|) |
|---|---:|---:|---:|
| q | 2.154e+00 | 3.919e+00 | 6.121e-01 |
| dq | 7.334e+00 | 1.163e+01 | 9.304e-01 |
| ddq | 9.403e+01 | 1.727e+02 | 9.232e-01 |
| G | 5.275e+00 | 7.412e+00 | 4.564e-01 |
| Cqdot | 5.906e+01 | 1.275e+02 | 9.546e-01 |
| M(flat) | 7.179e-01 | 1.173e+00 | 4.482e-01 |

## 3) Diagnostics (root-cause analysis)

### A) Joint limits / bounded angles

- PyBullet may clamp angles if URDF joint limits are set (commonly [-pi, +pi]).
- Manual models typically allow unbounded angles unless limits are explicitly implemented.

- PyBullet near-±pi hit rate (per joint): **[0.06 0.   0.31]**
- Manual |q| > 2π rate (per joint): **[0.   0.01 0.53]**
- ✅ Evidence suggests PyBullet is frequently reaching joint limits; this changes the final state and all dependent terms.

### B) Acceleration (ddq) definition

- Manual ddq is obtained directly from the dynamic equation.
- In this pipeline, PyBullet ddq is estimated using finite differences; this can deviate from model-based ddq.

- PyBullet ddq consistency (reported vs model-based): MAE = **4.138e+00**

### C) Solver / integration differences

- Manual uses a simple explicit integrator (semi-implicit Euler).
- PyBullet uses an engine solver with iterative updates and internal stabilization.
- Even with identical parameters, different numerical methods yield different trajectories and final states.

### D) Inertial / frame conventions mismatch

- Manual dynamics are derived from explicit (L, m, I, COM) assumptions.
- PyBullet dynamics are determined by URDF mass, inertia tensor, COM offsets, and link/joint frames.
- Small mismatches in URDF inertial parameters or gravity axis conventions can produce meaningful differences.

### Summary of dominant causes detected

- **PyBullet joint-limit saturation (URDF limits near ±pi).**
- **Manual model uses unbounded joint angles (no joint limits / wrapping).**
- **PyBullet ddq is estimated by finite-difference; model-based ddq can differ noticeably.**
- **Final state mismatch (q,dq) amplifies differences in M(q), G(q), and C(q,dq)*dq.**

## 4) Decision: adopt PyBullet as the main simulation reference

For future development and reporting, we adopt **PyBullet** as the primary simulation reference because:
- It enforces physically meaningful constraints (e.g., joint limits).
- It is widely used in robotics research and improves reproducibility for others.
- It scales better to later extensions (controllers, contacts, sensors, visualization).

The manual symbolic model will be kept as a **secondary baseline** for sanity checks and debugging (e.g., verifying trends, validating dynamic term extraction, and unit testing).

## 5) Recommendations to reduce mismatch (if desired)

1) **Align joint limits**: either increase/remove URDF limits or implement the same limits in the manual model.
2) **Align inertial parameters**: match URDF COM and inertia tensor to the manual assumptions exactly.
3) **Align gravity/frame conventions**: confirm axis directions match across both implementations.
4) **Standardize ddq**: compare ddq using the model-based definition `ddq = inv(M)*(tau - Cqdot - G)`.
