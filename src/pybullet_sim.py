import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data as pd_data


# ------------------------------
# URDF helpers (your existing file)
# ------------------------------
URDF_DEFAULT = "assets/three_link_planar.urdf"

TORQUE_COLS = ["tau1", "tau2", "tau3"]


def load_torque_csv(path: str) -> np.ndarray:
    """Load scenario torques from CSV with columns tau1,tau2,tau3."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"❌ Torque CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in TORQUE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"❌ Missing columns {missing} in torque CSV.\n"
            f"Found: {list(df.columns)}\nExpected: {TORQUE_COLS}"
        )

    return df[TORQUE_COLS].to_numpy(dtype=float)


def setup_pybullet(urdf_path: str, dt: float) -> int:
    """Connect (DIRECT), load URDF, set timestep and gravity, return robot_id."""
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pd_data.getDataPath())
    p.setTimeStep(float(dt))

    # Keep your setting (planar in XY, gravity along -Y)
    p.setGravity(0, -9.81, 0)

    robot_id = p.loadURDF(str(urdf_path), basePosition=[0, 0, 0], useFixedBase=True)

    # Disable default motors
    p.setJointMotorControlArray(
        robot_id,
        jointIndices=[0, 1, 2],
        controlMode=p.VELOCITY_CONTROL,
        forces=[0.0, 0.0, 0.0],
    )
    return robot_id


def reset_robot(robot_id: int) -> None:
    """Reset q and dq to zero."""
    for j in range(3):
        p.resetJointState(robot_id, j, targetValue=0.0, targetVelocity=0.0)


def get_q_dq(robot_id: int):
    """Read q and dq from PyBullet."""
    states = p.getJointStates(robot_id, [0, 1, 2])
    q = np.array([s[0] for s in states], dtype=float)
    dq = np.array([s[1] for s in states], dtype=float)
    return q, dq


def compute_M_G_Cqdot(robot_id: int, q: np.ndarray, dq: np.ndarray):
    """
    Compute:
      M(q) using calculateMassMatrix
      G(q) using inverse dynamics at dq=0, ddq=0
      C(q,dq)*dq using h(q,dq)=ID(q,dq,ddq=0) and subtract G
    """
    q_list = q.tolist()
    dq_list = dq.tolist()
    zeros = [0.0, 0.0, 0.0]

    # Mass matrix
    M = np.array(p.calculateMassMatrix(robot_id, q_list), dtype=float)  # (3,3)

    # Gravity term
    G = np.array(p.calculateInverseDynamics(robot_id, q_list, zeros, zeros), dtype=float)  # (3,)

    # h(q,dq) = C(q,dq)*dq + G(q) (for ddq=0)
    h = np.array(p.calculateInverseDynamics(robot_id, q_list, dq_list, zeros), dtype=float)

    Cqdot = h - G
    return M, G, Cqdot


def simulate_one_scenario(robot_id: int, tau: np.ndarray, dt: float, T: float):
    """
    Scenario-based simulation:
      - reset to q=dq=0
      - apply constant torque tau for duration T
      - return final q, dq, ddq (finite-difference), plus M,G,Cqdot at final state
    """
    tau = np.asarray(tau, dtype=float).reshape(3,)
    dt = float(dt)
    T = float(T)
    n_steps = int(T / dt)

    reset_robot(robot_id)

    dq_prev = None

    for _ in range(n_steps):
        # Apply constant torque and step
        p.setJointMotorControlArray(
            robot_id,
            jointIndices=[0, 1, 2],
            controlMode=p.TORQUE_CONTROL,
            forces=tau.tolist(),
        )
        p.stepSimulation()

        # Track dq for ddq estimate at the end
        _, dq_now = get_q_dq(robot_id)
        dq_prev = dq_now.copy()

    # Final state after T
    q, dq = get_q_dq(robot_id)

    # Estimate ddq with one extra step finite difference (same spirit as your old code)
    p.setJointMotorControlArray(
        robot_id,
        jointIndices=[0, 1, 2],
        controlMode=p.TORQUE_CONTROL,
        forces=tau.tolist(),
    )
    p.stepSimulation()
    _, dq2 = get_q_dq(robot_id)
    ddq = (dq2 - dq) / dt

    # Dynamics terms at final state (q,dq)
    M, G, Cqdot = compute_M_G_Cqdot(robot_id, q, dq)

    return q, dq, ddq, M, G, Cqdot


def save_results_csv(rows: np.ndarray, out_path: str) -> None:
    """
    Save scenario summary with column order:
    tau1 tau2 tau3 q1 q2 q3 dq1 dq2 dq3 ddq1 ddq2 ddq3 G1 G2 G3 Cqdot1 Cqdot2 Cqdot3 M1..M9
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "tau1", "tau2", "tau3",
        "q1", "q2", "q3",
        "dq1", "dq2", "dq3",
        "ddq1", "ddq2", "ddq3",
        "G1", "G2", "G3",
        "Cqdot1", "Cqdot2", "Cqdot3",
        "M1", "M2", "M3",
        "M4", "M5", "M6",
        "M7", "M8", "M9",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(out_path, index=False)


def parse_args():
    ap = argparse.ArgumentParser(description="PyBullet 3-link: scenario-based simulation with manual-like dynamics columns.")
    ap.add_argument("--input", "-i", required=True, help="Torque CSV (columns: tau1,tau2,tau3)")
    ap.add_argument("--output", "-o", required=True, help="Output CSV path")
    ap.add_argument("--dt", type=float, default=0.001, help="Timestep (default: 0.001)")
    ap.add_argument("--T", type=float, default=1.0, help="Duration per scenario in seconds (default: 1.0)")
    ap.add_argument("--urdf", type=str, default=URDF_DEFAULT, help="URDF path")
    ap.add_argument("--progress_every", type=int, default=10, help="Print progress every N scenarios (default: 10)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tau_array = load_torque_csv(args.input)
    print("✅ Loaded torque scenarios:", tau_array.shape)

    if not Path(args.urdf).exists():
        raise FileNotFoundError(f"❌ URDF not found: {args.urdf}")

    robot_id = setup_pybullet(args.urdf, dt=args.dt)

    rows = []
    for i, tau in enumerate(tau_array):
        q, dq, ddq, M, G, Cqdot = simulate_one_scenario(robot_id, tau, dt=args.dt, T=args.T)

        row = np.hstack([
            tau,            # 3
            q,              # 3
            dq,             # 3
            ddq,            # 3
            G,              # 3
            Cqdot,          # 3
            M.reshape(-1),  # 9
        ])
        rows.append(row)

        if args.progress_every > 0 and (i + 1) % args.progress_every == 0:
            print(f"Simulated {i+1}/{len(tau_array)} scenarios...")

    p.disconnect()

    rows = np.asarray(rows, dtype=float)
    save_results_csv(rows, args.output)

    print("\n✅ Done.")
    print("✅ Saved to:", Path(args.output).resolve())
