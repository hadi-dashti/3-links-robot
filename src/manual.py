import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp


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


class ThreeLinkArmExact:
    """
    Exact dynamics model for a planar 3-link arm (no friction).
    Builds symbolic M(q), G(q), and C(q,dq)*dq using Lagrangian mechanics.
    """

    def __init__(
        self,
        L=np.array([1.0, 1.0, 1.0]),
        m=np.array([1.0, 1.0, 1.0]),
        I=np.array([0.1, 0.1, 0.1]),
        g=9.81,
        torque_limits=np.array([10.0, 10.0, 10.0]),
    ):
        self.L = np.array(L, dtype=float)
        self.m = np.array(m, dtype=float)
        self.I = np.array(I, dtype=float)
        self.g = float(g)
        self.torque_limits = np.array(torque_limits, dtype=float)

        self._build_symbolic_dynamics()

    def _build_symbolic_dynamics(self):
        q1, q2, q3 = sp.symbols("q1 q2 q3", real=True)
        dq1, dq2, dq3 = sp.symbols("dq1 dq2 dq3", real=True)
        ddq1, ddq2, ddq3 = sp.symbols("ddq1 ddq2 ddq3", real=True)

        q_vec = [q1, q2, q3]
        dq_vec = [dq1, dq2, dq3]
        ddq_vec = [ddq1, ddq2, ddq3]

        L1, L2, L3 = map(float, self.L)
        m1, m2, m3 = map(float, self.m)
        I1, I2, I3 = map(float, self.I)
        g = float(self.g)

        # COM positions
        x1 = (L1 / 2.0) * sp.cos(q1)
        y1 = (L1 / 2.0) * sp.sin(q1)

        x2 = L1 * sp.cos(q1) + (L2 / 2.0) * sp.cos(q1 + q2)
        y2 = L1 * sp.sin(q1) + (L2 / 2.0) * sp.sin(q1 + q2)

        x3 = (
            L1 * sp.cos(q1)
            + L2 * sp.cos(q1 + q2)
            + (L3 / 2.0) * sp.cos(q1 + q2 + q3)
        )
        y3 = (
            L1 * sp.sin(q1)
            + L2 * sp.sin(q1 + q2)
            + (L3 / 2.0) * sp.sin(q1 + q2 + q3)
        )

        dq_sym = sp.Matrix(dq_vec)

        def jacobian_times_dq(x_expr, y_expr):
            Jx = sp.Matrix([x_expr]).jacobian(sp.Matrix(q_vec))
            Jy = sp.Matrix([y_expr]).jacobian(sp.Matrix(q_vec))
            vx = (Jx * dq_sym)[0]
            vy = (Jy * dq_sym)[0]
            return vx, vy

        vx1, vy1 = jacobian_times_dq(x1, y1)
        vx2, vy2 = jacobian_times_dq(x2, y2)
        vx3, vy3 = jacobian_times_dq(x3, y3)

        # Angular velocities
        w1 = dq1
        w2 = dq1 + dq2
        w3 = dq1 + dq2 + dq3

        # Kinetic energy
        T1 = 0.5 * m1 * (vx1**2 + vy1**2) + 0.5 * I1 * w1**2
        T2 = 0.5 * m2 * (vx2**2 + vy2**2) + 0.5 * I2 * w2**2
        T3 = 0.5 * m3 * (vx3**2 + vy3**2) + 0.5 * I3 * w3**2
        T = sp.simplify(T1 + T2 + T3)

        # Potential energy
        V = sp.simplify(m1 * g * y1 + m2 * g * y2 + m3 * g * y3)

        Lagr = T - V

        # Mass matrix
        M = sp.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                M[i, j] = sp.simplify(sp.diff(sp.diff(T, dq_vec[i]), dq_vec[j]))

        # Generalized torques tau(q,dq,ddq)
        tau_sym = []
        for i in range(3):
            dL_ddq_i = sp.diff(Lagr, dq_vec[i])

            time_derivative = 0
            for k in range(3):
                time_derivative += sp.diff(dL_ddq_i, q_vec[k]) * dq_vec[k]
                time_derivative += sp.diff(dL_ddq_i, dq_vec[k]) * ddq_vec[k]

            dL_dq_i = sp.diff(Lagr, q_vec[i])
            tau_i = sp.simplify(time_derivative - dL_dq_i)
            tau_sym.append(tau_i)

        tau_sym = sp.Matrix(tau_sym)

        # tau = M*ddq + h(q,dq)
        subs_ddq_zero = {ddq1: 0, ddq2: 0, ddq3: 0}
        h = sp.simplify(tau_sym.subs(subs_ddq_zero))

        # Gravity G(q) = h(q, dq=0)
        subs_dq_zero = {dq1: 0, dq2: 0, dq3: 0}
        G = sp.simplify(h.subs(subs_dq_zero))

        # Coriolis/Centrifugal term: C(q,dq)*dq = h - G
        Cqdot = sp.simplify(h - G)

        q_syms = (q1, q2, q3)
        dq_syms = (dq1, dq2, dq3)

        self._M_func = sp.lambdify(q_syms, M, "numpy")
        self._G_func = sp.lambdify(q_syms, G, "numpy")
        self._Cqdot_func = sp.lambdify(q_syms + dq_syms, Cqdot, "numpy")

    def mass_matrix(self, q):
        q = np.asarray(q, dtype=float).reshape(3,)
        return np.array(self._M_func(q[0], q[1], q[2]), dtype=float)

    def gravity(self, q):
        q = np.asarray(q, dtype=float).reshape(3,)
        return np.array(self._G_func(q[0], q[1], q[2]), dtype=float).reshape(3,)

    def coriolis_times_qdot(self, q, qdot):
        q = np.asarray(q, dtype=float).reshape(3,)
        qdot = np.asarray(qdot, dtype=float).reshape(3,)
        val = self._Cqdot_func(q[0], q[1], q[2], qdot[0], qdot[1], qdot[2])
        return np.array(val, dtype=float).reshape(3,)

    def clip_torque(self, tau):
        tau = np.asarray(tau, dtype=float).reshape(3,)
        return np.clip(tau, -self.torque_limits, self.torque_limits)

    def forward_dynamics(self, q, qdot, tau):
        q = np.asarray(q, dtype=float).reshape(3,)
        qdot = np.asarray(qdot, dtype=float).reshape(3,)
        tau = self.clip_torque(tau)

        M = self.mass_matrix(q)
        G = self.gravity(q)
        Cqdot = self.coriolis_times_qdot(q, qdot)

        rhs = tau - Cqdot - G
        qddot = np.linalg.solve(M, rhs)
        return qddot, M, Cqdot, G

    def step(self, q, qdot, tau, dt):
        """
        Semi-implicit Euler:
          qdot_{k+1} = qdot_k + dt*qddot
          q_{k+1}    = q_k + dt*qdot_{k+1}
        """
        qddot, M, Cqdot, G = self.forward_dynamics(q, qdot, tau)
        qdot_next = qdot + dt * qddot
        q_next = q + dt * qdot_next
        return q_next, qdot_next, qddot, M, Cqdot, G

    def simulate_one_scenario(self, tau, dt=0.001, T=1.0):
        """
        Scenario simulation:
          start q=dq=0
          apply constant tau for duration T
          return final q,dq,ddq plus M,G,Cqdot at the final state
        """
        tau = self.clip_torque(tau)
        dt = float(dt)
        T = float(T)
        n_steps = int(T / dt)

        q = np.zeros(3, dtype=float)
        qdot = np.zeros(3, dtype=float)

        qddot = np.zeros(3, dtype=float)
        M = np.eye(3, dtype=float)
        G = np.zeros(3, dtype=float)
        Cqdot = np.zeros(3, dtype=float)

        for _ in range(n_steps):
            q, qdot, qddot, M, Cqdot, G = self.step(q, qdot, tau, dt)

        # At the end, compute terms at final (q, qdot) for consistency
        M = self.mass_matrix(q)
        G = self.gravity(q)
        Cqdot = self.coriolis_times_qdot(q, qdot)

        return q, qdot, qddot, M, G, Cqdot


def parse_args():
    ap = argparse.ArgumentParser(description="Manual 3-link: scenario-based simulation (final-only) with PyBullet-like columns.")
    ap.add_argument("--input", "-i", required=True, help="Torque CSV (columns: tau1,tau2,tau3)")
    ap.add_argument("--output", "-o", required=True, help="Output CSV path")
    ap.add_argument("--dt", type=float, default=0.001, help="Timestep (default: 0.001)")
    ap.add_argument("--T", type=float, default=1.0, help="Duration per scenario in seconds (default: 1.0)")
    ap.add_argument("--progress_every", type=int, default=10, help="Print progress every N scenarios (default: 10)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tau_array = load_torque_csv(args.input)
    print("✅ Loaded torque scenarios:", tau_array.shape)

    env = ThreeLinkArmExact()

    rows = []
    for i, tau in enumerate(tau_array):
        q, dq, ddq, M, G, Cqdot = env.simulate_one_scenario(tau, dt=args.dt, T=args.T)

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

    rows = np.asarray(rows, dtype=float)
    save_results_csv(rows, args.output)

    print("\n✅ Done.")
    print("✅ Saved to:", Path(args.output).resolve())
