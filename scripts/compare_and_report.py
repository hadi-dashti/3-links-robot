import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLS = [
    "tau1","tau2","tau3",
    "q1","q2","q3",
    "dq1","dq2","dq3",
    "ddq1","ddq2","ddq3",
    "G1","G2","G3",
    "Cqdot1","Cqdot2","Cqdot3",
    "M1","M2","M3","M4","M5","M6","M7","M8","M9",
]

GROUPS = {
    "tau": ["tau1","tau2","tau3"],
    "q": ["q1","q2","q3"],
    "dq": ["dq1","dq2","dq3"],
    "ddq": ["ddq1","ddq2","ddq3"],
    "G": ["G1","G2","G3"],
    "Cqdot": ["Cqdot1","Cqdot2","Cqdot3"],
}
M_COLS = [f"M{i}" for i in range(1, 10)]


# ------------------------------
# Metrics
# ------------------------------
def mae(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.mean(np.abs(d)))

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(d**2)))

def rel_mae(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = max(eps, float(np.mean(np.abs(a))))  # relative to mean |manual|
    return mae(a, b) / denom

def fmt(x: float) -> str:
    return f"{x:.3e}"


# ------------------------------
# Diagnostics (heuristics)
# ------------------------------
def detect_joint_limit_hits(q_py: np.ndarray, limit: float = np.pi, margin: float = 0.02):
    """
    Heuristic: If |q| is frequently within margin of ±pi, PyBullet is likely hitting URDF joint limits.
    Returns: (any_hits, per_joint_hit_rate)
    """
    near = np.abs(np.abs(q_py) - limit) < margin
    hit_rate = np.mean(near, axis=0)  # per joint
    any_hits = np.any(hit_rate > 0.05)  # >5% of scenarios
    return any_hits, hit_rate

def detect_unbounded_manual(q_manual: np.ndarray, threshold: float = 2*np.pi):
    """
    Heuristic: If manual angles often exceed ±2π, it likely has no joint limits/wrapping.
    Returns per-joint rate.
    """
    return np.mean(np.abs(q_manual) > threshold, axis=0)

def model_based_ddq(M_flat: np.ndarray, tau: np.ndarray, Cqdot: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    ddq = inv(M) * (tau - Cqdot - G), computed per scenario.
    """
    n = M_flat.shape[0]
    out = np.zeros((n, 3), dtype=float)
    for i in range(n):
        M = M_flat[i].reshape(3, 3)
        rhs = tau[i] - Cqdot[i] - G[i]
        out[i] = np.linalg.solve(M, rhs)
    return out


# ------------------------------
# Report builder
# ------------------------------
def build_markdown_report(
    df_m: pd.DataFrame,
    df_p: pd.DataFrame,
    dt: float,
    T: float,
    joint_limit_margin: float,
    out_md: Path,
):
    n = len(df_m)

    # arrays
    tau_m = df_m[GROUPS["tau"]].to_numpy(dtype=float)
    tau_p = df_p[GROUPS["tau"]].to_numpy(dtype=float)

    q_m = df_m[GROUPS["q"]].to_numpy(dtype=float)
    q_p = df_p[GROUPS["q"]].to_numpy(dtype=float)

    dq_m = df_m[GROUPS["dq"]].to_numpy(dtype=float)
    dq_p = df_p[GROUPS["dq"]].to_numpy(dtype=float)

    ddq_m = df_m[GROUPS["ddq"]].to_numpy(dtype=float)
    ddq_p = df_p[GROUPS["ddq"]].to_numpy(dtype=float)

    G_m = df_m[GROUPS["G"]].to_numpy(dtype=float)
    G_p = df_p[GROUPS["G"]].to_numpy(dtype=float)

    C_m = df_m[GROUPS["Cqdot"]].to_numpy(dtype=float)
    C_p = df_p[GROUPS["Cqdot"]].to_numpy(dtype=float)

    M_m = df_m[M_COLS].to_numpy(dtype=float)
    M_p = df_p[M_COLS].to_numpy(dtype=float)

    # 1) tau consistency
    tau_mae = mae(tau_m, tau_p)

    # 2) numeric summary table
    summary = []
    for k in ["q", "dq", "ddq", "G", "Cqdot"]:
        a = df_m[GROUPS[k]].to_numpy(dtype=float)
        b = df_p[GROUPS[k]].to_numpy(dtype=float)
        summary.append((k, mae(a, b), rmse(a, b), rel_mae(a, b)))

    summary.append(("M(flat)", mae(M_m, M_p), rmse(M_m, M_p), rel_mae(M_m, M_p)))

    # 3) diagnostics
    limit_hits, hit_rate = detect_joint_limit_hits(q_p, limit=np.pi, margin=joint_limit_margin)
    unbounded_rate = detect_unbounded_manual(q_m, threshold=2*np.pi)

    # ddq consistency diagnostic for PyBullet
    ddq_mb_p = model_based_ddq(M_p, tau_p, C_p, G_p)
    ddq_py_consistency = mae(ddq_p, ddq_mb_p)

    # key interpretation:
    # If q differs a lot, large M mismatch is expected because M depends on q.
    q_diff_rel = rel_mae(q_m, q_p)

    # Decide dominant cause based on heuristics
    dominant_causes = []
    if limit_hits:
        dominant_causes.append("PyBullet joint-limit saturation (URDF limits near ±pi).")
    if np.any(unbounded_rate > 0.01):
        dominant_causes.append("Manual model uses unbounded joint angles (no joint limits / wrapping).")
    if ddq_py_consistency > 1e-2:
        dominant_causes.append("PyBullet ddq is estimated by finite-difference; model-based ddq can differ noticeably.")
    if q_diff_rel > 0.2:
        dominant_causes.append("Final state mismatch (q,dq) amplifies differences in M(q), G(q), and C(q,dq)*dq.")

    if not dominant_causes:
        dominant_causes.append("No single dominant issue detected by heuristics; likely combined solver/inertia/frame differences.")

    # 4) produce markdown
    lines = []
    add = lines.append

    add("# Validation Report: Manual vs PyBullet (Scenario-Based, Final-Only)")
    add("")
    add("This report compares a symbolic manual dynamics simulator against PyBullet for independent torque scenarios. "
        "Each scenario starts from q0=[0,0,0], dq0=[0,0,0], applies a constant torque vector for duration T, "
        "and records the final state and dynamic terms.")
    add("")
    add(f"- Scenarios: **{n}**")
    add(f"- dt: **{dt} s**")
    add(f"- T: **{T} s**")
    add("")

    add("## 1) Input consistency")
    add("")
    add(f"- Torque MAE (manual vs PyBullet): **{fmt(tau_mae)}**")
    if tau_mae < 1e-9:
        add("- ✅ Inputs are identical; differences are due to modeling/solver differences rather than different torques.")
    else:
        add("- ⚠️ Inputs are not identical; check torque clipping, scaling, or CSV column mapping.")
    add("")

    add("## 2) Error metrics summary")
    add("")
    add("| Quantity | MAE | RMSE | Rel. MAE (vs mean |manual|) |")
    add("|---|---:|---:|---:|")
    for name, m1, m2, m3 in summary:
        add(f"| {name} | {fmt(m1)} | {fmt(m2)} | {fmt(m3)} |")
    add("")

    add("## 3) Diagnostics (root-cause analysis)")
    add("")
    add("### A) Joint limits / bounded angles")
    add("")
    add("- PyBullet may clamp angles if URDF joint limits are set (commonly [-pi, +pi]).")
    add("- Manual models typically allow unbounded angles unless limits are explicitly implemented.")
    add("")
    add(f"- PyBullet near-±pi hit rate (per joint): **{hit_rate}**")
    add(f"- Manual |q| > 2π rate (per joint): **{unbounded_rate}**")
    if limit_hits:
        add("- ✅ Evidence suggests PyBullet is frequently reaching joint limits; this changes the final state and all dependent terms.")
    add("")

    add("### B) Acceleration (ddq) definition")
    add("")
    add("- Manual ddq is obtained directly from the dynamic equation.")
    add("- In this pipeline, PyBullet ddq is estimated using finite differences; this can deviate from model-based ddq.")
    add("")
    add(f"- PyBullet ddq consistency (reported vs model-based): MAE = **{fmt(ddq_py_consistency)}**")
    add("")

    add("### C) Solver / integration differences")
    add("")
    add("- Manual uses a simple explicit integrator (semi-implicit Euler).")
    add("- PyBullet uses an engine solver with iterative updates and internal stabilization.")
    add("- Even with identical parameters, different numerical methods yield different trajectories and final states.")
    add("")

    add("### D) Inertial / frame conventions mismatch")
    add("")
    add("- Manual dynamics are derived from explicit (L, m, I, COM) assumptions.")
    add("- PyBullet dynamics are determined by URDF mass, inertia tensor, COM offsets, and link/joint frames.")
    add("- Small mismatches in URDF inertial parameters or gravity axis conventions can produce meaningful differences.")
    add("")

    add("### Summary of dominant causes detected")
    add("")
    for c in dominant_causes:
        add(f"- **{c}**")
    add("")

    add("## 4) Decision: adopt PyBullet as the main simulation reference")
    add("")
    add("For future development and reporting, we adopt **PyBullet** as the primary simulation reference because:")
    add("- It enforces physically meaningful constraints (e.g., joint limits).")
    add("- It is widely used in robotics research and improves reproducibility for others.")
    add("- It scales better to later extensions (controllers, contacts, sensors, visualization).")
    add("")
    add("The manual symbolic model will be kept as a **secondary baseline** for sanity checks and debugging "
        "(e.g., verifying trends, validating dynamic term extraction, and unit testing).")
    add("")

    add("## 5) Recommendations to reduce mismatch (if desired)")
    add("")
    add("1) **Align joint limits**: either increase/remove URDF limits or implement the same limits in the manual model.")
    add("2) **Align inertial parameters**: match URDF COM and inertia tensor to the manual assumptions exactly.")
    add("3) **Align gravity/frame conventions**: confirm axis directions match across both implementations.")
    add("4) **Standardize ddq**: compare ddq using the model-based definition `ddq = inv(M)*(tau - Cqdot - G)`.")
    add("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    # concise console summary
    console = []
    console.append("=== Manual vs PyBullet: Validation Summary ===")
    console.append(f"Scenarios={n}, dt={dt}, T={T}")
    console.append(f"Torque MAE: {fmt(tau_mae)}")
    for name, m1, m2, m3 in summary:
        console.append(f"{name:7s}  MAE={fmt(m1)}  RMSE={fmt(m2)}  rel={fmt(m3)}")
    console.append("")
    console.append("Dominant causes detected:")
    for c in dominant_causes:
        console.append(f"- {c}")
    console.append("")
    console.append(f"Report written to: {out_md.resolve()}")
    return "\n".join(console)


def parse_args():
    ap = argparse.ArgumentParser(description="Generate a professional validation report (Markdown) for Manual vs PyBullet.")
    ap.add_argument("--manual", required=True, help="Manual CSV (full columns)")
    ap.add_argument("--pybullet", required=True, help="PyBullet CSV (full columns)")
    ap.add_argument("--dt", type=float, default=0.001, help="dt used (for metadata)")
    ap.add_argument("--T", type=float, default=1.0, help="T used (for metadata)")
    ap.add_argument("--out_md", default="docs/validation_report.md", help="Output markdown path")
    ap.add_argument("--joint_limit_margin", type=float, default=0.02, help="Margin around ±pi for joint-limit detection")
    return ap.parse_args()


def main():
    args = parse_args()

    df_m = pd.read_csv(args.manual)
    df_p = pd.read_csv(args.pybullet)

    # Validate columns
    missing_m = [c for c in EXPECTED_COLS if c not in df_m.columns]
    missing_p = [c for c in EXPECTED_COLS if c not in df_p.columns]
    if missing_m:
        raise ValueError(f"Manual file missing columns: {missing_m}")
    if missing_p:
        raise ValueError(f"PyBullet file missing columns: {missing_p}")

    # Validate rows
    if len(df_m) != len(df_p):
        raise ValueError(f"Row mismatch: manual={len(df_m)} pybullet={len(df_p)}")

    out_md = Path(args.out_md)
    console_text = build_markdown_report(
        df_m=df_m,
        df_p=df_p,
        dt=args.dt,
        T=args.T,
        joint_limit_margin=args.joint_limit_margin,
        out_md=out_md,
    )
    print(console_text)


if __name__ == "__main__":
    main()
