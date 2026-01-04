import os
import time

from optimize import search_min_energy_to_point
from config import L, M, g, target, phi_fixed, phi_step_deg, refine_deg, do_refine, output_dir
from plotter import plot_and_save


def main():
    res = search_min_energy_to_point(
        target_xy=target,
        L=L,
        M=M,
        g=g,
        phi_fixed=phi_fixed,
        phi_step_deg=phi_step_deg,
        refine_deg=refine_deg,
        do_refine=do_refine,
        joint_limits=None
    )

    if not res.get("ok"):
        print("❌ Unreachable:", res.get("reason", "unknown"))
        return

    theta = res["theta_rad"]
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_file = os.path.join(output_dir, f"arm_plot_{stamp}.png")

    plot_and_save(theta, target, L, out_file)

    print("✅ Saved plot to:", out_file)
    if "theta_deg" in res: print("θ (deg):", res["theta_deg"])
    if "phi_deg"   in res: print("φ (deg):", res["phi_deg"])
    if "branch"    in res: print("branch:", res["branch"])
    if "V_J"       in res: print("Potential Energy (J):", res["V_J"])
    if "fk_error_m" in res: print("FK error (m):", res["fk_error_m"])


if __name__ == "__main__":
    main()
