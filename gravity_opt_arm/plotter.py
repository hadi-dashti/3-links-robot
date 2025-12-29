import os
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _joints_from_theta(theta, L):
    

    t1, t2, t3 = theta
    L1, L2, L3 = L
    x0, y0 = 0.0, 0.0
    x1, y1 = L1*math.cos(t1), L1*math.sin(t1)
    x2, y2 = x1 + L2*math.cos(t1+t2), y1 + L2*math.sin(t1+t2)
    x3, y3 = x2 + L3*math.cos(t1+t2+t3), y2 + L3*math.sin(t1+t2+t3)
    return (x0, y0), (x1, y1), (x2, y2), (x3, y3)


def plot_and_save(theta, target_xy, L, save_path,
                  title="Predicted Arm Configuration vs Target"):
    base, j1, j2, eef = _joints_from_theta(theta, L)

    
    R = sum(L)
    xmin = ymin = -R - 0.2
    xmax = ymax =  R + 0.2

    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="both", alpha=0.35)

   
    circ = plt.Circle((0, 0), R, fill=False, ls="--", lw=1.0, color="gray", alpha=0.6)
    ax.add_patch(circ)

    
    xs = [base[0], j1[0], j2[0], eef[0]]
    ys = [base[1], j1[1], j2[1], eef[1]]
    ax.plot(xs, ys, "r--", lw=2, marker="o", ms=6, label="Predicted Arm")

    
    ax.plot([target_xy[0]], [target_xy[1]], marker="*", ms=12, mfc="none",
            mec="green", color="green", label="Target (x,y)")

    ax.legend(loc="upper right")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
