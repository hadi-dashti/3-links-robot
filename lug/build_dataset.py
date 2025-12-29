# build_dataset.py
import json
import numpy as np
from pathlib import Path
import config as cfg
from simulator import generate_trajectory, make_torque_profile

# -----------------------------
# کمک‌ها
# -----------------------------
def concat_trajectories(trajs):
    t   = np.concatenate([d["t"] for d in trajs], axis=0)
    q   = np.concatenate([d["q"] for d in trajs], axis=0)
    qd  = np.concatenate([d["qd"] for d in trajs], axis=0)
    qdd = np.concatenate([d["qdd"] for d in trajs], axis=0)
    tau = np.concatenate([d["tau"] for d in trajs], axis=0)
    traj_id = np.concatenate([np.full(d["t"].shape[0], i, dtype=int) for i, d in enumerate(trajs)], axis=0)
    return {"t": t, "q": q, "qd": qd, "qdd": qdd, "tau": tau, "traj_id": traj_id}

def split_by_trajectory(n_traj: int, train_ratio=0.7, val_ratio=0.15, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n_traj)
    rng.shuffle(idx)
    n_train = int(train_ratio * n_traj)
    n_val   = int(val_ratio   * n_traj)
    train_ids = idx[:n_train]
    val_ids   = idx[n_train:n_train+n_val]
    test_ids  = idx[n_train+n_val:]
    return train_ids, val_ids, test_ids

def filter_by_traj(data, keep_ids):
    mask = np.isin(data["traj_id"], keep_ids)
    out = {k: v[mask] if isinstance(v, np.ndarray) and v.shape[0]==mask.shape[0] else v
           for k, v in data.items()}
    return out

def save_npz(path: Path, data):
    np.savez_compressed(path, **data)

# -----------------------------
# بدنهٔ اصلی
# -----------------------------
def build_dataset(
    out_dir="dataset",
    n_traj=12,
    T=10.0,
    dt=0.01,
    base_seed=1000
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) ساخت تراژکتوری‌ها با پروفایل‌های گشتاور متفاوت
    trajs = []
    for i in range(n_traj):
        seed = base_seed + i*7
        torque_fn = make_torque_profile(seed)   # پروفایل مخصوص همین تراژکتوری
        d = generate_trajectory(T=T, dt=dt, torque_fn=torque_fn, seed=seed)
        # ذخیره هر تراژکتوری به صورت جدا (برای بازرسی)
        np.savez_compressed(out / f"traj_{i:03d}.npz", **d)
        trajs.append(d)

    # 2) چسباندن همهٔ تراژکتوری‌ها
    full = concat_trajectories(trajs)

    # 3) تقسیم بر اساس تراژکتوری (بدون نشت)
    train_ids, val_ids, test_ids = split_by_trajectory(n_traj, train_ratio=0.7, val_ratio=0.15, seed=base_seed)
    train = filter_by_traj(full, train_ids)
    val   = filter_by_traj(full, val_ids)
    test  = filter_by_traj(full, test_ids)

    # 4) ذخیره فایل‌های نهایی
    save_npz(out / "lnn_3link_train.npz", train)
    save_npz(out / "lnn_3link_val.npz",   val)
    save_npz(out / "lnn_3link_test.npz",  test)

    # 5) meta.json برای ثبت تنظیمات
    meta = {
        "n_traj": n_traj,
        "T_sec": T,
        "dt_sec": dt,
        "train_traj_ids": train_ids.tolist(),
        "val_traj_ids": val_ids.tolist(),
        "test_traj_ids": test_ids.tolist(),
        "units": {"q": "rad", "qd": "rad/s", "qdd": "rad/s^2", "tau": "N*m"},
        "robot_params": {
            "L_m":   cfg.L.tolist(),
            "M_kg":  cfg.M.tolist(),
            "I_kgm2": cfg.I.tolist(),
            "LC_m":  cfg.LC.tolist(),
            "g":     cfg.G_CONST,
            "friction_b": cfg.FRICTION_B.tolist(),
        },
        "limits": {
            "Q_LIMIT_rad":  float(cfg.Q_LIMIT),
            "QD_LIMIT_rads": float(cfg.QD_LIMIT),
            "TAU_CLIP_Nm":   cfg.TAU_CLIP.tolist(),
        },
        "notes": "Randomized torque per-trajectory with seed; simple M (diagonal), C=0, approx gravity."
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 6) چاپ خلاصه
    def shape_str(d): return " | ".join([f"{k}:{v.shape}" for k,v in d.items() if isinstance(v, np.ndarray)])
    print("[train]", shape_str(train))
    print("[val]  ", shape_str(val))
    print("[test] ", shape_str(test))
    print(f"Saved to folder: {out.resolve()}")

if __name__ == "__main__":
    # می‌تونی این اعدادو دلخواه عوض کنی یا از config بخونی
    build_dataset(out_dir="dataset", n_traj=12, T=10.0, dt=cfg.DT, base_seed=2025)
