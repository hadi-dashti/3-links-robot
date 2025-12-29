# inspect_dataset.py
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("dataset")
P_TRAIN = DATA_DIR / "lnn_3link_train.npz"
P_VAL   = DATA_DIR / "lnn_3link_val.npz"
P_TEST  = DATA_DIR / "lnn_3link_test.npz"
P_META  = DATA_DIR / "meta.json"

def load_npz(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path, allow_pickle=True)

def shape_str(d):
    # فقط آرایه‌ها را گزارش می‌کند
    keys = [k for k in d.files if isinstance(d[k], np.ndarray)]
    parts = []
    for k in keys:
        parts.append(f"{k}:{d[k].shape}")
    return " | ".join(parts)

def basic_stats(name, arr):
    arr = np.asarray(arr)
    return {
        "name": name,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std())
    }

def pct_near_clip(arr, clip_val, eps=1e-6, tol_frac=0.98):
    # چند درصد نمونه‌ها نزدیک کلیپ هستند (هشدار اگر زیاد بود)
    # نزدیک کلیپ یعنی |x| >= tol_frac * clip_val
    if clip_val is None: 
        return None
    arr = np.abs(np.asarray(arr))
    near = np.mean(arr >= (tol_frac * (clip_val - eps)))
    return float(near * 100.0)

def check_split_leakage(train_ids, val_ids, test_ids):
    # بررسی عدم-همپوشانی شناسه‌های تراژکتوری
    inter_tv = set(train_ids).intersection(set(val_ids))
    inter_tt = set(train_ids).intersection(set(test_ids))
    inter_vt = set(val_ids).intersection(set(test_ids))
    return len(inter_tv)==0 and len(inter_tt)==0 and len(inter_vt)==0, (inter_tv, inter_tt, inter_vt)

def preview_csv(npz_obj, out_csv: Path, rows=10):
    df = pd.DataFrame({
        "t":   npz_obj["t"][:rows],
        "q1":  npz_obj["q"][:rows, 0],
        "q2":  npz_obj["q"][:rows, 1],
        "q3":  npz_obj["q"][:rows, 2],
        "qd1": npz_obj["qd"][:rows, 0],
        "qd2": npz_obj["qd"][:rows, 1],
        "qd3": npz_obj["qd"][:rows, 2],
        "qdd1": npz_obj["qdd"][:rows, 0],
        "qdd2": npz_obj["qdd"][:rows, 1],
        "qdd3": npz_obj["qdd"][:rows, 2],
        "tau1": npz_obj["tau"][:rows, 0],
        "tau2": npz_obj["tau"][:rows, 1],
        "tau3": npz_obj["tau"][:rows, 2],
        "traj_id": npz_obj["traj_id"][:rows].astype(int) if "traj_id" in npz_obj.files else -1,
    })
    df.to_csv(out_csv, index=False)
    return out_csv

if __name__ == "__main__":
    # 1) لود فایل‌ها
    train = load_npz(P_TRAIN)
    val   = load_npz(P_VAL)
    test  = load_npz(P_TEST)

    print("[train]", shape_str(train))
    print("[val]  ", shape_str(val))
    print("[test] ", shape_str(test))

    # 2) خواندن meta و پارامترهای مهم
    meta = {}
    if P_META.exists():
        meta = json.loads(P_META.read_text(encoding="utf-8"))
        print("\nMeta loaded.")
        if "limits" in meta:
            tau_clip = meta["limits"].get("TAU_CLIP_Nm", [None, None, None])
        else:
            tau_clip = [None, None, None]
    else:
        print("\nWarning: meta.json not found.")
        tau_clip = [None, None, None]

    # 3) آمار پایه از train
    stats_list = []
    stats_list += [basic_stats("q1(rad)",  train["q"][:,0]),
                   basic_stats("q2(rad)",  train["q"][:,1]),
                   basic_stats("q3(rad)",  train["q"][:,2]),
                   basic_stats("qd1",      train["qd"][:,0]),
                   basic_stats("qd2",      train["qd"][:,1]),
                   basic_stats("qd3",      train["qd"][:,2]),
                   basic_stats("qdd1",     train["qdd"][:,0]),
                   basic_stats("qdd2",     train["qdd"][:,1]),
                   basic_stats("qdd3",     train["qdd"][:,2]),
                   basic_stats("tau1(Nm)", train["tau"][:,0]),
                   basic_stats("tau2(Nm)", train["tau"][:,1]),
                   basic_stats("tau3(Nm)", train["tau"][:,2])]

    print("\n[Basic stats on TRAIN]")
    for s in stats_list:
        print(f"{s['name']}: min={s['min']:.4f}, max={s['max']:.4f}, mean={s['mean']:.4f}, std={s['std']:.4f}")

    # 4) درصد نزدیکی به کلیپ گشتاور (اگر تعریف شده)
    if tau_clip and all([x is not None for x in tau_clip]):
        p1 = pct_near_clip(train["tau"][:,0], tau_clip[0])
        p2 = pct_near_clip(train["tau"][:,1], tau_clip[1])
        p3 = pct_near_clip(train["tau"][:,2], tau_clip[2])
        print("\n[% near torque clip on TRAIN]")
        print(f"tau1 near-clip: {p1:.2f}% | tau2: {p2:.2f}% | tau3: {p3:.2f}%")
        if any([p is not None and p > 10.0 for p in [p1,p2,p3]]):
            print("Note: A high near-clip percentage may indicate too-aggressive torque limits.")
    else:
        print("\nTorque clip not defined in meta.json (skipping near-clip check).")

    # 5) چک عدم نشت بین اسپلیت‌ها
    ok, inters = (True, (set(), set(), set()))
    if "train_traj_ids" in meta:
        ok, inters = check_split_leakage(meta["train_traj_ids"], meta["val_traj_ids"], meta["test_traj_ids"])
    print("\n[Split leakage check]")
    print("OK (no overlap):", ok, "| intersections:", [list(x) for x in inters])

    # 6) پیش‌نمایش CSV از train
    csv_path = DATA_DIR / "train_preview.csv"
    out = preview_csv(train, csv_path, rows=12)
    print(f"\nSaved preview CSV: {out.resolve()}")

    print("\nDone.")
