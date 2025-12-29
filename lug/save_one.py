# save_one.py
import numpy as np
from simulator import generate_trajectory

if __name__ == "__main__":
    data = generate_trajectory(T=5.0, dt=0.01)
    np.savez_compressed("traj_test.npz", **data)
    print("Saved traj_test.npz with", data["t"].shape[0], "samples")
