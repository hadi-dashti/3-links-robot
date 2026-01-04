# config.py
import numpy as np

# ----- Robot geometry & physical params -----
L = np.array([1, 1,1], dtype=float)  # link lengths [m]
M = np.array([1.5, 1.5, 1.5], dtype=float)     # masses [kg]
LC = L / 2.0                                   # COM at half-length
I = M * (L**2) / 12.0                          # rod inertia about COM
G_CONST = 9.81

# ----- Friction (viscous) -----
FRICTION_B = np.array([0.05, 0.05, 0.05], dtype=float)

# ----- Safe limits -----
Q_LIMIT = np.pi               # joint angle soft limit (±)
QD_LIMIT = 6.0                # joint speed soft limit (±)
TAU_CLIP = np.array([10.0, 9.0, 8.0])  # max |torque| per joint

# ----- Simulation -----
DT = 0.01     # time step [s]
T_TRAJ = 5.0  # trajectory duration for quick tests [s]

# You can change any number above freely; all code reads from here.
