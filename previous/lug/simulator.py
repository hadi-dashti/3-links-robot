# simulator.py
import math
import numpy as np
import config as cfg

def M_matrix(q):
    # Simple diagonal M to bootstrap; later replace with the exact M(q)
    return np.diag(cfg.I + cfg.M * (cfg.LC**2) + 1e-3)

def C_matrix(q, qd):
    # Minimal model for now
    return np.zeros((3, 3))

def G_vector(q):
    # Approx gravity for planar chain
    c0 = math.cos(q[0])
    c01 = math.cos(q[0] + q[1])
    c012 = math.cos(q[0] + q[1] + q[2])
    G0 = cfg.G_CONST * (cfg.M[0]*cfg.LC[0]*c0
         + cfg.M[1]*(cfg.L[0]*c0 + cfg.LC[1]*c01)
         + cfg.M[2]*(cfg.L[0]*c0 + cfg.L[1]*c01 + cfg.LC[2]*c012))
    G1 = cfg.G_CONST * (cfg.M[1]*cfg.LC[1]*c01 + cfg.M[2]*(cfg.L[1]*c01 + cfg.LC[2]*c012))
    G2 = cfg.G_CONST * (cfg.M[2]*cfg.LC[2]*c012)
    return np.array([G0, G1, G2], dtype=float)

def friction(qd):
    return cfg.FRICTION_B * qd

def dynamics(q, qd, tau):
    M = M_matrix(q)
    C = C_matrix(q, qd)
    G = G_vector(q)
    rhs = tau - C @ qd - G - friction(qd)
    qdd = np.linalg.solve(M, rhs)
    return qdd

def rk4_step(q, qd, tau, dt):
    def f(state, tau):
        q = state[:3]; qd = state[3:]
        qdd = dynamics(q, qd, tau)
        return np.hstack([qd, qdd])

    s = np.hstack([q, qd])
    k1 = f(s, tau)
    k2 = f(s + 0.5*dt*k1, tau)
    k3 = f(s + 0.5*dt*k2, tau)
    k4 = f(s + dt*k3, tau)
    s_next = s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    q_next, qd_next = s_next[:3], s_next[3:]
    # Soft clamp to keep numbers sane
    q_next = np.clip(q_next, -cfg.Q_LIMIT, cfg.Q_LIMIT)
    qd_next = np.clip(qd_next, -cfg.QD_LIMIT, cfg.QD_LIMIT)
    return q_next, qd_next

def simple_torque_profile(t):
    # Very basic sine torques to see motion
    tau = np.array([
        3.0*math.sin(0.7*t),
        2.5*math.sin(1.1*t + 0.3),
        2.0*math.sin(0.9*t + 0.6),
    ])
    # clip for safety
    return np.clip(tau, -cfg.TAU_CLIP, cfg.TAU_CLIP)

def make_torque_profile(seed: int = 0):
    rng = np.random.default_rng(seed)
    # دامنه‌ها، فرکانس‌ها و فازهای تصادفی (ملایم و ایمن)
    A = rng.uniform(2.0, 6.0, size=3)           # amplitudes [N*m]
    f = rng.uniform(0.3, 1.2, size=3)           # frequencies [Hz]
    ph = rng.uniform(0.0, 2*np.pi, size=3)      # phases [rad]
    # یک مؤلفه PRBS نرم (تقریب مربع موج) برای تنوع
    prbs_amp  = rng.uniform(0.5, 3.0, size=3)
    prbs_rate = rng.uniform(0.2, 0.8)           # switches per second

    def tau_fn(t: float):
        base = A * np.sin(2*np.pi*f*t + ph)
        prbs = np.sign(np.sin(2*np.pi*prbs_rate*t + 0.5)) * prbs_amp
        tau = base + 0.4 * prbs
        # کلیپ ایمنی با مقادیر config
        return np.clip(tau, -cfg.TAU_CLIP, cfg.TAU_CLIP)

    return tau_fn


def generate_trajectory(T=None, dt=None, q0=None, qd0=None, torque_fn=None, seed: int = 0):
    if T is None:  T = cfg.T_TRAJ
    if dt is None: dt = cfg.DT

    steps = int(T / dt)
    rng = np.random.default_rng(seed)
    q  = rng.uniform(-0.5, 0.5, size=3) if q0  is None else np.array(q0,  dtype=float)
    qd = rng.uniform(-0.2, 0.2, size=3) if qd0 is None else np.array(qd0, dtype=float)

    # اگر torque_fn ندادیم، یکی با seed بساز
    if torque_fn is None:
        torque_fn = make_torque_profile(seed + 12345)

    t_list, q_list, qd_list, qdd_list, tau_list = [], [], [], [], []
    for k in range(steps):
        t = k * dt
        tau = torque_fn(t)
        qdd = dynamics(q, qd, tau)

        t_list.append(t); q_list.append(q.copy()); qd_list.append(qd.copy())
        qdd_list.append(qdd.copy()); tau_list.append(tau.copy())

        q, qd = rk4_step(q, qd, tau, dt)

    data = {
        "t":   np.array(t_list),
        "q":   np.vstack(q_list),
        "qd":  np.vstack(qd_list),
        "qdd": np.vstack(qdd_list),
        "tau": np.vstack(tau_list),
    }
    return data

if __name__ == "__main__":
    # در VS Code همین فایل را باز نگه دار و Play (Run Python File) را بزن
    d = generate_trajectory(T=3.0, dt=0.01)
    print("Samples:", d["t"].shape[0], "| q shape:", d["q"].shape)
