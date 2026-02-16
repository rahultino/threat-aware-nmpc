# sim/run_lateral_cutin_demo.py
import math
import numpy as np
import matplotlib.pyplot as plt

from predictor.ctrv import predict_ctrv
from threat.prioritizer import Prioritizer
from planner.mpc_casadi import NMPCCasadi
from sim.plot_utils import plot_2d_scene

DT = 0.1
STEPS = 100


def step_ego(state, u):
    x, y, psi, v = state
    a, delta = u
    x += v * math.cos(psi) * DT
    y += v * math.sin(psi) * DT
    psi += (v / 2.5) * math.tan(delta) * DT
    v = max(0.0, v + a * DT)
    return np.array([x, y, psi, v])


def run_single(use_threat):
    ego = np.array([0.0, 0.0, 0.0, 10.0])

    obstacles = {
        1: {"x": 14.0, "y": -3.2, "vx": 6.0, "vy": 1.8, "kind": "car"},
        2: {"x": 20.0, "y": 0.0,  "vx": 3.0, "vy": 0.0, "kind": "car"},
        3: {"x": 18.0, "y": 2.8,  "vx": 5.0, "vy": 0.0, "kind": "car"},
    }

    prio = Prioritizer("configs/threat.yaml")
    mpc = NMPCCasadi()

    ego_traj = []
    obs_traj = {tid: [] for tid in obstacles}
    threat_log = {tid: [] for tid in obstacles}

    y_log, steer_log, clr_log = [], [], []

    for step in range(STEPS):
        preds, threats = {}, {}
        ego_traj.append((ego[0], ego[1]))

        for tid, obj in obstacles.items():
            obs_traj[tid].append((obj["x"], obj["y"]))

            track = {
                "x": obj["x"],
                "y": obj["y"],
                "vx": obj["vx"],
                "vy": obj["vy"],
                "yaw": 0.0
            }

            pred = predict_ctrv(track, mpc.N, DT)
            preds[tid] = pred

            th, _ = prio.compute(
                tid,
                (ego[0], ego[1], ego[2], ego[3]),
                pred,
                kind=obj["kind"],
                class_conf=1.0,
                dt=DT
            )

            threats[tid] = th
            threat_log[tid].append(th)

        threats_used = threats if use_threat else {tid: 1.0 for tid in threats}

        refs = np.array([
            [ego[0] + ego[3] * DT * (k + 1), 0.0, 0.0, ego[3]]
            for k in range(mpc.N)
        ])

        res = mpc.solve(ego, refs, preds, threats_used)
        u = res["u0"] if res.get("success", False) else (-1.0, 0.3)

        ego = step_ego(ego, u)

        for obj in obstacles.values():
            obj["x"] += obj["vx"] * DT
            obj["y"] += obj["vy"] * DT

        y_log.append(ego[1])
        steer_log.append(u[1])

        min_clr = min(
            math.hypot(obj["x"] - ego[0], obj["y"] - ego[1])
            for obj in obstacles.values()
        )
        clr_log.append(min_clr)

    return ego_traj, obs_traj, y_log, steer_log, clr_log, threat_log


def run():
    base = run_single(use_threat=False)
    th = run_single(use_threat=True)

    t = np.arange(len(base[2])) * DT

    # --- Time plots ---
    plt.figure()
    plt.plot(t, base[2], "--", label="Baseline")
    plt.plot(t, th[2], "-", label="Threat-Aware")
    plt.xlabel("Time (s)")
    plt.ylabel("Lateral position (m)")
    plt.title("Lateral Position vs Time")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t, base[3], "--", label="Baseline")
    plt.plot(t, th[3], "-", label="Threat-Aware")
    plt.xlabel("Time (s)")
    plt.ylabel("Steering angle (rad)")
    plt.title("Steering vs Time")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t, base[4], "--", label="Baseline")
    plt.plot(t, th[4], "-", label="Threat-Aware")
    plt.xlabel("Time (s)")
    plt.ylabel("Clearance (m)")
    plt.title("Clearance vs Time")
    plt.grid()
    plt.legend()
    plt.show()

    # --- Threat Evolution Plot ---
    plt.figure()
    for tid in th[5]:
        plt.plot(t, th[5][tid], label=f"Threat Obstacle {tid}")
    plt.xlabel("Time (s)")
    plt.ylabel("Threat Score")
    plt.title("Threat Evolution Over Time (Threat-Aware)")
    plt.grid()
    plt.legend()
    plt.show()

    # --- 2D scene ---
    plot_2d_scene(base[0], base[1], "2D — Baseline NMPC")
    plot_2d_scene(th[0], th[1], "2D — Threat-Aware NMPC")


if __name__ == "__main__":
    run()
