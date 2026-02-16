# sim/run_lane_constrained_demo.py
import math
import numpy as np
import matplotlib.pyplot as plt

from predictor.ctrv import predict_ctrv
from threat.prioritizer import Prioritizer
from planner.mpc_casadi import NMPCCasadi

DT = 0.1
STEPS = 80


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

    # Obstacle blocks lane center
    obstacles = {
        1: {
            "x": 16.0,
            "y": 0.0,
            "vx": 4.0,
            "vy": 0.0,
            "kind": "car"
        }
    }

    prio = Prioritizer("configs/threat.yaml")
    mpc = NMPCCasadi()

    y_log, steer_log, clr_log = [], [], []
    reaction_time = None
    integrated_clearance = 0.0

    for step in range(STEPS):
        preds, threats = {}, {}

        for tid, obj in obstacles.items():
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

        threats_used = threats if use_threat else {tid: 1.0 for tid in threats}

        refs = []
        for k in range(mpc.N):
            refs.append([ego[0] + ego[3] * DT * (k + 1), 0.0, 0.0, ego[3]])
        refs = np.array(refs)

        res = mpc.solve(ego, refs, preds, threats_used)
        u = res["u0"] if res.get("success", False) else (-1.0, 0.2)

        ego = step_ego(ego, u)
        obstacles[1]["x"] += obstacles[1]["vx"] * DT

        y_log.append(ego[1])
        steer_log.append(u[1])

        d = math.hypot(obstacles[1]["x"] - ego[0], obstacles[1]["y"] - ego[1])
        clr_log.append(d)
        integrated_clearance += d * DT

        if reaction_time is None and abs(u[1]) > 0.02:
            reaction_time = step * DT

    return y_log, steer_log, clr_log, reaction_time, integrated_clearance


def run():
    base = run_single(use_threat=False)
    th = run_single(use_threat=True)

    t = np.arange(len(base[0])) * DT

    plt.figure()
    plt.plot(t, base[0], "--", label="Baseline")
    plt.plot(t, th[0], "-", label="Threat-Aware")
    plt.xlabel("Time (s)")
    plt.ylabel("Lateral position y (m)")
    plt.title("Lane-Constrained Lateral Deviation")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t, base[2], "--", label="Baseline")
    plt.plot(t, th[2], "-", label="Threat-Aware")
    plt.xlabel("Time (s)")
    plt.ylabel("Clearance (m)")
    plt.title("Lane-Constrained Minimum Clearance")
    plt.grid()
    plt.legend()
    plt.show()

    print("\n===== LANE-CONSTRAINED METRICS =====")
    print(f"Baseline reaction time: {base[3]}")
    print(f"Threat-aware reaction time: {th[3]}")
    print(f"Baseline integrated clearance: {base[4]:.2f}")
    print(f"Threat-aware integrated clearance: {th[4]:.2f}")


if __name__ == "__main__":
    run()
