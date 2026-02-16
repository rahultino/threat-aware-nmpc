# sim/run_synthetic_conflict_demo.py
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
    v = max(0.0, v + a * DT)
    return np.array([x, y, psi, v])


def run_single(use_threat):
    ego = np.array([0.0, 0.0, 0.0, 10.0])

    obstacles = {
    # Low-threat obstacle (off-center)
        1: {
            "x": 14.0,
            "y": 0.8,     # lateral offset
            "vx": 1.0,
            "vy": 0.0,
            "kind": "car"
        },

        # High-threat obstacle (directly in ego path)
        2: {
            "x": 16.0,
            "y": 0.0,     # CENTERLINE → forces decision
            "vx": 6.0,
            "vy": 0.0,
            "kind": "car"
        }
    }


    prio = Prioritizer("configs/threat.yaml")
    mpc = NMPCCasadi()

    acc_log = []
    ttc_log = []

    for _ in range(STEPS):
        preds = {}
        threats = {}

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
            refs.append([ego[0] + ego[3]*DT*(k+1), 0.0, 0.0, ego[3]])
        refs = np.array(refs)

        res = mpc.solve(ego, refs, preds, threats_used)
        u = res["u0"] if res.get("success", False) else (-3.0, 0.0)

        ego = step_ego(ego, u)

        for obj in obstacles.values():
            obj["x"] += obj["vx"] * DT

        acc_log.append(u[0])

        ttc_vals = []
        for obj in obstacles.values():
            rel_d = math.hypot(obj["x"] - ego[0], obj["y"] - ego[1])
            rel_v = ego[3] - obj["vx"]
            if rel_v > 0:
                ttc_vals.append(rel_d / rel_v)

        ttc_log.append(min(ttc_vals) if ttc_vals else 100.0)

    return acc_log, ttc_log


def run():
    acc_base, ttc_base = run_single(use_threat=False)
    acc_th, ttc_th = run_single(use_threat=True)

    t = np.arange(len(acc_base)) * DT

    plt.figure()
    plt.plot(t, acc_base, "--", label="Baseline")
    plt.plot(t, acc_th, "-", label="Threat-Aware")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Acceleration vs Time (Synthetic Conflict)")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t, ttc_base, "--", label="Baseline")
    plt.plot(t, ttc_th, "-", label="Threat-Aware")
    plt.xlabel("Time (s)")
    plt.ylabel("TTC (s)")
    plt.title("Minimum TTC vs Time (Synthetic Conflict)")
    plt.grid()
    plt.legend()
    plt.show()
    print("\n===== TERMINAL METRICS =====")

    print("Baseline NMPC:")
    print(f"  Min TTC: {min(ttc_base):.2f} s")
    print(f"  Max Decel: {min(acc_base):.2f} m/s²")
    print(f"  Avg |Acc|: {np.mean(np.abs(acc_base)):.2f} m/s²")

    print("\nThreat-Aware NMPC:")
    print(f"  Min TTC: {min(ttc_th):.2f} s")
    print(f"  Max Decel: {min(acc_th):.2f} m/s²")
    print(f"  Avg |Acc|: {np.mean(np.abs(acc_th)):.2f} m/s²")



if __name__ == "__main__":
    run()
