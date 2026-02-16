import numpy as np
import math
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


def run_mode(threat_scale):

    ego = np.array([0.0, 0.0, 0.0, 10.0])

    obstacles = {
        1: {"x": 14.0, "y": -3.0, "vx": 6.0, "vy": 1.5, "kind": "car"},
        2: {"x": 18.0, "y":  2.5, "vx": 5.0, "vy": 0.0, "kind": "car"},
        3: {"x": 22.0, "y": -2.5, "vx": 4.0, "vy": 0.0, "kind": "car"},
    }

    prio = Prioritizer("configs/threat.yaml")
    mpc = NMPCCasadi()

    risk_integral = 0.0
    clearance_log = []

    for step in range(STEPS):

        preds = {}
        scaled_threats = {}
        original_threats = {}

        min_clr = float("inf")

        for tid, obj in obstacles.items():

            dx = obj["x"] - ego[0]
            dy = obj["y"] - ego[1]
            dist = math.hypot(dx, dy)
            min_clr = min(min_clr, dist)

            track = {
                "x": obj["x"],
                "y": obj["y"],
                "vx": obj["vx"],
                "vy": obj["vy"],
                "yaw": 0.0
            }

            pred = predict_ctrv(track, mpc.N, DT)
            preds[tid] = pred

            # Compute original threat
            th, _ = prio.compute(
                tid,
                (ego[0], ego[1], ego[2], ego[3]),
                pred,
                kind=obj["kind"],
                class_conf=1.0,
                dt=DT
            )

            original_threats[tid] = th
            scaled_threats[tid] = threat_scale * th

        refs = np.array([
            [ego[0] + ego[3] * DT * (k + 1), 0.0, 0.0, ego[3]]
            for k in range(mpc.N)
        ])

        # Use scaled threats inside NMPC
        res = mpc.solve(ego, refs, preds, scaled_threats)
        u = res["u0"] if res.get("success", False) else (-1.0, 0.3)

        ego = step_ego(ego, u)

        for obj in obstacles.values():
            obj["x"] += obj["vx"] * DT
            obj["y"] += obj["vy"] * DT

        clearance_log.append(min_clr)

        # Risk computed using ORIGINAL threat (not scaled)
        step_risk = sum(original_threats[tid] * math.exp(-min_clr)
                        for tid in original_threats)
        risk_integral += step_risk * DT

    return risk_integral, clearance_log


def run():

    scales = [0.0, 0.5, 1.0, 2.0]

    results = {}

    print("\n===== Threat Strength Ablation (Corrected) =====")

    for s in scales:
        risk, clr = run_mode(s)
        results[s] = (risk, clr)
        print(f"Threat scale {s}: Risk integral = {risk:.3f}")

    # Plot clearance evolution
    plt.figure()
    for s in scales:
        plt.plot(results[s][1], label=f"Scale {s}")

    plt.xlabel("Time Step")
    plt.ylabel("Minimum Clearance (m)")
    plt.title("Threat Strength Ablation - Clearance")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    run()
