# sim/run_lateral_conflict_demo.py
import math
import numpy as np
import matplotlib.pyplot as plt

from predictor.ctrv import predict_ctrv
from threat.prioritizer import Prioritizer
from planner.mpc_casadi import NMPCCasadi

def fmt_time(t):
    return f"{t:.2f} s" if t is not None else "No reaction"


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

    # -------- LATERAL CONFLICT SCENARIO --------
    obstacles = {
        # Low-threat obstacle (off centerline)
        1: {
            "x": 15.0,
            "y": 1.2,
            "vx": 2.0,
            "vy": 0.0,
            "kind": "car"
        },

        # High-threat obstacle (centerline)
        2: {
            "x": 18.0,
            "y": 0.0,
            "vx": 6.0,
            "vy": 0.0,
            "kind": "car"
        }
    }

    prio = Prioritizer("configs/threat.yaml")
    mpc = NMPCCasadi()

    # -------- LOGS --------
    y_log = []
    steer_log = []
    clr_log = []

    reaction_time = None
    integrated_clearance = 0.0
    steer_jerk = 0.0
    prev_steer = 0.0
    safe_clearance_time = 0.0
    clearance_at_critical = None

    SAFE_CLR = 1.5
    CRITICAL_T = 1.5  # seconds

    for step in range(STEPS):
        preds = {}
        threats = {}

        # Predict obstacles and compute threat
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

        # Straight reference
        refs = []
        for k in range(mpc.N):
            refs.append([ego[0] + ego[3] * DT * (k + 1), 0.0, 0.0, ego[3]])
        refs = np.array(refs)

        res = mpc.solve(ego, refs, preds, threats_used)
        u = res["u0"] if res.get("success", False) else (-1.0, 0.0)

        ego = step_ego(ego, u)

        # Move obstacles
        for obj in obstacles.values():
            obj["x"] += obj["vx"] * DT

        # -------- METRICS --------
        y_log.append(ego[1])
        steer_log.append(u[1])

        # Minimum clearance
        dists = []
        for obj in obstacles.values():
            d = math.hypot(obj["x"] - ego[0], obj["y"] - ego[1])
            dists.append(d)
        min_clr = min(dists)
        clr_log.append(min_clr)

        # Reaction time
        if reaction_time is None and abs(u[1]) > 0.02:
            reaction_time = step * DT

        # Integrated clearance
        integrated_clearance += min_clr * DT

        # Steering jerk
        steer_jerk += abs(u[1] - prev_steer)
        prev_steer = u[1]

        # Time above safe clearance
        if min_clr > SAFE_CLR:
            safe_clearance_time += DT

        # Clearance at critical time
        if clearance_at_critical is None and step * DT >= CRITICAL_T:
            clearance_at_critical = min_clr

    return {
        "y": y_log,
        "steer": steer_log,
        "clearance": clr_log,
        "reaction_time": reaction_time,
        "int_clearance": integrated_clearance,
        "steer_jerk": steer_jerk,
        "safe_time": safe_clearance_time,
        "critical_clearance": clearance_at_critical
    }


def run():
    print("Running BASELINE NMPC (lateral conflict)")
    base = run_single(use_threat=False)

    print("Running THREAT-AWARE NMPC (lateral conflict)")
    th = run_single(use_threat=True)

    t = np.arange(len(base["y"])) * DT

    # -------- PLOTS --------
    plt.figure()
    plt.plot(t, base["y"], "--", label="Baseline")
    plt.plot(t, th["y"], "-", label="Threat-Aware")
    plt.xlabel("Time (s)")
    plt.ylabel("Lateral position y (m)")
    plt.title("Lateral Deviation vs Time")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t, base["steer"], "--", label="Baseline")
    plt.plot(t, th["steer"], "-", label="Threat-Aware")
    plt.xlabel("Time (s)")
    plt.ylabel("Steering angle (rad)")
    plt.title("Steering vs Time")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t, base["clearance"], "--", label="Baseline")
    plt.plot(t, th["clearance"], "-", label="Threat-Aware")
    plt.xlabel("Time (s)")
    plt.ylabel("Minimum clearance (m)")
    plt.title("Minimum Clearance vs Time")
    plt.grid()
    plt.legend()
    plt.show()

    # -------- TERMINAL METRICS --------
    print("\n===== ADVANCED SAFETY METRICS =====")

    print("\nBaseline NMPC:")
    print(f"  Reaction time: {fmt_time(base['reaction_time'])}")

    print(f"  Integrated clearance: {base['int_clearance']:.2f} m·s")
    print(f"  Clearance @1.5s: {base['critical_clearance']:.2f} m")
    print(f"  Avg |steer|: {np.mean(np.abs(base['steer'])):.3f} rad")
    print(f"  Steering jerk: {base['steer_jerk']:.3f}")
    print(f"  Time >1.5m clearance: {base['safe_time']:.2f} s")

    print("\nThreat-Aware NMPC:")
    print(f"  Reaction time: {fmt_time(th['reaction_time'])}")

    print(f"  Integrated clearance: {th['int_clearance']:.2f} m·s")
    print(f"  Clearance @1.5s: {th['critical_clearance']:.2f} m")
    print(f"  Avg |steer|: {np.mean(np.abs(th['steer'])):.3f} rad")
    print(f"  Steering jerk: {th['steer_jerk']:.3f}")
    print(f"  Time >1.5m clearance: {th['safe_time']:.2f} s")


if __name__ == "__main__":
    run()
