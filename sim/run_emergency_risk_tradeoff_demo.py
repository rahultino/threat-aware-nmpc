# sim/run_emergency_risk_tradeoff_demo.py
import math
import numpy as np
import matplotlib.pyplot as plt

from predictor.ctrv import predict_ctrv
from threat.prioritizer import Prioritizer
from planner.mpc_casadi import NMPCCasadi
from sim.plot_utils import plot_2d_scene

DT = 0.1
STEPS = 80
LANE_HALF = 1.75


def step_ego(state, u):
    x, y, psi, v = state
    a, delta = u
    x += v * math.cos(psi) * DT
    y += v * math.sin(psi) * DT
    psi += (v / 2.5) * math.tan(delta) * DT
    v = max(0.0, v + a * DT)
    return np.array([x, y, psi, v])


def run_single(use_threat):
    # Ego starts fast; braking alone insufficient
    ego = np.array([0.0, 0.0, 0.0, 12.0])

    # -------------------------------------------------
    # EMERGENCY RISK TRADE-OFF SCENARIO (SAFE FRAMING)
    # -------------------------------------------------
    # Pedestrian: high vulnerability, lateral drift
    # Vehicle: lower vulnerability, larger clearance potential
    obstacles = {
        1: {  # Pedestrian (higher risk)
            "x": 14.0, "y": -0.6, "vx": 0.6, "vy": 0.8, "kind": "pedestrian"
        },
        2: {  # Vehicle (lower relative risk)
            "x": 16.0, "y": 0.9, "vx": 4.0, "vy": 0.0, "kind": "car"
        }
    }

    prio = Prioritizer("configs/threat.yaml")
    mpc = NMPCCasadi()

    # Logs
    ego_traj = []
    obs_traj = {tid: [] for tid in obstacles}

    y_log, steer_log, clr_log = [], [], []
    reaction_time = None
    prev_steer = 0.0
    steer_jerk = 0.0

    # Risk integral (threat × proximity)
    risk_integral = 0.0

    for step in range(STEPS):
        preds, threats = {}, {}
        ego_traj.append((ego[0], ego[1]))

        for tid, obj in obstacles.items():
            obs_traj[tid].append((obj["x"], obj["y"]))

            track = {
                "x": obj["x"], "y": obj["y"],
                "vx": obj["vx"], "vy": obj["vy"], "yaw": 0.0
            }

            pred = predict_ctrv(track, mpc.N, DT)
            preds[tid] = pred

            th, _ = prio.compute(
                tid, (ego[0], ego[1], ego[2], ego[3]),
                pred, kind=obj["kind"], class_conf=1.0, dt=DT
            )
            threats[tid] = th

        threats_used = threats if use_threat else {tid: 1.0 for tid in threats}

        # Straight reference within lane
        refs = np.array([
            [ego[0] + ego[3] * DT * (k + 1), 0.0, 0.0, ego[3]]
            for k in range(mpc.N)
        ])

        res = mpc.solve(ego, refs, preds, threats_used)
        u = res["u0"] if res.get("success", False) else (-3.0, 0.2)

        ego = step_ego(ego, u)

        # Move obstacles
        for obj in obstacles.values():
            obj["x"] += obj["vx"] * DT
            obj["y"] += obj["vy"] * DT

        # Logs
        y_log.append(ego[1])
        steer_log.append(u[1])

        # Clearance & risk
        min_clr = float("inf")
        step_risk = 0.0
        for tid, obj in obstacles.items():
            d = math.hypot(obj["x"] - ego[0], obj["y"] - ego[1])
            min_clr = min(min_clr, d)
            step_risk += threats_used[tid] * math.exp(-d)
        clr_log.append(min_clr)
        risk_integral += step_risk * DT

        if reaction_time is None and abs(u[1]) > 0.02:
            reaction_time = step * DT

        steer_jerk += abs(u[1] - prev_steer)
        prev_steer = u[1]

    return {
        "ego_traj": ego_traj,
        "obs_traj": obs_traj,
        "y": y_log,
        "steer": steer_log,
        "clearance": clr_log,
        "reaction_time": reaction_time,
        "steer_jerk": steer_jerk,
        "risk_integral": risk_integral
    }


def fmt_time(t):
    return f"{t:.2f} s" if t is not None else "No reaction"


def run():
    base = run_single(use_threat=False)
    th = run_single(use_threat=True)

    t = np.arange(len(base["y"])) * DT

    # -------- Time-domain plots --------
    plt.figure()
    plt.plot(t, base["y"], "--", label="Baseline")
    plt.plot(t, th["y"], "-", label="Threat-Aware")
    plt.axhline(LANE_HALF, color="gray", linestyle="--")
    plt.axhline(-LANE_HALF, color="gray", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Lateral position y (m)")
    plt.title("Emergency Trade-off: Lateral Position")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t, base["steer"], "--", label="Baseline")
    plt.plot(t, th["steer"], "-", label="Threat-Aware")
    plt.xlabel("Time (s)")
    plt.ylabel("Steering angle (rad)")
    plt.title("Emergency Trade-off: Steering")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t, base["clearance"], "--", label="Baseline")
    plt.plot(t, th["clearance"], "-", label="Threat-Aware")
    plt.xlabel("Time (s)")
    plt.ylabel("Minimum clearance (m)")
    plt.title("Emergency Trade-off: Clearance")
    plt.grid()
    plt.legend()
    plt.show()

    # -------- 2D scene --------
    plot_2d_scene(base["ego_traj"], base["obs_traj"],
                  "2D Emergency Trade-off — Baseline NMPC")
    plot_2d_scene(th["ego_traj"], th["obs_traj"],
                  "2D Emergency Trade-off — Threat-Aware NMPC")

    # -------- Metrics --------
    print("\n===== EMERGENCY RISK TRADE-OFF METRICS =====")

    print("\nBaseline NMPC:")
    print(f"  Reaction time: {fmt_time(base['reaction_time'])}")
    print(f"  Steering jerk: {base['steer_jerk']:.3f}")
    print(f"  Risk integral: {base['risk_integral']:.3f}")

    print("\nThreat-Aware NMPC:")
    print(f"  Reaction time: {fmt_time(th['reaction_time'])}")
    print(f"  Steering jerk: {th['steer_jerk']:.3f}")
    print(f"  Risk integral: {th['risk_integral']:.3f}")


if __name__ == "__main__":
    run()
