# sim/run_ablation_demo.py
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


def compute_ttc(ego, obj):
    dx = obj["x"] - ego[0]
    dy = obj["y"] - ego[1]

    rel_vx = obj["vx"] - ego[3] * math.cos(ego[2])
    rel_vy = obj["vy"] - ego[3] * math.sin(ego[2])

    rel_speed_sq = rel_vx**2 + rel_vy**2
    if rel_speed_sq < 1e-4:
        return float("inf")

    ttc = -(dx * rel_vx + dy * rel_vy) / rel_speed_sq

    if ttc <= 0:
        return float("inf")

    return ttc


def run_mode(mode):
    """
    mode:
        "baseline"
        "prediction_only"
        "threat_aware"
    """

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

    y_log, steer_log, clr_log = [], [], []
    ttc_log = []

    risk_integral = 0.0

    for step in range(STEPS):

        preds = {}
        threats = {}
        ego_traj.append((ego[0], ego[1]))

        min_ttc_step = float("inf")

        for tid, obj in obstacles.items():
            obs_traj[tid].append((obj["x"], obj["y"]))

            # Compute TTC
            ttc_val = compute_ttc(ego, obj)
            min_ttc_step = min(min_ttc_step, ttc_val)

            if mode == "baseline":
                preds[tid] = [(obj["x"], obj["y"])] * mpc.N
                threats[tid] = 1.0

            else:
                track = {
                    "x": obj["x"],
                    "y": obj["y"],
                    "vx": obj["vx"],
                    "vy": obj["vy"],
                    "yaw": 0.0
                }

                pred = predict_ctrv(track, mpc.N, DT)
                preds[tid] = pred

                if mode == "prediction_only":
                    threats[tid] = 1.0
                else:
                    th, _ = prio.compute(
                        tid,
                        (ego[0], ego[1], ego[2], ego[3]),
                        pred,
                        kind=obj["kind"],
                        class_conf=1.0,
                        dt=DT
                    )
                    threats[tid] = th

        ttc_log.append(min_ttc_step)

        refs = np.array([
            [ego[0] + ego[3] * DT * (k + 1), 0.0, 0.0, ego[3]]
            for k in range(mpc.N)
        ])

        res = mpc.solve(ego, refs, preds, threats)
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

        step_risk = sum(threats[tid] * math.exp(-min_clr)
                        for tid in threats)
        risk_integral += step_risk * DT

    return ego_traj, obs_traj, y_log, steer_log, clr_log, risk_integral, ttc_log


def run():
    base = run_mode("baseline")
    pred = run_mode("prediction_only")
    threat = run_mode("threat_aware")

    t = np.arange(len(base[2])) * DT

    # --- Steering ---
    plt.figure()
    plt.plot(t, base[3], "--", label="Baseline")
    plt.plot(t, pred[3], "-.", label="Prediction Only")
    plt.plot(t, threat[3], "-", label="Threat-Aware")
    plt.xlabel("Time (s)")
    plt.ylabel("Steering (rad)")
    plt.title("Ablation: Steering Comparison")
    plt.grid()
    plt.legend()
    plt.show()

    # --- Clearance ---
    plt.figure()
    plt.plot(t, base[4], "--", label="Baseline")
    plt.plot(t, pred[4], "-.", label="Prediction Only")
    plt.plot(t, threat[4], "-", label="Threat-Aware")
    plt.xlabel("Time (s)")
    plt.ylabel("Clearance (m)")
    plt.title("Ablation: Clearance Comparison")
    plt.grid()
    plt.legend()
    plt.show()

    # --- TTC ---
    plt.figure()
    plt.plot(t, base[6], "--", label="Baseline")
    plt.plot(t, pred[6], "-.", label="Prediction Only")
    plt.plot(t, threat[6], "-", label="Threat-Aware")
    plt.xlabel("Time (s)")
    plt.ylabel("Minimum TTC (s)")
    plt.title("Ablation: Minimum TTC Comparison")
    plt.grid()
    plt.legend()
    plt.show()

    # -------------------------------
    # Advanced TTC Metrics
    # -------------------------------
    def ttc_metrics(ttc_log):
        ttc_array = np.array(ttc_log)

        avg_ttc = np.mean(ttc_array[np.isfinite(ttc_array)])

        time_below_1 = np.sum(ttc_array < 1.0) * DT
        time_below_05 = np.sum(ttc_array < 0.5) * DT

        percent_below_1 = 100 * np.sum(ttc_array < 1.0) / len(ttc_array)
        percent_below_05 = 100 * np.sum(ttc_array < 0.5) / len(ttc_array)

        return avg_ttc, time_below_1, time_below_05, percent_below_1, percent_below_05

    base_metrics = ttc_metrics(base[6])
    pred_metrics = ttc_metrics(pred[6])
    threat_metrics = ttc_metrics(threat[6])

    print("\n===== ABLATION STUDY RESULTS =====")
    print(f"Baseline risk integral:        {base[5]:.3f}")
    print(f"Prediction-only risk integral: {pred[5]:.3f}")
    print(f"Threat-aware risk integral:    {threat[5]:.3f}")

    print("\n===== ADVANCED TTC METRICS =====")

    print("\nBaseline:")
    print(f"  Avg TTC: {base_metrics[0]:.2f} s")
    print(f"  Time TTC < 1.0s: {base_metrics[1]:.2f} s")
    print(f"  Time TTC < 0.5s: {base_metrics[2]:.2f} s")
    print(f"  % TTC < 1.0s: {base_metrics[3]:.1f}%")
    print(f"  % TTC < 0.5s: {base_metrics[4]:.1f}%")

    print("\nPrediction-only:")
    print(f"  Avg TTC: {pred_metrics[0]:.2f} s")
    print(f"  Time TTC < 1.0s: {pred_metrics[1]:.2f} s")
    print(f"  Time TTC < 0.5s: {pred_metrics[2]:.2f} s")
    print(f"  % TTC < 1.0s: {pred_metrics[3]:.1f}%")
    print(f"  % TTC < 0.5s: {pred_metrics[4]:.1f}%")

    print("\nThreat-aware:")
    print(f"  Avg TTC: {threat_metrics[0]:.2f} s")
    print(f"  Time TTC < 1.0s: {threat_metrics[1]:.2f} s")
    print(f"  Time TTC < 0.5s: {threat_metrics[2]:.2f} s")
    print(f"  % TTC < 1.0s: {threat_metrics[3]:.1f}%")
    print(f"  % TTC < 0.5s: {threat_metrics[4]:.1f}%")

    # --- 2D Trajectories ---
    plot_2d_scene(base[0], base[1], "Baseline 2D")
    plot_2d_scene(pred[0], pred[1], "Prediction Only 2D")
    plot_2d_scene(threat[0], threat[1], "Threat-Aware 2D")


if __name__ == "__main__":
    run()
