# sim/run_kitti_demo.py
import os
import sys
import math
import json
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.kitti_loader import load_kitti_sequence
from predictor.ctrv import predict_ctrv
from threat.prioritizer import Prioritizer
from planner.mpc_casadi import NMPCCasadi

# ================= USER CONFIG =================
DT = 0.1
SEQ = "0002"
MAX_FRAMES = 120

# COMPARISON SWITCH
BASELINE_MODE = True   # True = Baseline NMPC, False = Threat-Aware NMPC
# ==============================================


def step_ego(state, u):
    """Simple bicycle model"""
    x, y, psi, v = state
    a, delta = u

    x += v * math.cos(psi) * DT
    y += v * math.sin(psi) * DT
    psi += (v / 2.5) * math.tan(delta) * DT
    v = max(0.0, v + a * DT)

    return np.array([x, y, psi, v])


def run():
    print("\n=== Running KITTI NMPC Demo ===")
    print("Sequence:", SEQ)
    print("Mode:", "BASELINE" if BASELINE_MODE else "THREAT-AWARE")

    # ---------- Load KITTI ----------
    frames = load_kitti_sequence(f"datasets/kitti/labels/{SEQ}.txt")

    # ---------- Ego initial state ----------
    ego = np.array([0.0, 0.0, 0.0, 8.0])

    # ---------- Modules ----------
    prio = Prioritizer("configs/threat.yaml")
    mpc = NMPCCasadi()

    # ---------- Logging ----------
    log = {
        "time": [],
        "ego": [],
        "acc": [],
        "steer": [],
        "min_dist": [],
        "mode": "BASELINE" if BASELINE_MODE else "THREAT_AWARE"
    }

    t = 0.0
    ego_traj = []

    # ---------- Main loop ----------
    for frame_id in sorted(frames.keys())[:MAX_FRAMES]:
        objs = frames[frame_id]

        preds = {}
        threats = {}

        # ---- Build predictions & threats ----
        for tid, obj in objs.items():
            track = {
                "x": obj["x"],
                "y": obj["y"],
                "vx": obj["vx"],
                "vy": obj["vy"],
                "yaw": math.atan2(obj["vy"], obj["vx"] + 1e-6)
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

        # ---- Baseline vs Threat-Aware ----
        if BASELINE_MODE:
            threats_used = {tid: 1.0 for tid in threats}
        else:
            threats_used = threats

        # ---- Reference (straight driving) ----
        refs = []
        for k in range(mpc.N):
            xr = ego[0] + ego[3] * DT * (k + 1)
            yr = ego[1]
            refs.append([xr, yr, ego[2], ego[3]])
        refs = np.array(refs)

        # ---- Solve NMPC ----
        if preds:
            res = mpc.solve(ego, refs, preds, threats_used)
            if res.get("success", False):
                u = res["u0"]
            else:
                u = (-3.0, 0.0)  # safe fallback
        else:
            u = (0.0, 0.0)

        # ---- Apply control ----
        ego = step_ego(ego, u)

        # ---- Minimum distance ----
        dmin = float("inf")
        for obj in objs.values():
            d = math.hypot(obj["x"] - ego[0], obj["y"] - ego[1])
            dmin = min(dmin, d)

        # ---- Log ----
        ego_traj.append(ego.copy())
        log["time"].append(t)
        log["ego"].append(ego.tolist())
        log["acc"].append(float(u[0]))
        log["steer"].append(float(u[1]))
        log["min_dist"].append(dmin)

        t += DT

    ego_traj = np.array(ego_traj)

    # ---------- SAVE LOG (IMPORTANT) ----------
    os.makedirs("logs", exist_ok=True)
    out_path = f"logs/{log['mode']}_{SEQ}.json"

    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)

    print(">>> Saved log:", out_path)

    # ---------- Console summary ----------
    print("\n===== RUN SUMMARY =====")
    print("Frames:", len(log["time"]))
    print("Min distance overall:", min(log["min_dist"]))
    print(
        "Avg steering magnitude:",
        sum(abs(s) for s in log["steer"]) / len(log["steer"])
    )

    # ---------- Plot trajectory ----------
    plt.figure(figsize=(10, 6))
    plt.plot(ego_traj[:, 0], ego_traj[:, 1], "b", linewidth=2, label="Ego")

    for f in frames:
        for o in frames[f].values():
            plt.scatter(o["x"], o["y"], c="r", s=5, alpha=0.3)

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(
        "Baseline NMPC" if BASELINE_MODE
        else "Threat-Aware NMPC (Safety-Optimal)"
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()


if __name__ == "__main__":
    run()
