import numpy as np
import math

from predictor.ctrv import predict_ctrv
from threat.prioritizer import Prioritizer
from planner.mpc_casadi import NMPCCasadi

DT = 0.1
HORIZON_STEPS = 80

KITTI_FILE = "datasets/kitti/labels/0002.txt"


def load_kitti_tracks(path):

    tracks = {}

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            frame = int(parts[0])
            track_id = int(parts[1])
            obj_type = parts[2]

            if obj_type != "Car":
                continue

            x = float(parts[13])
            z = float(parts[15])

            if track_id not in tracks:
                tracks[track_id] = []

            tracks[track_id].append((frame, x, z))

    return tracks


def compute_velocity(track, idx):
    if idx == 0:
        return 0.0, 0.0

    x0, z0 = track[idx - 1][1], track[idx - 1][2]
    x1, z1 = track[idx][1], track[idx][2]

    vx = (x1 - x0) / DT
    vz = (z1 - z0) / DT
    return vx, vz


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
    return ttc if ttc > 0 else float("inf")


def run_mode(mode):

    tracks = load_kitti_tracks(KITTI_FILE)

    # choose one main vehicle to create conflict with
    main_id = list(tracks.keys())[0]
    track = tracks[main_id]

    prio = Prioritizer("configs/threat.yaml")
    mpc = NMPCCasadi()

    # Place ego 8m behind first frame of real vehicle
    _, init_x, init_z = track[0]
    ego = np.array([init_x - 8.0, init_z, 0.0, 8.0])

    risk_integral = 0.0
    ttc_exposure = 0.0
    min_clearance = float("inf")

    for step in range(min(HORIZON_STEPS, len(track))):

        _, x, z = track[step]
        vx, vz = compute_velocity(track, step)

        obstacle = {
            "x": x,
            "y": z,
            "vx": vx,
            "vy": vz,
            "kind": "vehicle"
        }

        dist = math.hypot(obstacle["x"] - ego[0],
                          obstacle["y"] - ego[1])
        min_clearance = min(min_clearance, dist)

        pred = predict_ctrv({
            "x": x,
            "y": z,
            "vx": vx,
            "vy": vz,
            "yaw": 0.0
        }, mpc.N, DT)

        preds = {0: pred}

        if mode == "baseline":
            threats = {0: 1.0}
        else:
            th, _ = prio.compute(
                0,
                (ego[0], ego[1], ego[2], ego[3]),
                pred,
                kind="vehicle",
                class_conf=1.0,
                dt=DT
            )
            threats = {0: th}

        refs = np.array([
            [ego[0] + ego[3] * DT * (k + 1), ego[1], 0.0, ego[3]]
            for k in range(mpc.N)
        ])

        res = mpc.solve(ego, refs, preds, threats)
        u = res["u0"] if res.get("success", False) else (-2.0, 0.3)

        ego = step_ego(ego, u)

        ttc = compute_ttc(ego, obstacle)
        if ttc < 1.5:
            ttc_exposure += DT

        risk_integral += math.exp(-dist) * DT

    return risk_integral, min_clearance, ttc_exposure


def run():

    print("\n===== KITTI Conflict Replay (Injected) =====")

    base = run_mode("baseline")
    threat = run_mode("threat")

    print("\nBaseline:")
    print(f"Risk integral: {base[0]:.3f}")
    print(f"Min clearance: {base[1]:.3f}")
    print(f"TTC exposure (<1.5s): {base[2]:.3f}")

    print("\nThreat-aware:")
    print(f"Risk integral: {threat[0]:.3f}")
    print(f"Min clearance: {threat[1]:.3f}")
    print(f"TTC exposure (<1.5s): {threat[2]:.3f}")


if __name__ == "__main__":
    run()
    