import numpy as np
import time
import math

from predictor.ctrv import predict_ctrv
from threat.prioritizer import Prioritizer
from planner.mpc_casadi import NMPCCasadi

DT = 0.1
STEPS = 60


def step_ego(state, u):
    x, y, psi, v = state
    a, delta = u
    x += v * math.cos(psi) * DT
    y += v * math.sin(psi) * DT
    psi += (v / 2.5) * math.tan(delta) * DT
    v = max(0.0, v + a * DT)
    return np.array([x, y, psi, v])


def generate_obstacles(n):
    obstacles = {}
    for i in range(n):
        obstacles[i] = {
            "x": 15.0 + 3*i,
            "y": (-1)**i * 2.5,
            "vx": 4.0,
            "vy": 0.0,
            "kind": "car"
        }
    return obstacles


def run_mode(mode, obstacle_count):

    ego = np.array([0.0, 0.0, 0.0, 10.0])
    obstacles = generate_obstacles(obstacle_count)

    prio = Prioritizer("configs/threat.yaml")
    mpc = NMPCCasadi()

    solve_times = []
    risk_integral = 0.0

    for step in range(STEPS):

        preds = {}
        threats = {}

        min_clr = float("inf")

        for tid, obj in obstacles.items():

            dx = obj["x"] - ego[0]
            dy = obj["y"] - ego[1]
            dist = math.hypot(dx, dy)
            min_clr = min(min_clr, dist)

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

                th, _ = prio.compute(
                    tid,
                    (ego[0], ego[1], ego[2], ego[3]),
                    pred,
                    kind=obj["kind"],
                    class_conf=1.0,
                    dt=DT
                )
                threats[tid] = th

        refs = np.array([
            [ego[0] + ego[3] * DT * (k + 1), 0.0, 0.0, ego[3]]
            for k in range(mpc.N)
        ])

        start = time.time()
        res = mpc.solve(ego, refs, preds, threats)
        end = time.time()

        solve_times.append(end - start)

        u = res["u0"] if res.get("success", False) else (-1.0, 0.3)
        ego = step_ego(ego, u)

        for obj in obstacles.values():
            obj["x"] += obj["vx"] * DT

        # risk calculation
        step_risk = sum(threats[tid] * math.exp(-min_clr)
                        for tid in threats)
        risk_integral += step_risk * DT

    return np.mean(solve_times), np.max(solve_times), risk_integral


def run():

    obstacle_levels = [1, 3, 5, 7]

    print("\n===== SCALABILITY TEST =====")

    for n in obstacle_levels:

        base_mean, base_max, base_risk = run_mode("baseline", n)
        threat_mean, threat_max, threat_risk = run_mode("threat_aware", n)

        print(f"\nObstacle Count: {n}")
        print(f"Baseline -> Mean: {base_mean*1000:.1f} ms | Max: {base_max*1000:.1f} ms | Risk: {base_risk:.3f}")
        print(f"Threat   -> Mean: {threat_mean*1000:.1f} ms | Max: {threat_max*1000:.1f} ms | Risk: {threat_risk:.3f}")


if __name__ == "__main__":
    run()
