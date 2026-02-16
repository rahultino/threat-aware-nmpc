# sim/run_performance_analysis.py
import math
import time
import numpy as np

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


def run_mode(mode):
    """
    mode = "baseline"
    mode = "prediction_only"
    mode = "threat_aware"
    """

    ego = np.array([0.0, 0.0, 0.0, 10.0])

    obstacles = {
        1: {"x": 14.0, "y": -3.2, "vx": 6.0, "vy": 1.8, "kind": "car"},
        2: {"x": 20.0, "y": 0.0,  "vx": 3.0, "vy": 0.0, "kind": "car"},
        3: {"x": 18.0, "y": 2.8,  "vx": 5.0, "vy": 0.0, "kind": "car"},
    }

    prio = Prioritizer("configs/threat.yaml")
    mpc = NMPCCasadi()

    solve_times = []

    for step in range(STEPS):
        preds = {}
        threats = {}

        for tid, obj in obstacles.items():

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

        refs = np.array([
            [ego[0] + ego[3] * DT * (k + 1), 0.0, 0.0, ego[3]]
            for k in range(mpc.N)
        ])

        start = time.perf_counter()
        res = mpc.solve(ego, refs, preds, threats)
        end = time.perf_counter()

        solve_times.append(end - start)

        u = res["u0"] if res.get("success", False) else (-1.0, 0.3)
        ego = step_ego(ego, u)

        for obj in obstacles.values():
            obj["x"] += obj["vx"] * DT
            obj["y"] += obj["vy"] * DT

    solve_times = np.array(solve_times)

    return {
        "mean": np.mean(solve_times),
        "max": np.max(solve_times),
        "std": np.std(solve_times)
    }


def run():
    print("\nRunning performance analysis...\n")

    base = run_mode("baseline")
    pred = run_mode("prediction_only")
    threat = run_mode("threat_aware")

    print("===== COMPUTATIONAL PERFORMANCE =====\n")

    print("Baseline NMPC:")
    print(f"  Mean solve time: {base['mean']*1000:.2f} ms")
    print(f"  Max solve time:  {base['max']*1000:.2f} ms")
    print(f"  Std deviation:   {base['std']*1000:.2f} ms\n")

    print("Prediction-only NMPC:")
    print(f"  Mean solve time: {pred['mean']*1000:.2f} ms")
    print(f"  Max solve time:  {pred['max']*1000:.2f} ms")
    print(f"  Std deviation:   {pred['std']*1000:.2f} ms\n")

    print("Threat-Aware NMPC:")
    print(f"  Mean solve time: {threat['mean']*1000:.2f} ms")
    print(f"  Max solve time:  {threat['max']*1000:.2f} ms")
    print(f"  Std deviation:   {threat['std']*1000:.2f} ms\n")

    print("Control cycle time (DT): 100 ms")

    if threat["mean"] < DT:
        print("\nResult: Real-time feasible on current hardware.")
    else:
        print("\nResult: Not real-time feasible.")


if __name__ == "__main__":
    run()
