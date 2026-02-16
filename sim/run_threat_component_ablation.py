import numpy as np
import math

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


def configure_weights(prio, mode):
    cfg = prio.cfg

    if mode == "pos_only":
        cfg['w_pos'] = 1.0
        cfg['w_ttc'] = 0.0
        cfg['w_vel'] = 0.0
        cfg['w_class'] = 0.0

    elif mode == "ttc_only":
        cfg['w_pos'] = 0.0
        cfg['w_ttc'] = 1.0
        cfg['w_vel'] = 0.0
        cfg['w_class'] = 0.0

    elif mode == "full":
        # restore original values from config file
        # reload config fresh
        prio.__init__("configs/threat.yaml")

    elif mode == "uniform":
        pass


def run_mode(mode):

    ego = np.array([0.0, 0.0, 0.0, 10.0])

    obstacles = {
        1: {"x": 14.0, "y": -3.0, "vx": 6.0, "vy": 1.5, "kind": "car"},
        2: {"x": 18.0, "y":  2.5, "vx": 5.0, "vy": 0.0, "kind": "car"},
        3: {"x": 22.0, "y": -2.5, "vx": 4.0, "vy": 0.0, "kind": "car"},
    }

    prio = Prioritizer("configs/threat.yaml")
    configure_weights(prio, mode)

    mpc = NMPCCasadi()

    risk_integral = 0.0

    for step in range(STEPS):

        preds = {}
        threats = {}
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

            if mode == "uniform":
                th = 1.0
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
            original_threats[tid] = th

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

        step_risk = sum(original_threats[tid] * math.exp(-min_clr)
                        for tid in original_threats)
        risk_integral += step_risk * DT

    return risk_integral


def run():

    modes = ["uniform", "pos_only", "ttc_only", "full"]

    print("\n===== Threat Component Ablation =====")

    for mode in modes:
        risk = run_mode(mode)
        print(f"{mode:12s} -> Risk integral = {risk:.3f}")


if __name__ == "__main__":
    run()
