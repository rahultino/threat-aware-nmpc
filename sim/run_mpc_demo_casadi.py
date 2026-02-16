#!/usr/bin/env python3
"""
sim/run_mpc_demo_casadi.py (logged)

Closed-loop demo: sim -> noisy detections -> tracker -> ctrv -> threat -> CasADi NMPC -> apply control
Provides run_demo(return_logs=True) to return/save detailed logs.

Run (example):
  (1) activate your conda env av_nmpc
  (2) cd to your project root
  (3) python -m sim.run_mpc_demo_casadi         # interactive plot (default)
  # or get logs:
  (in Python) from sim.run_mpc_demo_casadi import run_demo
             logs = run_demo(return_logs=True, save_path='logs/run1.json')
"""

import os
import sys
import math
import random
import time
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np

from perception.tracker import Tracker
from predictor.ctrv import predict_ctrv
from threat.prioritizer import Prioritizer
from planner.mpc_casadi import NMPCCasadi

DT = 0.1
SIM_T = 12.0

class SimObstacle:
    def __init__(self, x, y, vx, vy, kind='vehicle', id=None):
        self.x = float(x); self.y = float(y)
        self.vx = float(vx); self.vy = float(vy)
        self.kind = kind
        self.id = id

    def step(self, dt=DT):
        self.x += self.vx * dt
        self.y += self.vy * dt

    def pos(self):
        return (self.x, self.y)

def kinematic_step_np(state, u, dt, wheelbase=2.5):
    x, y, psi, v = state
    a, delta = u
    x_next = x + v * math.cos(psi) * dt
    y_next = y + v * math.sin(psi) * dt
    psi_next = psi + (v / wheelbase) * math.tan(delta) * dt
    v_next = max(0.0, v + a * dt)
    return np.array([x_next, y_next, psi_next, v_next])

def color_from_threat(th):
    cmap = plt.get_cmap('RdYlGn_r')
    return cmap(th)

def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def run_demo(return_logs: bool = False, save_path: str = None, random_seed: int = 0):
    """
    Run the CasADi NMPC demo.
    If return_logs=True, returns a dict of logs and optionally saves JSON to save_path.
    Logs include:
      - ego_hist (list of states)
      - ctrl_hist (list of applied controls)
      - per_step: [{'t':..., 'solve_time':..., 'success':bool, 'threats':{tid:th}, 'min_distances':{tid:d}}]
      - mean_solve_time, fallback_count, min_distance_overall
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # scenario: crossing pedestrian + cut-in vehicle
    obs = [
        SimObstacle(15.0, 3.0, -1.2, 0.0, kind='pedestrian', id=101),
        SimObstacle(30.0, 0.5, -3.33, 0.0, kind='vehicle', id=102)
    ]

    ego = np.array([0.0, 0.0, 0.0, 8.0])  # x,y,psi,v

    tracker = Tracker(dist_thresh=4.0, max_missed=8)
    prio = Prioritizer(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'threat.yaml'))
    mpc = NMPCCasadi(horizon=14, dt=DT, slack_penalty=800.0, d_safe=3.0,
                     ipopt_options={'ipopt.max_iter':200, 'ipopt.tol':1e-4, 'ipopt.print_level':0})

    # logs
    ego_hist = []
    ctrl_hist = []
    tracks_log = []
    threats_log = []
    preds_log = []
    per_step = []
    solve_times = []
    fallback_count = 0
    min_distances_overall = []

    t = 0.0
    target_speed = 8.0

    while t < SIM_T:
        # step obstacles
        for o in obs:
            o.step(DT)

        # detections (noisy)
        detections = []
        for o in obs:
            ox, oy = o.pos()
            nx = ox + random.gauss(0, 0.25)
            ny = oy + random.gauss(0, 0.25)
            detections.append((nx, ny))

        # perception
        tracks = tracker.step(detections, DT)

        # prepare preds and threats (keys are tracker IDs)
        preds_map = {}
        threats_map = {}
        per_step_min_dist = {}
        for tid, tr in tracks.items():
            preds = predict_ctrv(tr, steps=mpc.N, dt=DT)
            preds_map[tid] = preds
            kind = tr.get('kind', 'vehicle')
            th, info = prio.compute(tid, (ego[0], ego[1], ego[2], ego[3]), preds, kind=kind, class_conf=1.0, dt=DT)
            threats_map[tid] = th
            per_step_min_dist[tid] = float(info.get('min_d', 1e9))

        # build refs
        refs = []
        for k in range(mpc.N):
            t_ref = (k+1) * DT
            xr = ego[0] + target_speed * math.cos(ego[2]) * t_ref
            yr = ego[1] + target_speed * math.sin(ego[2]) * t_ref
            refs.append(np.array([xr, yr, ego[2], target_speed]))
        refs_arr = np.array(refs)

        # ensure solver built for current obs count
        num_obs = len(preds_map)
        if mpc._nlp is None or mpc._dims is None or mpc._dims.get('num_obs', -1) != num_obs:
            mpc.create_nlpsol(num_obs=num_obs)

        # solve and measure time
        t_solve_start = time.time()
        res = mpc.solve(ego, refs_arr, preds_map, threats_map, warm_start=True, max_cpu_time=0.25)
        t_solve = time.time() - t_solve_start
        solve_times.append(t_solve)

        if not res.get('success', False):
            fallback_count += 1
            applied = (-4.0, 0.0)
            success_flag = False
        else:
            u0 = res['u_seq'][0]
            applied = (float(u0[0]), float(u0[1]))
            success_flag = True

        # apply control
        ego = kinematic_step_np(ego, applied, DT)
        ego_hist.append(ego.copy())
        ctrl_hist.append(applied)
        tracks_log.append(tracks)
        threats_log.append(threats_map.copy())
        preds_log.append({k: [(float(p[0][0]), float(p[0][1])) for p in v] for k,v in preds_map.items()})

        # collect per-step metrics
        per_step.append({
            't': round(t, 4),
            'solve_time': float(t_solve),
            'success': bool(success_flag),
            'threats': {int(k): float(v) for k,v in threats_map.items()},
            'min_distances': {int(k): float(per_step_min_dist.get(k, 1e9)) for k in preds_map.keys()}
        })

        # update overall min distances to true obstacles (compute truth dist)
        for o in obs:
            d = math.hypot(o.x - ego[0], o.y - ego[1])
            min_distances_overall.append(float(d))

        t += DT

    mean_solve = float(np.mean(solve_times)) if solve_times else 0.0
    overall_min_dist = float(np.min(min_distances_overall)) if min_distances_overall else float('inf')

    # summary
    summary = {
        'mean_solve_time': mean_solve,
        'fallback_count': int(fallback_count),
        'overall_min_distance': overall_min_dist,
        'steps': len(per_step),
        'horizon': mpc.N
    }

    # prepare logs object
    logs = {
        'summary': summary,
        'ego_hist': [list(map(float, s)) for s in ego_hist],
        'ctrl_hist': [list(map(float, u)) for u in ctrl_hist],
        'per_step': per_step,
        'threats_log': threats_log,
        'preds_log': preds_log
    }

    # save if requested
    if save_path:
        _ensure_dir(save_path)
        with open(save_path, 'w') as f:
            json.dump(logs, f, indent=2)
        print(f"Saved logs to {save_path}")

    # print summary
    print(f"NMPC mean solve time per step: {mean_solve:.4f} s (N={mpc.N}), fallbacks: {fallback_count}, min dist: {overall_min_dist:.3f} m")

    if return_logs:
        return logs

    # otherwise show plot
    ego_arr = np.array(ego_hist)
    plt.figure(figsize=(10,6))
    plt.plot(ego_arr[:,0], ego_arr[:,1], '-b', linewidth=2, label='Ego trajectory')

    last_tracks = tracks_log[-1] if len(tracks_log)>0 else {}
    last_threats = threats_log[-1] if len(threats_log)>0 else {}
    for tid, tr in last_tracks.items():
        th = last_threats.get(tid, 0.0)
        plt.scatter(tr['x'], tr['y'], s=120, color=color_from_threat(th), edgecolor='k', label=f"track {tid} th={th:.2f}")
        preds = predict_ctrv(tr, steps=mpc.N, dt=DT)
        px = [p[0][0] for p in preds]; py = [p[0][1] for p in preds]
        plt.plot(px, py, '--', linewidth=1, alpha=0.6)

    plt.title("CasADi NMPC closed-loop demo (threat-weighted slacks)")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.axis('equal'); plt.grid(True); plt.legend()
    plt.show()

if __name__ == "__main__":
    # default interactive run (shows plot)
    run_demo(return_logs=False)
    