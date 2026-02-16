#!/usr/bin/env python3
"""
sim/run_threat_demo.py
Stage 3 demo: run tracker + CTRV + threat prioritizer and plot color-coded tracks.

Run:
    python -m sim.run_threat_demo
or
    python sim/run_threat_demo.py
(from project root)
"""

import math, random, time, os, sys
import matplotlib.pyplot as plt
from perception.tracker import Tracker
from predictor.ctrv import predict_ctrv, cov_to_ellipse
from threat.prioritizer import Prioritizer, load_config

# ensure project root on path when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DT = 0.1
SIM_T = 10.0

class SimObstacle:
    def __init__(self, x, y, vx, vy, kind='vehicle'):
        self.x = float(x); self.y = float(y)
        self.vx = float(vx); self.vy = float(vy)
        self.kind = kind

    def step(self, dt=DT):
        self.x += self.vx * dt
        self.y += self.vy * dt

    def pos(self):
        return (self.x, self.y)

def color_from_threat(th):
    # map 0..1 to a color (green low -> red high)
    cmap = plt.get_cmap('RdYlGn_r')
    return cmap(th)

def run_demo(cfg_path=None):
    # scenario
    obs = [
        SimObstacle(15.0, 3.0, -1.2, 0.0, kind='pedestrian'),
        SimObstacle(30.0, 0.5, -3.33, 0.0, kind='vehicle'),
        SimObstacle(20.0, -2.0, -5.0, 0.0, kind='cyclist')
    ]

    tracker = Tracker(dist_thresh=4.0, max_missed=8)
    pr = Prioritizer(cfg_path)

    t = 0.0
    history = {'tracks': [], 'threats': []}
    ego_state = (0.0, 0.0, 0.0, 8.0)  # static ego for demo (x,y,yaw,v)

    while t < SIM_T:
        for o in obs:
            o.step(DT)

        # noisy detections
        detections = []
        for o in obs:
            ox, oy = o.pos()
            nx = ox + random.gauss(0, 0.3)
            ny = oy + random.gauss(0, 0.3)
            detections.append((nx, ny))

        tracks = tracker.step(detections, DT)
        threats_this_frame = {}
        preds_map = {}
        # compute predictions and threats
        for tid, tr in tracks.items():
            preds = predict_ctrv(tr, steps=20, dt=DT)
            preds_map[tid] = preds
            th, info = pr.compute(tid, ego_state, preds, kind=tr.get('kind', 'vehicle'), class_conf=1.0, dt=DT)
            threats_this_frame[tid] = (th, info)

        history['tracks'].append(tracks)
        history['threats'].append(threats_this_frame)

        t += DT
        time.sleep(0.0)

    # plotting final frame
    last_tracks = history['tracks'][-1]
    last_threats = history['threats'][-1]
    plt.figure(figsize=(10,6))

    # show detections across frames faintly
    for frame_idx, trset in enumerate(history['tracks']):
        for tid, tr in trset.items():
            plt.scatter(tr['x'], tr['y'], c='lightgray', s=8, alpha=0.2)

    # plot final tracks color-coded by threat
    for tid, tr in last_tracks.items():
        th, info = last_threats.get(tid, (0.0, {}))
        plt.scatter(tr['x'], tr['y'], s=120, color=color_from_threat(th), edgecolor='k', label=f"track {tid}: {th:.2f}")
        # plot predicted means
        preds = predict_ctrv(tr, steps=20, dt=DT)
        px = [p[0][0] for p in preds]; py = [p[0][1] for p in preds]
        plt.plot(px, py, '--', linewidth=1)
        # ellipses
        for idx in range(0, len(preds), 5):
            pos, cov = preds[idx]
            w,h,ang = cov_to_ellipse(cov, nsig=2.0)
            thetas = [i*2*math.pi/60 for i in range(60)]
            xs = [(w/2) * math.cos(th) for th in thetas]
            ys = [(h/2) * math.sin(th) for th in thetas]
            c = math.cos(ang); s = math.sin(ang)
            rx = [c*xx - s*yy + pos[0] for xx,yy in zip(xs,ys)]
            ry = [s*xx + c*yy + pos[1] for xx,yy in zip(xs,ys)]
            plt.plot(rx, ry, color='orange', alpha=0.5)

    plt.title("Threat Prioritizer demo: final tracks colored by threat (green low -> red high)")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.axis('equal'); plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'threat.yaml')
    run_demo(cfg_path)
