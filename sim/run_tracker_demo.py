    #!/usr/bin/env python3
"""
sim/run_tracker_demo.py
Stage 2 demo: simulate obstacles, produce noisy detections, run Tracker + CTRV predictor,
and plot tracks with predicted trajectories and uncertainty ellipses.

Run:
    python sim/run_tracker_demo.py
"""

import random
import math
import time
import matplotlib.pyplot as plt
from perception.tracker import Tracker
from predictor.ctrv import predict_ctrv, cov_to_ellipse

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

def run_demo():
    obs = [
        SimObstacle(15.0, 3.0, -1.2, 0.0, kind='pedestrian'),
        SimObstacle(30.0, 0.5, -3.33, 0.0, kind='vehicle'),
        SimObstacle(20.0, -2.0, -5.0, 0.0, kind='cyclist')
    ]

    tracker = Tracker(dist_thresh=4.0, max_missed=8)

    t = 0.0
    historian = {'dets': [], 'tracks': [], 'preds': []}

    while t < SIM_T:
        for o in obs:
            o.step(DT)

        detections = []
        for o in obs:
            ox, oy = o.pos()
            nx = ox + random.gauss(0, 0.3)
            ny = oy + random.gauss(0, 0.3)
            detections.append((nx, ny))

        tracks = tracker.step(detections, DT)
        preds_for_plot = {}
        for tid, tr in tracks.items():
            preds = predict_ctrv(tr, steps=20, dt=DT)
            preds_for_plot[tid] = preds

        historian['dets'].append(detections)
        historian['tracks'].append(tracks)
        historian['preds'].append(preds_for_plot)

        t += DT
        time.sleep(0.0)

    # plotting
    plt.figure(figsize=(9,6))
    for frame_dets in historian['dets']:
        xs = [d[0] for d in frame_dets]; ys = [d[1] for d in frame_dets]
        plt.scatter(xs, ys, c='lightgray', s=8, alpha=0.6)

    last_tracks = historian['tracks'][-1]
    last_preds = historian['preds'][-1]
    for tid, tr in last_tracks.items():
        plt.scatter(tr['x'], tr['y'], s=80, label=f"track {tid}")
        if tid in last_preds:
            preds = last_preds[tid]
            px = [p[0][0] for p in preds]; py = [p[0][1] for p in preds]
            plt.plot(px, py, '--', linewidth=1)
            for idx in range(0, len(preds), 5):
                pos, cov = preds[idx]
                w,h,ang = cov_to_ellipse(cov, nsig=2.0)
                thetas = [i*2*math.pi/60 for i in range(60)]
                xs = [(w/2) * math.cos(th) for th in thetas]
                ys = [(h/2) * math.sin(th) for th in thetas]
                c = math.cos(ang); s = math.sin(ang)
                rx = [c*xx - s*yy + pos[0] for xx,yy in zip(xs,ys)]
                ry = [s*xx + c*yy + pos[1] for xx,yy in zip(xs,ys)]
                plt.plot(rx, ry, color='orange', alpha=0.6)

    plt.title("Tracker + CTRV predictions (detections in gray)")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.axis('equal'); plt.grid(True); plt.legend()
    plt.show()

if __name__ == "__main__":
    run_demo()
