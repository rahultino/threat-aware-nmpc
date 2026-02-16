# perception/tracker.py
"""
Simple multi-target tracker:
- Kalman filter per track (state: x, y, vx, vy)
- Hungarian data association
- Track lifecycle: tentative -> confirmed -> lost -> delete
Dependencies: numpy, scipy
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import math
from dataclasses import dataclass
from typing import Dict

@dataclass
class Track:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    P: np.ndarray
    last_update: float
    hits: int = 1
    age: int = 1
    state: str = "tentative"

    def state_vec(self):
        return np.array([self.x, self.y, self.vx, self.vy])

    def predict(self, dt, process_q=1e-2):
        F = np.array([[1,0,dt,0],
                      [0,1,0,dt],
                      [0,0,1,0],
                      [0,0,0,1]])
        Q = process_q * np.eye(4)
        x = self.state_vec().reshape(4,1)
        x = F @ x
        self.x, self.y, self.vx, self.vy = x.flatten()
        self.P = F @ self.P @ F.T + Q
        self.age += 1

    def update(self, meas_xy, meas_R=None):
        H = np.array([[1,0,0,0],[0,1,0,0]])
        z = np.array(meas_xy).reshape(2,1)
        x = self.state_vec().reshape(4,1)
        R = meas_R if meas_R is not None else 0.5 * np.eye(2)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - H @ x
        x = x + K @ y
        self.x, self.y, self.vx, self.vy = x.flatten()
        self.P = (np.eye(4) - K @ H) @ self.P
        self.hits += 1
        self.last_update = time.time()
        if self.state == "tentative" and self.hits >= 3:
            self.state = "confirmed"

class Tracker:
    def __init__(self, dist_thresh=4.0, max_missed=8):
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.dist_thresh = dist_thresh
        self.max_missed = max_missed

    def _predict_all(self, dt):
        for tr in list(self.tracks.values()):
            tr.predict(dt)

    def _cost_matrix(self, preds, dets):
        n = len(preds); m = len(dets)
        if n == 0 or m == 0:
            return np.zeros((n,m))
        C = np.zeros((n,m))
        for i,(px,py) in enumerate(preds):
            for j,(dx,dy) in enumerate(dets):
                C[i,j] = math.hypot(px-dx, py-dy)
        return C

    def _create_track(self, meas_xy):
        x,y = meas_xy
        P0 = np.diag([2.0,2.0,5.0,5.0])
        tr = Track(self.next_id, float(x), float(y), 0.0, 0.0, P0, time.time())
        self.tracks[self.next_id] = tr
        self.next_id += 1

    def step(self, detections, dt):
        """
        detections: list of (x,y)
        dt: time since last call
        returns dict {id: {'id', 'x','y','vx','vy','P','state'}}
        """
        # 1) predict
        self._predict_all(dt)

        track_ids = list(self.tracks.keys())
        preds = [(self.tracks[tid].x, self.tracks[tid].y) for tid in track_ids]
        dets = detections.copy()

        assigned_track_ids = set()
        assigned_det_indices = set()

        if len(preds) > 0 and len(dets) > 0:
            C = self._cost_matrix(preds, dets)
            row_ind, col_ind = linear_sum_assignment(C)
            for r,c in zip(row_ind, col_ind):
                if C[r,c] <= self.dist_thresh:
                    tid = track_ids[r]
                    meas = dets[c]
                    self.tracks[tid].update(meas)
                    self.tracks[tid].last_update = time.time()
                    assigned_track_ids.add(tid)
                    assigned_det_indices.add(c)

        # create tracks for unassigned detections
        for i,d in enumerate(dets):
            if i not in assigned_det_indices:
                self._create_track(d)

        # age-out old tracks
        to_delete = []
        now = time.time()
        for tid,tr in list(self.tracks.items()):
            if tid not in assigned_track_ids:
                if (now - tr.last_update) > (self.max_missed * 0.05 + 0.1):
                    tr.state = "lost"
            if tr.age > 200 or (tr.state == "lost" and (now - tr.last_update) > (self.max_missed * 0.1)):
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        return self._output_active()

    def _output_active(self):
        out = {}
        for tid, tr in self.tracks.items():
            out[tid] = {
                'id': tid,
                'x': float(tr.x),
                'y': float(tr.y),
                'vx': float(tr.vx),
                'vy': float(tr.vy),
                'P': tr.P.copy(),
                'state': tr.state
            }
        return out
