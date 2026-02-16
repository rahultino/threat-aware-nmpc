# threat/prioritizer.py
"""
Threat Prioritizer (Option A - weighted components)

Compute a threat score [0..1] per track using:
 - proximity (min distance over horizon)
 - TTC (approximate, numerically stable)
 - relative speed
 - class weight (pedestrian/cyclist/vehicle)

Features:
 - smoothing (exponential low-pass)
 - persistence/hysteresis (avoid flicker)
 - returns diagnostics for logging/plots
"""

import math
import numpy as np
import yaml
from typing import List, Tuple, Dict, Any

DEFAULT_CFG = {
    'd_safe': 5.0,
    'sigma_d': 3.0,
    'T0': 1.5,
    'k': 0.5,
    'vmax': 30.0,
    'w_pos': 0.25,
    'w_ttc': 0.40,
    'w_vel': 0.20,
    'w_class': 0.15,
    'class_weights': {'pedestrian': 1.0, 'cyclist': 0.8, 'vehicle': 0.6},
    'alpha': 0.35,           # smoothing factor for exponential low-pass
    'persist_frames': 2      # require this many consecutive increases before escalating
}

def load_config(path: str = None) -> Dict[str, Any]:
    cfg = DEFAULT_CFG.copy()
    if path:
        try:
            with open(path, 'r') as f:
                user = yaml.safe_load(f)
            if isinstance(user, dict):
                cfg.update(user)
        except Exception:
            # ignore errors and use defaults
            pass
    return cfg

class Prioritizer:
    def __init__(self, cfg_path: str = None):
        self.cfg = load_config(cfg_path)
        # per-track state: smoothed threat + consecutive_up_count
        self.state = {}  # track_id -> {'smoothed': float, 'count_up': int}

    def _estimate_obs_speed(self, preds: List[Tuple[Tuple[float,float], np.ndarray]], dt: float):
        # Estimate obstacle speed magnitude from first two predictions (finite diff).
        if preds is None or len(preds) < 2:
            return 0.0, np.array([0.0, 0.0])
        p0 = np.array(preds[0][0])
        p1 = np.array(preds[1][0])
        # protect dt
        dt_safe = dt if (dt is not None and dt > 0) else 1.0
        vvec = (p1 - p0) / dt_safe
        return float(np.linalg.norm(vvec)), vvec

    def _compute_min_distance_and_ttc(self, ego_state, preds, dt):
        ex, ey, eyaw, ev = ego_state
        if preds is None or len(preds) == 0:
            return float('inf'), float('inf'), 0.0

        min_d = float('inf')
        min_ttc = float('inf')
        obs_v, obs_vvec = self._estimate_obs_speed(preds, dt)
        for (px, py), cov in preds:
            rel = np.array([px - ex, py - ey])
            dist = float(np.linalg.norm(rel) + 1e-9)
            if dist < min_d:
                min_d = dist
            # approximate closing speed: projection of relative velocity onto relative position
            ego_dir = np.array([math.cos(eyaw), math.sin(eyaw)])
            rel_vel_vec = obs_vvec - ego_dir * ev
            # avoid division by zero in normalization
            if dist > 1e-6:
                closing = float(np.dot(rel_vel_vec, rel / dist))
            else:
                closing = float(np.dot(rel_vel_vec, rel))  # fallback
            if closing > 1e-6:
                ttc = dist / closing
            else:
                ttc = float('inf')
            if ttc < min_ttc:
                min_ttc = ttc
        return min_d, min_ttc, obs_v

    def compute_single(self, ego_state: Tuple[float,float,float,float],
                       preds: List[Tuple[Tuple[float,float], np.ndarray]],
                       kind: str = 'vehicle',
                       class_conf: float = 1.0,
                       dt: float = 0.1) -> Tuple[float, Dict[str, float]]:
        """
        Inputs:
         - ego_state: (x,y,yaw,v)
         - preds: list of ((x_k,y_k), cov_k)
         - kind: 'pedestrian'|'cyclist'|'vehicle'
         - class_conf: confidence [0..1]
         - dt: prediction time-step used by preds
        Returns:
         - raw threat [0..1] (unsmoothed) and diagnostics dict
        """
        cfg = self.cfg
        ex, ey, eyaw, ev = ego_state

        # compute min distance across horizon, min TTC, estimate obs speed
        min_d, min_ttc, obs_v = self._compute_min_distance_and_ttc(ego_state, preds, dt)

        # positional component (closer => larger). Use a smooth exponential decay.
        try:
            P_pos = math.exp(- (min_d - cfg['d_safe']) / cfg['sigma_d'])
        except OverflowError:
            P_pos = 0.0 if (min_d - cfg['d_safe']) > 0 else 1.0
        P_pos = max(0.0, min(1.0, P_pos))

        # TTC component (smaller TTC => larger). Use numerically-stable sigmoid.
        if min_ttc == float('inf') or min_ttc > 1e6:
            P_ttc = 0.0
        else:
            x = (min_ttc - cfg['T0']) / cfg['k']
            # clamp x to avoid math.exp overflow
            if x > 50.0:
                P_ttc = 0.0
            elif x < -50.0:
                P_ttc = 1.0
            else:
                # sigmoid mapping so that ttc << T0 -> P_ttc ~1, ttc >> T0 -> P_ttc ~0
                P_ttc = 1.0 / (1.0 + math.exp(x))
        P_ttc = max(0.0, min(1.0, P_ttc))

        # speed component (higher relative speed => larger)
        P_vel = 0.0
        try:
            P_vel = min(1.0, (obs_v + ev) / cfg['vmax'])
        except Exception:
            P_vel = 0.0

        # class component
        class_w = cfg['class_weights'].get(kind, 0.6)
        P_class = float(class_conf) * float(class_w)

        # weighted sum
        raw = (cfg['w_pos'] * P_pos +
               cfg['w_ttc'] * P_ttc +
               cfg['w_vel'] * P_vel +
               cfg['w_class'] * P_class)

        threat = max(0.0, min(1.0, raw))

        diagnostics = {
            'min_d': min_d,
            'min_ttc': (min_ttc if min_ttc != float('inf') else 1e6),
            'P_pos': P_pos,
            'P_ttc': P_ttc,
            'P_vel': P_vel,
            'P_class': P_class,
            'raw': raw
        }
        return threat, diagnostics

    def compute(self, track_id: int, ego_state, preds, kind='vehicle', class_conf=1.0, dt=0.1):
        new_threat, info = self.compute_single(ego_state, preds, kind, class_conf, dt)
        st = self.state.get(track_id, {'smoothed': 0.0, 'count_up': 0})
        prev_sm = float(st.get('smoothed', 0.0))

        # smoothing (exponential low-pass)
        alpha = float(self.cfg.get('alpha', 0.35))
        smoothed = alpha * prev_sm + (1.0 - alpha) * new_threat

        # persistence: if smoothed is increasing, increment count_up; else reset
        if smoothed > prev_sm + 1e-6:
            st['count_up'] = st.get('count_up', 0) + 1
        else:
            st['count_up'] = 0

        # only allow immediate small decreases; escalate only if persisted
        persist = int(self.cfg.get('persist_frames', 2))
        if st['count_up'] < persist and smoothed > prev_sm:
            # keep previous until persistence satisfied (avoid instant jump up)
            smoothed = prev_sm

        # finalize and store
        st['smoothed'] = float(smoothed)
        self.state[track_id] = st

        # return final smoothed threat and diagnostics
        info_out = info.copy()
        info_out.update({'smoothed': smoothed})
        return float(smoothed), info_out
