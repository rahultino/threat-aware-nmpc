# predictor/ctrv.py
"""
CTRV predictor: convert a track dict (x,y,vx,vy) to CTRV-like predictions.
Returns list of ((x_k,y_k), cov2x2) for k=1..steps.
"""

import numpy as np
import math

def vec_to_ctrv_state(track_info):
    x = float(track_info.get('x', 0.0))
    y = float(track_info.get('y', 0.0))
    vx = float(track_info.get('vx', 0.0))
    vy = float(track_info.get('vy', 0.0))
    v = math.hypot(vx, vy)
    if v > 1e-3:
        psi = math.atan2(vy, vx)
    else:
        psi = 0.0
    omega = 0.0
    return x, y, v, psi, omega

def predict_ctrv(track_info, steps=30, dt=0.1, base_var=0.05):
    x0, y0, v, psi, omega = vec_to_ctrv_state(track_info)
    preds = []
    for k in range(1, steps+1):
        t = k * dt
        if abs(omega) < 1e-6:
            xk = x0 + v * math.cos(psi) * t
            yk = y0 + v * math.sin(psi) * t
        else:
            xk = x0 + (v/omega) * (math.sin(psi + omega*t) - math.sin(psi))
            yk = y0 + (v/omega) * (-math.cos(psi + omega*t) + math.cos(psi))
        var = base_var + 0.02 * t + 0.01 * (v**2) * t
        cov = np.array([[var, 0.0],[0.0, var]])
        preds.append(((float(xk), float(yk)), cov))
    return preds

def cov_to_ellipse(cov, nsig=2.0):
    vals, vecs = np.linalg.eig(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:,order]
    width = 2 * nsig * math.sqrt(max(vals[0], 1e-9))
    height = 2 * nsig * math.sqrt(max(vals[1], 1e-9))
    angle = math.atan2(vecs[1,0], vecs[0,0])
    return width, height, angle
