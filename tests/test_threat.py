# tests/test_threat.py
from threat.prioritizer import Prioritizer
from predictor.ctrv import predict_ctrv
import numpy as np

DT = 0.1

def make_pred_for_motion(x, y, vx, vy, steps=10, dt=DT):
    # fake track dict
    tr = {'x': x, 'y': y, 'vx': vx, 'vy': vy}
    preds = predict_ctrv(tr, steps=steps, dt=dt)
    return preds

def test_pedestrian_higher_threat():
    pr = Prioritizer()
    ego = (0.0, 0.0, 0.0, 8.0)
    # pedestrian crossing close
    ped_preds = make_pred_for_motion(12.0, 2.0, -1.0, 0.0)
    veh_preds = make_pred_for_motion(30.0, 0.5, -3.0, 0.0)
    th_ped, _ = pr.compute(1, ego, ped_preds, kind='pedestrian', class_conf=1.0, dt=DT)
    th_veh, _ = pr.compute(2, ego, veh_preds, kind='vehicle', class_conf=1.0, dt=DT)
    assert th_ped >= th_veh
