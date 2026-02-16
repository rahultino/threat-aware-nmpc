#!/usr/bin/env python3
"""
sim/run_demo.py
Stage 1: Minimal 2D simulator demo (ego vehicle + multiple obstacles).
Run:
    python sim/run_demo.py --case default
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import argparse

DT = 0.1
SIM_T = 12.0

class Ego:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=8.0, wheelbase=2.5):
        self.x = float(x); self.y = float(y); self.yaw = float(yaw)
        self.v = float(v); self.wheelbase = float(wheelbase)

    def state(self):
        return (self.x, self.y, self.yaw, self.v)

    def step(self, a, delta, dt=DT):
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += (self.v / self.wheelbase) * math.tan(delta) * dt
        self.v += a * dt
        if self.v < 0.0:
            self.v = 0.0

class Obstacle:
    def __init__(self, x, y, vx=0.0, vy=0.0, kind='vehicle'):
        self.x = float(x); self.y = float(y)
        self.vx = float(vx); self.vy = float(vy)
        self.kind = kind

    def pos(self):
        return (self.x, self.y)

    def step(self, dt=DT):
        self.x += self.vx * dt
        self.y += self.vy * dt

def make_scenario_case(case='default'):
    ego = Ego(x=0.0, y=0.0, yaw=0.0, v=8.0)
    obstacles = []
    if case == 'default':
        obstacles.append(Obstacle(15.0, 3.0, vx=-1.2, vy=0.0, kind='pedestrian'))
        obstacles.append(Obstacle(30.0, 0.5, vx=-12.0/3.6, vy=0.0, kind='vehicle'))
        obstacles.append(Obstacle(20.0, -2.0, vx=-5.0, vy=0.0, kind='cyclist'))
    elif case == 'single_ped':
        obstacles.append(Obstacle(18.0, 2.5, vx=-1.0, vy=0.0, kind='pedestrian'))
    elif case == 'dense':
        obstacles.extend([
            Obstacle(20.0, 0.6, vx=-10.0, vy=0.0, kind='vehicle'),
            Obstacle(22.0, -1.8, vx=-6.0, vy=0.0, kind='cyclist'),
            Obstacle(16.0, 2.5, vx=-1.2, vy=0.0, kind='pedestrian'),
            Obstacle(35.0, 1.0, vx=-14.0, vy=0.0, kind='vehicle')
        ])
    else:
        raise ValueError("unknown scenario case")
    return ego, obstacles

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def high_level_policy(ego: Ego, obs_list: list):
    safe_dist = 12.0
    emergency_dist = 5.0
    brake_dec = -3.0
    hard_brake = -6.0
    steer = 0.0
    accel = 0.0

    ego_pos = (ego.x, ego.y)
    for o in obs_list:
        ox, oy = o.pos()
        d = distance(ego_pos, (ox, oy))
        if d < emergency_dist:
            return hard_brake, 0.0
        if d < safe_dist:
            if o.kind == 'pedestrian':
                return brake_dec, 0.0
            elif o.kind == 'cyclist':
                return -2.0, 0.0
            else:
                return -1.0, 0.0
    return accel, steer

def run_sim(scenario='default', dt=DT, sim_t=SIM_T, render=True):
    ego, obstacles = make_scenario_case(scenario)
    steps = int(sim_t / dt)
    ego_hist = []
    obs_hist = [[] for _ in obstacles]
    ctrl_hist = []

    for k in range(steps):
        a, delta = high_level_policy(ego, obstacles)
        ctrl_hist.append((a, delta))
        ego.step(a, delta, dt)
        for i,o in enumerate(obstacles):
            o.step(dt)
            obs_hist[i].append((o.x, o.y))
        ego_hist.append((ego.x, ego.y, ego.yaw, ego.v))

    return ego_hist, obs_hist, ctrl_hist

def plot_results(ego_hist, obs_hist, title="Demo"):
    ego_arr = np.array(ego_hist)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,6))
    plt.plot(ego_arr[:,0], ego_arr[:,1], '-b', linewidth=2, label='Ego trajectory')
    for i,otr in enumerate(obs_hist):
        arr = np.array(otr)
        plt.plot(arr[:,0], arr[:,1], '--', linewidth=1.5, label=f'Obs {i+1}')
        plt.scatter(arr[0,0], arr[0,1], marker='x')
    plt.scatter(ego_arr[0,0], ego_arr[0,1], c='green', label='Ego start')
    plt.title(title)
    plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    plt.axis('equal'); plt.grid(True); plt.legend()
    plt.show()

def print_summary(ego_hist, obs_hist):
    ego_arr = np.array(ego_hist)
    print("Simulation summary:")
    print(f" - Ego final pos: x={ego_arr[-1,0]:.2f}, y={ego_arr[-1,1]:.2f}, v={ego_arr[-1,3]:.2f} m/s")
    for i,otr in enumerate(obs_hist):
        arr = np.array(otr)
        n = min(len(arr), len(ego_arr))
        dists = np.linalg.norm(arr[:n,:] - ego_arr[:n,:2], axis=1)
        print(f" - Obs {i+1}: min distance to ego = {dists.min():.2f} m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run minimal 2D simulator demo")
    parser.add_argument('--case', type=str, default='default', choices=['default','single_ped','dense'],
                        help='Scenario case to run')
    parser.add_argument('--dt', type=float, default=DT, help='Simulation timestep')
    parser.add_argument('--t', type=float, default=SIM_T, help='Sim length (s)')
    args = parser.parse_args()

    ego_hist, obs_hist, ctrl_hist = run_sim(scenario=args.case, dt=args.dt, sim_t=args.t)
    print_summary(ego_hist, obs_hist)
    plot_results(ego_hist, obs_hist, title=f"Sim case: {args.case}")
