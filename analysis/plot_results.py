# analysis/plot_results.py
import json
import math
import matplotlib.pyplot as plt


# ---------- Load logs ----------
def load_log(path):
    with open(path, "r") as f:
        return json.load(f)


baseline = load_log("logs/BASELINE_0000.json")
threat = load_log("logs/THREAT_AWARE_0000.json")


# ---------- Helper: compute TTC ----------
def compute_ttc(min_dist, speed, eps=1e-3):
    """
    Approximate TTC = distance / speed
    (used consistently for baseline & threat-aware)
    """
    if speed < eps:
        return float("inf")
    return min_dist / speed


# ---------- Extract data ----------
t_base = baseline["time"]
t_th = threat["time"]

d_base = baseline["min_dist"]
d_th = threat["min_dist"]

acc_base = baseline["acc"]
acc_th = threat["acc"]

steer_base = baseline["steer"]
steer_th = threat["steer"]

speed_base = [s[3] for s in baseline["ego"]]
speed_th = [s[3] for s in threat["ego"]]

ttc_base = [compute_ttc(d, v) for d, v in zip(d_base, speed_base)]
ttc_th = [compute_ttc(d, v) for d, v in zip(d_th, speed_th)]


# =================== PLOTS ===================

# ---- 1. Minimum Distance vs Time ----
plt.figure(figsize=(9, 4))
plt.plot(t_base, d_base, "--o", label="Baseline", markersize=3)
plt.plot(t_th, d_th, "-x", label="Threat-Aware", markersize=3)
plt.xlabel("Time (s)")
plt.ylabel("Minimum Distance (m)")
plt.title("Minimum Distance vs Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ---- 2. Time-to-Collision (TTC) vs Time ----
plt.figure(figsize=(9, 4))
plt.plot(t_base, ttc_base, "--o", label="Baseline", markersize=3)
plt.plot(t_th, ttc_th, "-x", label="Threat-Aware", markersize=3)
plt.xlabel("Time (s)")
plt.ylabel("TTC (s)")
plt.title("Time-to-Collision vs Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ---- 3. Acceleration vs Time ----
plt.figure(figsize=(9, 4))
plt.plot(t_base, acc_base, "--", label="Baseline")
plt.plot(t_th, acc_th, "-", label="Threat-Aware")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration vs Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ---- 4. Steering vs Time ----
plt.figure(figsize=(9, 4))
plt.plot(t_base, steer_base, "--", label="Baseline")
plt.plot(t_th, steer_th, "-", label="Threat-Aware")
plt.xlabel("Time (s)")
plt.ylabel("Steering Angle (rad)")
plt.title("Steering vs Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# =================== SUMMARY METRICS ===================

def summary(log, ttc):
    return {
        "min_distance": min(log["min_dist"]),
        "avg_abs_acc": sum(abs(a) for a in log["acc"]) / len(log["acc"]),
        "avg_abs_steer": sum(abs(s) for s in log["steer"]) / len(log["steer"]),
        "min_ttc": min(ttc)
    }


print("\n===== SUMMARY METRICS =====")
print("Baseline NMPC:")
print(summary(baseline, ttc_base))

print("\nThreat-Aware NMPC:")
print(summary(threat, ttc_th))
