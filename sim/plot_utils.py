# sim/plot_utils.py
import matplotlib.pyplot as plt


def plot_2d_scene(ego_traj, obs_traj, title):
    plt.figure(figsize=(8, 5))

    ex, ey = zip(*ego_traj)
    plt.plot(ex, ey, "b-", linewidth=2, label="Ego vehicle")

    for tid, traj in obs_traj.items():
        ox, oy = zip(*traj)
        plt.plot(ox, oy, "r--", alpha=0.6, label=f"Obstacle {tid}")

    # Lane boundaries
    plt.axhline(1.75, color="gray", linestyle="--")
    plt.axhline(-1.75, color="gray", linestyle="--")

    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.show()
