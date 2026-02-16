# planner/mpc_casadi.py
import casadi as ca
import numpy as np


class NMPCCasadi:
    def __init__(self, horizon=14, dt=0.1):
        self.N = horizon
        self.dt = dt
        self.L = 2.5

        # Cost weights
        self.Q_pos = 10.0
        self.Q_vel = 1.0
        self.R_acc = 0.5
        self.R_steer = 0.2
        self.Q_obs = 25.0

        # Limits
        self.max_acc = 3.0
        self.max_dec = -3.0
        self.max_steer = 0.5

        # Lane
        self.LANE_HALF_WIDTH = 1.75  # meters

        self._build_solver()

    # --------------------------------------------------
    def _build_solver(self):
        N = self.N
        dt = self.dt

        # States: x, y, yaw, v
        X = ca.SX.sym("X", 4, N + 1)
        U = ca.SX.sym("U", 2, N)

        # Parameters
        X0 = ca.SX.sym("X0", 4)
        REF = ca.SX.sym("REF", 4, N)

        OBS_X = ca.SX.sym("OBS_X", 20, N)
        OBS_Y = ca.SX.sym("OBS_Y", 20, N)
        THREAT = ca.SX.sym("THREAT", 20)

        obj = 0
        g = []

        # Initial condition
        g.append(X[:, 0] - X0)

        for k in range(N):
            x = X[0, k]
            y = X[1, k]
            yaw = X[2, k]
            v = X[3, k]

            a = U[0, k]
            delta = U[1, k]

            # Vehicle model
            x_next = x + v * ca.cos(yaw) * dt
            y_next = y + v * ca.sin(yaw) * dt
            yaw_next = yaw + (v / self.L) * ca.tan(delta) * dt
            v_next = v + a * dt

            g.append(X[:, k + 1] - ca.vertcat(x_next, y_next, yaw_next, v_next))

            # Lane boundary constraints: g(x) <= 0
            g.append(y - self.LANE_HALF_WIDTH)     # y <= +1.75
            g.append(-y - self.LANE_HALF_WIDTH)    # y >= -1.75

            # Tracking cost
            pos_err = X[0:2, k] - REF[0:2, k]
            vel_err = X[3, k] - REF[3, k]

            obj += self.Q_pos * ca.dot(pos_err, pos_err)
            obj += self.Q_vel * vel_err**2
            obj += self.R_acc * a**2
            obj += self.R_steer * delta**2

            # Threat-aware obstacle cost
            for i in range(20):
                dx = X[0, k] - OBS_X[i, k]
                dy = X[1, k] - OBS_Y[i, k]
                dist = ca.sqrt(dx * dx + dy * dy + 1e-3)
                obj += self.Q_obs * THREAT[i] * ca.exp(-dist)

        g = ca.vertcat(*g)

        OPT = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1)
        )

        P = ca.vertcat(
            X0,
            ca.reshape(REF, -1, 1),
            ca.reshape(OBS_X, -1, 1),
            ca.reshape(OBS_Y, -1, 1),
            THREAT
        )

        nlp = {"x": OPT, "f": obj, "g": g, "p": P}

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0
        }

        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        self.nx = X.numel()
        self.nu = U.numel()
        self.ng = g.numel()

    # --------------------------------------------------
    def solve(self, x0, ref, preds, threats):
        N = self.N

        obs_x = np.zeros((20, N))
        obs_y = np.zeros((20, N))
        th = np.ones(20)

        for i, (oid, pred) in enumerate(preds.items()):
            if i >= 20:
                break
            for k in range(N):
                pk = pred[k]
                if isinstance(pk[0], (list, tuple, np.ndarray)):
                    px, py = pk[0]
                else:
                    px, py = pk[0], pk[1]
                obs_x[i, k] = float(px)
                obs_y[i, k] = float(py)
            th[i] = float(threats.get(oid, 1.0))

        p = np.concatenate([
            x0,
            ref.flatten(),
            obs_x.flatten(),
            obs_y.flatten(),
            th
        ])

        x_init = np.zeros(self.nx + self.nu)

        lbx = -np.inf * np.ones_like(x_init)
        ubx = np.inf * np.ones_like(x_init)

        for k in range(self.nu):
            idx = self.nx + k
            if k % 2 == 0:
                lbx[idx] = self.max_dec
                ubx[idx] = self.max_acc
            else:
                lbx[idx] = -self.max_steer
                ubx[idx] = self.max_steer

        lbg = -np.inf * np.ones(self.ng)
        ubg = np.zeros(self.ng)

        try:
            sol = self.solver(
                x0=x_init,
                p=p,
                lbx=lbx,
                ubx=ubx,
                lbg=lbg,
                ubg=ubg
            )
        except RuntimeError:
            return {"success": False}

        u = sol["x"][self.nx:self.nx + 2].full().flatten()
        return {"success": True, "u0": u}
