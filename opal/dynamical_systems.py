"""Autonomous dynamical systems used by opal.

"duffing_eqn" and "bistable" are copied verbatim from ``src_deep_koopman/dynamical_systems.py``
(Benchmarking_Deep_Koopman), so their stored datasets regenerate bit-for-bit.

  - "duffing_eqn":           damped Duffing oscillator (2D), RK4-integrated, dt = 5e-2.
  - "bistable":              1D bistable system (Euler-discretized), dt = 5e-2.
  - "bistable_oscillations": 2D bistable limit cycles, exact flow map, dt = 5e-2.

Interface used by scripts/generate_data.py, scripts/generate_validation_data.py and
scripts/build_multiple_linear_models.py:

class SystemsAutonomous(name, *, rng=None, verbose=False)
    .n : int                 State dimension.
    .X : ndarray (n, 2)      Initial-condition box, one [low, high] row per state.
    .dt : float              Sampling period (informational only).
    .simulate_trajectory_true(x0, time_horizon) -> ndarray, shape (N, time_horizon, n)
        Integrate from each row of x0 (shape (N, n)); returns time_horizon stored
        states per trajectory, including x0 as the first one.
    .gather_traj_data(x_init=None, num_traj, traj_length, plot_results=False,
                      show_progress=False)
        -> (X, Y), each ndarray of shape (num_traj * (traj_length - 1), n)
        Sample num_traj initial conditions uniformly from x_init (default: .X) with
        the instance rng, simulate traj_length states per trajectory, and stack the
        one-step snapshot pairs trajectory by trajectory (Y[i] is the successor of X[i]).

New systems can be added to the ``systems_dict`` dictionary below.        
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def _trajectory_progress(iterable, *, total: int, desc: str = "Simulating trajectories"):
    """Yield items while updating a single-line terminal progress bar."""
    for i, item in enumerate(iterable, 1):
        pct = i / total
        width = 40
        filled = int(width * pct)
        bar = "=" * filled + "-" * (width - filled)
        end = "\n" if i == total else ""
        print(f"\r{desc}: |{bar}| {i}/{total} ({100 * pct:.1f}%)", end=end, flush=True)
        yield item


def duffing_equation_dynamics(x):
    """Damped Duffing oscillator, RK4-integrated, dt = 5e-2. x in R^{N x 2}."""
    assert len(x.shape) == 2
    N = x.shape[0]
    n = x.shape[1]
    x1 = x[:, 0]
    x2 = x[:, 1]

    dt = 5e-2

    def f1(x1, x2):
        return x2

    def f2(x1, x2):
        return -0.5 * x2 + x1 - x1**3  # Damped Duffing oscillator

    k1_x1 = dt * f1(x1, x2)
    k1_x2 = dt * f2(x1, x2)

    k2_x1 = dt * f1(x1 + 0.5 * k1_x1, x2 + 0.5 * k1_x2)
    k2_x2 = dt * f2(x1 + 0.5 * k1_x1, x2 + 0.5 * k1_x2)

    k3_x1 = dt * f1(x1 + 0.5 * k2_x1, x2 + 0.5 * k2_x2)
    k3_x2 = dt * f2(x1 + 0.5 * k2_x1, x2 + 0.5 * k2_x2)

    k4_x1 = dt * f1(x1 + k3_x1, x2 + k3_x2)
    k4_x2 = dt * f2(x1 + k3_x1, x2 + k3_x2)

    x1_next = x1 + (k1_x1 + 2 * k2_x1 + 2 * k3_x1 + k4_x1) / 6.0
    x2_next = x2 + (k1_x2 + 2 * k2_x2 + 2 * k3_x2 + k4_x2) / 6.0

    x_next = np.concatenate([x1_next[:, np.newaxis], x2_next[:, np.newaxis]], axis=1)
    assert x_next.shape == (N, n)
    return x_next


def bistable_dynamics(x):
    """
    Bistable system dynamics:
        x1_dot = a*x1 + b*x1^3
        a = 4; b = -4
        dt = 5e-2 (Euler discretization)
    Picked from Bevanda's KKR paper.
    """
    assert len(x.shape) == 2
    N = x.shape[0]
    n = x.shape[1]
    assert n == 1, "This is a 1D system."
    x1 = x[:, 0]

    a = 4
    b = -4
    dt = 5e-2
    x1_next = x1 + dt * (a * x1 + b * x1**3)
    x1_next = x1_next[:, np.newaxis]
    assert x1_next.shape == (N, n)
    return x1_next

def bistable_oscillations_dynamics(x):
    """
    Bistable oscillations dynamics (paper Sec. "Bistable Oscillations"):
        r' = (r - 2) - (r - 2)^3,   theta' = 1
        dt = 5e-2 (exact flow map, not Euler)
    Stable limit cycles at r = 1 and r = 3, unstable one at r = 2.
    With u = r - 2 the radial ODE u' = u - u^3 has the closed-form flow
        u(dt) = u e^dt / sqrt(1 + u^2 (e^{2 dt} - 1)),
    and theta advances by exactly dt. This reproduces the stored datasets to ~1e-7
    (an Euler step drifts by ~0.7). The origin is kept fixed (theta is undefined there).
    """
    assert len(x.shape) == 2
    N = x.shape[0]
    n = x.shape[1]
    assert n == 2, "This is a 2D system."

    dt = 5e-2
    r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    u = r - 2.0
    r_next = 2.0 + u * np.exp(dt) / np.sqrt(1.0 + u**2 * np.expm1(2.0 * dt))

    # Rotate by dt and rescale the radius from r to r_next.
    scale = np.divide(r_next, r, out=np.zeros_like(r), where=r > 0)
    c, s = np.cos(dt), np.sin(dt)
    x1_next = scale * (c * x[:, 0] - s * x[:, 1])
    x2_next = scale * (s * x[:, 0] + c * x[:, 1])

    x_next = np.concatenate([x1_next[:, np.newaxis], x2_next[:, np.newaxis]], axis=1)
    assert x_next.shape == (N, n)
    return x_next

########## Systems Dictionary ##################################################################

systems_dict = {
    "duffing_eqn": {
        "description": "Duffing equations",
        "X": np.array([[-2, 2], [-2, 2]]),
        "state_update": duffing_equation_dynamics,
        "n": 2,
    },
    "bistable": {
        "description": "1D Bistable system",
        "X": np.array([[-2, 2]]),
        "state_update": bistable_dynamics,
        "n": 1,
    },
    "bistable_oscillations": {
        "description": "2D Bistable oscillations (limit cycles at r = 1 and r = 3)",
        "X": np.array([[-4, 4], [-4, 4]]),
        "state_update": bistable_oscillations_dynamics,
        "n": 2,
    },
}

########### Do not modify below this line unless you know what you're doing. ###########
################################################################################################

class SystemsAutonomous:
    """Autonomous dynamical system wrapper backed by ``systems_dict``."""

    @classmethod
    def print_registry(cls) -> None:
        width = max(len(key) for key in systems_dict)
        for key, info in systems_dict.items():
            print(f"{key:<{width}}  {info['description']}")

    def __init__(self, name: str, *, rng: Optional[np.random.Generator] = None, verbose: bool = False):
        if name not in systems_dict:
            raise ValueError(f"System '{name}' not found in database.")

        self.name = name
        info = systems_dict[name]
        self.X = info["X"]
        self.state_update = info["state_update"]
        self.n = info["n"]
        self.dt = info.get("dt", 5e-2)
        self._rng = rng if rng is not None else np.random.default_rng(42)
        self.verbose = verbose
        if self.verbose:
            print(
                f"[SystemsAutonomous] Initialized '{self.name}' with state dim={self.n} "
                f"and bounds shape={self.X.shape}."
            )

    def gather_traj_data(
        self,
        x_init=None,
        num_traj=10,
        traj_length=100,
        plot_results=False,
        show_progress=False,
    ):
        """Gathers trajectory data by simulating the system from sampled initial conditions."""
        if x_init is None:
            x_init = self.X
        if self.verbose:
            print(
                f"[SystemsAutonomous] gather_traj_data(num_traj={num_traj}, "
                f"traj_length={traj_length}, plot_results={plot_results})"
            )

        X_initial_conditions = self._rng.uniform(
            low=x_init[:, 0],
            high=x_init[:, 1],
            size=(num_traj, self.n),
        )
        if self.verbose:
            print(
                f"[SystemsAutonomous] Sampled initial conditions with shape "
                f"{X_initial_conditions.shape}."
            )

        traj_data = []
        x0_iter = X_initial_conditions
        if show_progress:
            x0_iter = _trajectory_progress(x0_iter, total=num_traj)
        for x0 in x0_iter:
            x_traj = self.simulate_trajectory_true(
                x0=x0.reshape(1, -1), time_horizon=traj_length
            )[0]
            traj_data.append(x_traj)

        X = np.vstack([traj[:-1] for traj in traj_data])
        Y = np.vstack([traj[1:] for traj in traj_data])
        if self.verbose:
            print(f"[SystemsAutonomous] Built snapshot pairs: X{X.shape}, Y{Y.shape}.")

        if plot_results and self.n == 2:
            self.plot_data_scatter(X)

        return X, Y

    def generate_phase_portrait(
        self,
        x_init=None,
        num_samples_sqrt=30,
        arrow_length=0.25,
        num_traj=10,
        num_steps=500,
        plot_results=True,
    ):
        """
        Generates a 2D phase portrait (vector field) and simulates sample trajectories.
        Only defined for 2D systems.
        """
        if self.n != 2:
            raise ValueError("Phase portrait is only defined for 2D systems (n=2).")

        if x_init is None:
            x_init = self.X

        x1_vals = np.linspace(x_init[0, 0], x_init[0, 1], num_samples_sqrt)
        x2_vals = np.linspace(x_init[1, 0], x_init[1, 1], num_samples_sqrt)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        X_grid = np.vstack([X1.ravel(), X2.ravel()]).T

        Y_grid = self.state_update(X_grid)
        U = Y_grid[:, 0] - X_grid[:, 0]
        V = Y_grid[:, 1] - X_grid[:, 1]

        norms = np.sqrt(U**2 + V**2) + 1e-12
        U_plot = arrow_length * (U / norms)
        V_plot = arrow_length * (V / norms)

        if plot_results:
            plt.figure(figsize=(6, 6))
            plt.quiver(
                X_grid[:, 0], X_grid[:, 1],
                U_plot, V_plot,
                color="black",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.003,
                pivot="mid",
            )

            x0_1 = self._rng.uniform(x_init[0, 0], x_init[0, 1], size=num_traj)
            x0_2 = self._rng.uniform(x_init[1, 0], x_init[1, 1], size=num_traj)
            X0 = np.stack([x0_1, x0_2], axis=1)

            for i in range(num_traj):
                traj = np.zeros((num_steps + 1, 2))
                traj[0] = X0[i]
                xk = traj[0]
                for k in range(num_steps):
                    xk = self.state_update(xk.reshape(1, -1))[0]
                    traj[k + 1] = xk
                plt.plot(traj[:, 0], traj[:, 1], linewidth=1.5)
                plt.plot(traj[0, 0], traj[0, 1], "o", markersize=4)

            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.title("Phase Portrait with Sample Trajectories")
            plt.xlim(x_init[0, 0], x_init[0, 1])
            plt.ylim(x_init[1, 0], x_init[1, 1])
            plt.gca().set_aspect("equal", adjustable="box")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return X_grid, Y_grid

    def plot_data_scatter(self, X):
        plt.scatter(X[:, 0], X[:, 1], alpha=1)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Scatter plot of all data points X")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.show()

    def gather_data(self, x_init=None, num_samples=1000, plot_results=False):
        if x_init is None:
            x_init = self.X

        X = self._rng.uniform(low=x_init[:, 0], high=x_init[:, 1], size=(num_samples, self.n))
        assert X.shape == (num_samples, self.n)
        Y = self.state_update(X)

        if plot_results:
            self.plot_data_scatter(X)

        return X, Y

    def plot_multiple_traj(self, N=100, time_horizon=50, plt_limit=None):
        X0 = self._rng.uniform(low=self.X[:, 0], high=self.X[:, 1], size=(N, self.n))
        X_traj = self.simulate_trajectory_true(X0, time_horizon=time_horizon)

        if len(X_traj.shape) != 3:
            raise ValueError("Expected trajectory array of rank 3.")
        N = X_traj.shape[0]
        T = X_traj.shape[1]

        state_labels = [f"x{i+1}" for i in range(X_traj.shape[2])]
        n_plot = X_traj.shape[2]

        fig, axs = plt.subplots(n_plot, 1, figsize=(8, 3 * n_plot), sharex=True)
        if n_plot == 1:
            axs = [axs]
        time = np.arange(T)

        for i in range(n_plot):
            for j in range(N):
                axs[i].plot(time, X_traj[j, :, i], linewidth=1.0, alpha=0.7)
            axs[i].set_ylabel(state_labels[i])
            axs[i].set_title(f"Trajectories of {state_labels[i]} over time")
            axs[i].grid(True)
            if plt_limit is not None:
                axs[i].set_ylim(plt_limit)

        axs[-1].set_xlabel("Time")
        plt.tight_layout()
        plt.show()

    def compare_traj(self, x_traj_list, label_list, plt_limit=None):
        """Plot the i-th component of every trajectory in x_traj_list vs time, one subplot per state."""
        if len(x_traj_list) != len(label_list):
            raise ValueError("x_traj_list and label_list must have the same length.")
        n = x_traj_list[0].shape[1]
        T = x_traj_list[0].shape[0]
        for traj in x_traj_list:
            if traj.shape != (T, n):
                raise ValueError("All trajectories must have shape (T, n) with common T and n.")
        fig, axs = plt.subplots(n, 1, figsize=(8, 3 * n), sharex=True)
        if n == 1:
            axs = [axs]
        time = np.arange(T)
        for i in range(n):
            for traj, label in zip(x_traj_list, label_list):
                axs[i].plot(time, traj[:, i], label=label)
            axs[i].set_ylabel(f"x{i+1}")
            axs[i].set_title(f"x{i+1} vs Time")
            axs[i].legend()
            axs[i].grid(True)
            if plt_limit is not None:
                axs[i].set_ylim(plt_limit)
        axs[-1].set_xlabel("Time")
        plt.tight_layout()
        plt.show()

    def simulate_trajectory_true(self, x0, time_horizon=100):
        """
        Roll out the discrete state_update map one step at a time.
        Returns array of shape (N, time_horizon, n).
        """
        if x0.shape[1] != self.n:
            raise ValueError(f"x0 must have shape (N, {self.n}); got {x0.shape}.")
        N = x0.shape[0]
        if self.verbose:
            print(
                f"[SystemsAutonomous] simulate_trajectory_true (discrete) with x0 shape={x0.shape}, "
                f"time_horizon={time_horizon}."
            )
        x_traj = np.zeros((N, time_horizon, self.n))
        x_traj[:, 0, :] = x0
        for t in range(1, time_horizon):
            x_traj[:, t, :] = self.state_update(x_traj[:, t - 1, :])
        return x_traj


def systems_autonomous(
    name: Optional[str] = None, *, rng: Optional[np.random.Generator] = None
) -> Optional[SystemsAutonomous]:
    """Construct a system, or print the registry when ``name`` is None."""
    if name is None:
        SystemsAutonomous.print_registry()
        return None
    return SystemsAutonomous(name, rng=rng)


if __name__ == "__main__":
    system = SystemsAutonomous("duffing_eqn")
    system.plot_multiple_traj(N=100, time_horizon=200, plt_limit=None)