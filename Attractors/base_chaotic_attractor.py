import numpy as np
from scipy.integrate import solve_ivp


class BaseChaoticAttractor:
    """Base class for 3D chaotic attractors integrated via scipy's solve_ivp.

    Subclasses declare ``PARAM_NAMES`` (a tuple of parameter names in the order
    that ``next_state`` expects to unpack them) and ``DEFAULT_<NAME>`` class
    constants for each, plus ``DEFAULT_INITIAL_STATE``. The generic ``__init__``
    handles wiring: missing kwargs fall back to the class constants.
    """

    PARAM_NAMES: tuple[str, ...] = ()
    DEFAULT_INITIAL_STATE: list[float] = [0.0, 0.0, 0.0]

    def __init__(self, initial_state=None, **params):
        if initial_state is None:
            initial_state = self.DEFAULT_INITIAL_STATE
        self.initial_state = np.array(initial_state, dtype=float)
        self.parameters = tuple(
            params[name] if params.get(name) is not None
            else getattr(self, f"DEFAULT_{name.upper()}")
            for name in self.PARAM_NAMES
        )

    def next_state(self, t, current_state, parameters):
        """Compute derivatives. Override in subclasses."""
        raise NotImplementedError

    def generate_trajectory(self, t_span=(0, 40), t_steps=10000, **solve_kwargs):
        """Integrate the system over ``t_span`` and return a ``(3, t_steps)`` array."""
        t_eval = np.linspace(*t_span, t_steps)
        sol = solve_ivp(
            fun=self.next_state,
            t_span=t_span,
            y0=self.initial_state,
            t_eval=t_eval,
            args=(self.parameters,),
            **solve_kwargs,
        )
        return sol.y

    def generate_perturbed_trajectories(
        self,
        num_trajectories=10,
        perturbation_magnitude=0.01,
        t_span=(0, 40),
        t_steps=10000,
    ):
        """Generate trajectories from initial states perturbed around the default."""
        original = self.initial_state.copy()
        trajectories = []
        try:
            for _ in range(num_trajectories):
                self.initial_state = original + np.random.uniform(
                    -perturbation_magnitude,
                    perturbation_magnitude,
                    size=original.shape,
                )
                trajectories.append(self.generate_trajectory(t_span, t_steps))
        finally:
            self.initial_state = original
        return trajectories

    def integrate_rk4(self, total_time=30.0, dt=0.01, bound=1e6):
        """Fixed-step RK4 integration. Fast and predictable on stiff systems.

        Returns a ``(3, n_steps)`` array, or stops early if the trajectory
        diverges past ``bound`` and returns whatever was integrated so far.
        Use this for screening random candidates where speed beats precision.
        """
        n_steps = max(int(total_time / dt), 1)
        traj = np.empty((3, n_steps), dtype=float)
        y = self.initial_state.copy()
        t = 0.0
        for i in range(n_steps):
            y = self._rk4_step(t, y, dt)
            t += dt
            if not np.all(np.isfinite(y)) or np.abs(y).max() > bound:
                return traj[:, :i]
            traj[:, i] = y
        return traj

    def _rk4_step(self, t, y, dt):
        f = self.next_state
        p = self.parameters
        k1 = np.asarray(f(t, y, p))
        k2 = np.asarray(f(t + dt / 2, y + dt * k1 / 2, p))
        k3 = np.asarray(f(t + dt / 2, y + dt * k2 / 2, p))
        k4 = np.asarray(f(t + dt, y + dt * k3, p))
        return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def lyapunov_exponent(
        self,
        total_time=30.0,
        dt=0.01,
        renorm_every=50,
        d0=1e-6,
        transient_frac=0.2,
        bound=1e4,
    ):
        """Estimate the largest Lyapunov exponent via the two-orbit method.

        Integrates a reference and a perturbed copy with fixed-step RK4 in
        lock-step, renormalizes the separation every ``renorm_every`` steps,
        and accumulates ``log(d/d0)``. Fixed-step RK4 is ~100× faster than
        adaptive ``solve_ivp`` for the small, often-stiff random systems we
        screen here — and we don't need integration precision, just a stable
        estimate of the divergence rate.

        Returns ``nan`` if either trajectory diverges past ``bound``.
        """
        n_steps = max(int(total_time / dt), 1)
        skip = int(transient_frac * n_steps)
        x = self.initial_state.copy()
        x_pert = x.copy()
        x_pert[0] += d0

        log_sum = 0.0
        n_renorms = 0
        t = 0.0
        for i in range(1, n_steps + 1):
            x = self._rk4_step(t, x, dt)
            x_pert = self._rk4_step(t, x_pert, dt)
            t += dt
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(x_pert))):
                return float("nan")
            if np.abs(x).max() > bound or np.abs(x_pert).max() > bound:
                return float("nan")
            if i % renorm_every == 0:
                d = float(np.linalg.norm(x_pert - x))
                if d == 0.0 or not np.isfinite(d):
                    return float("nan")
                if i > skip:
                    log_sum += np.log(d / d0)
                    n_renorms += 1
                x_pert = x + (x_pert - x) * (d0 / d)

        if n_renorms == 0:
            return float("nan")
        return log_sum / (n_renorms * renorm_every * dt)

    def is_chaotic(self, total_time=40.0, bound=1e4, le_threshold=0.01):
        """Heuristic: bounded trajectory + positive Lyapunov exponent."""
        traj = self.generate_trajectory(
            t_span=(0, total_time), t_steps=int(total_time * 100)
        )
        if not np.all(np.isfinite(traj)) or np.abs(traj).max() > bound:
            return False
        le = self.lyapunov_exponent(total_time=total_time)
        return np.isfinite(le) and le > le_threshold
