from .base_chaotic_attractor import BaseChaoticAttractor


class ChenAttractor(BaseChaoticAttractor):
    """
    Represents the Chen attractor, a chaotic dynamical system.

    Inherits from BaseChaoticAttractor and implements the specific system of differential equations
    that define the Chen attractor.
    """

    # Default parameters for the Chen attractor
    DEFAULT_ALPHA = 5.0
    DEFAULT_BETA = -10.0
    DEFAULT_DELTA = -0.38
    DEFAULT_INITIAL_STATE = [5.0, 10.0, 10.0]

    def __init__(self, initial_state=None, alpha=None, beta=None, delta=None):
        """
        Initialize the Chen attractor with default or specified initial state and parameters.
        """
        if initial_state is None:
            initial_state = self.DEFAULT_INITIAL_STATE
        if alpha is None:
            alpha = self.DEFAULT_ALPHA
        if beta is None:
            beta = self.DEFAULT_BETA
        if delta is None:
            delta = self.DEFAULT_DELTA
        super().__init__(initial_state, (alpha, beta, delta))

    def next_state(self, t, state, parameters):
        """
        Compute the next state of the Chen system given the current state and time.

        Args:
            t (float): Current time step (unused in Chen equations, but required by solve_ivp).
            state (np.ndarray): Current state of the system, [x, y, z].
            parameters (tuple): Parameters of the Chen system (alpha, beta, delta).

        Returns:
            list: The derivatives [dx/dt, dy/dt, dz/dt].
        """
        alpha, beta, delta = parameters
        x, y, z = state
        dxdt = alpha * x - y * z
        dydt = beta * y + x * z
        dzdt = delta * z + x * y / 3
        return [dxdt, dydt, dzdt]
