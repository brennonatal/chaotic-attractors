from .base_chaotic_attractor import BaseChaoticAttractor


class RabinovichFabrikantAttractor(BaseChaoticAttractor):
    """
    Represents the Rabinovich-Fabrikant attractor, a chaotic dynamical system.

    Inherits from BaseChaoticAttractor and implements the specific system of differential equations
    that define the Rabinovich-Fabrikant attractor.
    """

    # Default parameters for the Rabinovich-Fabrikant attractor
    DEFAULT_ALPHA = 0.14
    DEFAULT_GAMMA = 0.10
    DEFAULT_INITIAL_STATE = [-1.0, 0.0, 0.5]

    def __init__(self, initial_state=None, alpha=None, gamma=None):
        """
        Initialize the Rabinovich-Fabrikant attractor with default or specified initial state and parameters.
        """
        if initial_state is None:
            initial_state = self.DEFAULT_INITIAL_STATE
        if alpha is None:
            alpha = self.DEFAULT_ALPHA
        if gamma is None:
            gamma = self.DEFAULT_GAMMA
        super().__init__(initial_state, (alpha, gamma))

    def next_state(self, t, state, parameters):
        """
        Compute the next state of the Rabinovich-Fabrikant system given the current state and time.

        Args:
            t (float): Current time step (unused in Rabinovich-Fabrikant equations, but required by solve_ivp).
            state (np.ndarray): Current state of the system, [x, y, z].
            parameters (tuple): Parameters of the Rabinovich-Fabrikant system (alpha, gamma).

        Returns:
            list: The derivatives [dx/dt, dy/dt, dz/dt].
        """
        alpha, gamma = parameters
        x, y, z = state
        dxdt = y * (z - 1 + x**2) + gamma * x
        dydt = x * (3 * z + 1 - x**2) + gamma * y
        dzdt = -2 * z * (alpha + x * y)
        return [dxdt, dydt, dzdt]
