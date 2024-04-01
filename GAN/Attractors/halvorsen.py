from .base_chaotic_attractor import BaseChaoticAttractor


class HalvorsenAttractor(BaseChaoticAttractor):
    """
    Represents the Halvorsen attractor, a chaotic dynamical system.

    Inherits from BaseChaoticAttractor and implements the specific system of differential equations
    that define the Halvorsen attractor.
    """

    # Default parameter for the Halvorsen attractor
    DEFAULT_A = 1.89
    DEFAULT_INITIAL_STATE = [-1.48, -1.51, 2.04]

    def __init__(self, initial_state=None, a=None):
        """
        Initialize the Halvorsen attractor with default or specified initial state and parameter.
        """
        if initial_state is None:
            initial_state = self.DEFAULT_INITIAL_STATE
        if a is None:
            a = self.DEFAULT_A
        super().__init__(initial_state, (a,))

    def next_state(self, t, state, parameters):
        """
        Compute the next state of the Halvorsen system given the current state and time.

        Args:
            t (float): Current time step (unused in Halvorsen equations, but required by solve_ivp).
            state (np.ndarray): Current state of the system, [x, y, z].
            parameters (tuple): Parameter of the Halvorsen system (a).

        Returns:
            list: The derivatives [dx/dt, dy/dt, dz/dt].
        """
        (a,) = parameters
        x, y, z = state
        dxdt = -a * x - 4 * y - 4 * z - y**2
        dydt = -a * y - 4 * z - 4 * x - z**2
        dzdt = -a * z - 4 * x - 4 * y - x**2
        return [dxdt, dydt, dzdt]
