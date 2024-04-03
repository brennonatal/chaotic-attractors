from .base_chaotic_attractor import BaseChaoticAttractor


class RosslerAttractor(BaseChaoticAttractor):
    """
    Represents the Rössler attractor, a chaotic dynamical system.

    Inherits from BaseChaoticAttractor and implements the specific system of differential equations
    that define the Rössler attractor.
    """

    # Default parameters for the Rössler attractor
    DEFAULT_A = 0.2
    DEFAULT_B = 0.2
    DEFAULT_C = 5.7
    DEFAULT_INITIAL_STATE = [10.0, 0.0, 10.0]

    def __init__(self, initial_state=None, a=None, b=None, c=None):
        """
        Initialize the Rössler attractor with default or specified initial state and parameters.
        """
        if initial_state is None:
            initial_state = self.DEFAULT_INITIAL_STATE
        if a is None:
            a = self.DEFAULT_A
        if b is None:
            b = self.DEFAULT_B
        if c is None:
            c = self.DEFAULT_C
        super().__init__(initial_state, (a, b, c))

    def next_state(self, t, state, parameters):
        """
        Compute the next state of the Rössler system given the current state and time.

        Args:
            t (float): Current time step (unused in Rössler equations, but required by solve_ivp).
            state (np.ndarray): Current state of the system, [x, y, z].
            parameters (tuple): Parameters of the Rössler system (a, b, c).

        Returns:
            list: The derivatives [dx/dt, dy/dt, dz/dt].
        """
        a, b, c = parameters
        x, y, z = state
        dxdt = -y - z
        dydt = x + a * y
        dzdt = b + z * (x - c)
        return [dxdt, dydt, dzdt]
