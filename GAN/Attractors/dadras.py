from .base_chaotic_attractor import BaseChaoticAttractor


class DadrasAttractor(BaseChaoticAttractor):
    """
    Represents the Dadras attractor, a chaotic dynamical system.

    Inherits from BaseChaoticAttractor and implements the specific system of differential equations
    that define the Dadras attractor.
    """

    # Default parameters for the Dadras attractor
    DEFAULT_A = 3.0
    DEFAULT_B = 2.7
    DEFAULT_C = 1.7
    DEFAULT_D = 2.0
    DEFAULT_E = 9.0
    DEFAULT_INITIAL_STATE = [1.1, 2.1, -2.0]

    def __init__(self, initial_state=None, a=None, b=None, c=None, d=None, e=None):
        """
        Initialize the Dadras attractor with default or specified initial state and parameters.
        """
        if initial_state is None:
            initial_state = self.DEFAULT_INITIAL_STATE
        if a is None:
            a = self.DEFAULT_A
        if b is None:
            b = self.DEFAULT_B
        if c is None:
            c = self.DEFAULT_C
        if d is None:
            d = self.DEFAULT_D
        if e is None:
            e = self.DEFAULT_E
        super().__init__(initial_state, (a, b, c, d, e))

    def next_state(self, t, state, parameters):
        """
        Compute the next state of the Dadras system given the current state and time.

        Args:
            t (float): Current time step (unused in Dadras equations, but required by solve_ivp).
            state (np.ndarray): Current state of the system, [x, y, z].
            parameters (tuple): Parameters of the Dadras system (a, b, c, d, e).

        Returns:
            list: The derivatives [dx/dt, dy/dt, dz/dt].
        """
        a, b, c, d, e = parameters
        x, y, z = state
        dxdt = y - a * x + b * y * z
        dydt = c * y - a * x + z
        dzdt = d * x * y - e * z
        return [dxdt, dydt, dzdt]
