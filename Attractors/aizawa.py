from .base_chaotic_attractor import BaseChaoticAttractor


class AizawaAttractor(BaseChaoticAttractor):
    """
    Represents the Aizawa (Langford) attractor, a chaotic dynamical system.

    Inherits from BaseChaoticAttractor and implements the specific system of differential equations
    that define the Aizawa attractor.
    """

    # Default parameters for the Aizawa attractor
    DEFAULT_A = 0.95
    DEFAULT_B = 0.7
    DEFAULT_C = 0.6
    DEFAULT_D = 3.5
    DEFAULT_E = 0.25
    DEFAULT_F = 0.1
    DEFAULT_INITIAL_STATE = [0.1, 1.0, 0.01]

    def __init__(
        self, initial_state=None, a=None, b=None, c=None, d=None, e=None, f=None
    ):
        """
        Initialize the Aizawa attractor with default or specified initial state and parameters.
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
        if f is None:
            f = self.DEFAULT_F
        super().__init__(initial_state, (a, b, c, d, e, f))

    def next_state(self, t, state, parameters):
        """
        Compute the next state of the Aizawa system given the current state and time.

        Args:
            t (float): Current time step (unused in Aizawa equations, but required by solve_ivp).
            state (np.ndarray): Current state of the system, [x, y, z].
            parameters (tuple): Parameters of the Aizawa system (a, b, c, d, e, f).

        Returns:
            list: The derivatives [dx/dt, dy/dt, dz/dt].
        """
        a, b, c, d, e, f = parameters
        x, y, z = state
        dxdt = (z - b) * x - d * y
        dydt = d * x + (z - b) * y
        dzdt = c + a * z - (z**3 / 3) - (x**2 + y**2) * (1 + e * z) + f * z * x**3
        return [dxdt, dydt, dzdt]
