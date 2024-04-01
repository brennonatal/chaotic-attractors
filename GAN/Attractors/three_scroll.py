from .base_chaotic_attractor import BaseChaoticAttractor


class ThreeScrollUnifiedAttractor(BaseChaoticAttractor):
    """
    Represents the Three-Scroll Unified Chaotic System.

    Inherits from BaseChaoticAttractor and implements the specific system of differential equations
    that define the Three-Scroll Unified Chaotic System.
    """

    # Default parameters for the Three-Scroll Unified Chaotic System
    DEFAULT_A = 32.48
    DEFAULT_B = 45.84
    DEFAULT_C = 1.18
    DEFAULT_D = 0.13
    DEFAULT_E = 0.57
    DEFAULT_F = 14.7
    DEFAULT_INITIAL_STATE = [-0.29, -0.25, -0.59]

    def __init__(
        self, initial_state=None, a=None, b=None, c=None, d=None, e=None, f=None
    ):
        """
        Initialize the Three-Scroll Unified System with default or specified initial state and parameters.
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
        Compute the next state of the Three-Scroll Unified System given the current state and time.

        Args:
            t (float): Current time step (unused in Three-Scroll equations, but required by solve_ivp).
            state (np.ndarray): Current state of the system, [x, y, z].
            parameters (tuple): Parameters of the system (a, b, c, d, e, f).

        Returns:
            list: The derivatives [dx/dt, dy/dt, dz/dt].
        """
        a, b, c, d, e, f = parameters
        x, y, z = state
        dxdt = a * (y - x) + d * x * z
        dydt = b * x - x * z + f * y
        dzdt = c * z + x * y - e * x**2
        return [dxdt, dydt, dzdt]
