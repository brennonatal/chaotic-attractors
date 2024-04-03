from .base_chaotic_attractor import BaseChaoticAttractor


class FourWingAttractor(BaseChaoticAttractor):
    """
    Represents the Four-Wing attractor, a chaotic dynamical system.

    Inherits from BaseChaoticAttractor and implements the specific system of differential equations
    that define the Four-Wing attractor.
    """

    # Default parameters for the Four-Wing attractor
    DEFAULT_A = 0.2
    DEFAULT_B = 0.01
    DEFAULT_C = -0.4
    DEFAULT_INITIAL_STATE = [1.3, -0.18, 0.01]

    def __init__(self, initial_state=None, a=None, b=None, c=None):
        """
        Initialize the Four-Wing attractor with default or specified initial state and parameters.
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
        Compute the next state of the Four-Wing system given the current state and time.

        Args:
            t (float): Current time step (unused in Four-Wing equations, but required by solve_ivp).
            state (np.ndarray): Current state of the system, [x, y, z].
            parameters (tuple): Parameters of the Four-Wing system (a, b, c).

        Returns:
            list: The derivatives [dx/dt, dy/dt, dz/dt].
        """
        a, b, c = parameters
        x, y, z = state
        dxdt = a * x + y * z
        dydt = b * x + c * y - x * z
        dzdt = -z - x * y
        return [dxdt, dydt, dzdt]
