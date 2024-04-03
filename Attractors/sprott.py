from .base_chaotic_attractor import BaseChaoticAttractor


class SprottAttractor(BaseChaoticAttractor):
    """
    Represents the Sprott attractor, a chaotic dynamical system.

    Inherits from BaseChaoticAttractor and implements the specific system of differential equations
    that define the Sprott attractor.
    """

    # Default parameters for the Sprott attractor
    DEFAULT_A = 2.07
    DEFAULT_B = 1.79
    DEFAULT_INITIAL_STATE = [0.63, 0.47, -0.54]

    def __init__(self, initial_state=None, a=None, b=None):
        """
        Initialize the Sprott attractor with default or specified initial state and parameters.
        """
        if initial_state is None:
            initial_state = self.DEFAULT_INITIAL_STATE
        if a is None:
            a = self.DEFAULT_A
        if b is None:
            b = self.DEFAULT_B
        super().__init__(initial_state, (a, b))

    def next_state(self, t, state, parameters):
        """
        Compute the next state of the Sprott system given the current state and time.

        Args:
            t (float): Current time step (unused in Sprott equations, but required by solve_ivp).
            state (np.ndarray): Current state of the system, [x, y, z].
            parameters (tuple): Parameters of the Sprott system (a, b).

        Returns:
            list: The derivatives [dx/dt, dy/dt, dz/dt].
        """
        a, b = parameters
        x, y, z = state
        dxdt = y + a * y * z + x * z
        dydt = 1 - b**2 * x + y * z
        dzdt = x - x**2 - y**2
        return [dxdt, dydt, dzdt]
