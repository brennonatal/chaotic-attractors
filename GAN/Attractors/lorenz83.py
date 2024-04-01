from .base_chaotic_attractor import BaseChaoticAttractor


class Lorenz83Attractor(BaseChaoticAttractor):
    """
    Represents the Lorenz '83 attractor, a chaotic dynamical system.

    Inherits from BaseChaoticAttractor and implements the specific system of differential equations
    that define the Lorenz '83 attractor.
    """

    # Default parameters for the Lorenz '83 attractor
    DEFAULT_A = 0.95
    DEFAULT_B = 7.91
    DEFAULT_F = 4.83
    DEFAULT_G = 4.66
    DEFAULT_INITIAL_STATE = [-0.2, -2.82, 4.66]

    def __init__(self, initial_state=None, a=None, b=None, f=None, g=None):
        """
        Initialize the Lorenz '83 attractor with default or specified initial state and parameters.
        """
        if initial_state is None:
            initial_state = self.DEFAULT_INITIAL_STATE
        if a is None:
            a = self.DEFAULT_A
        if b is None:
            b = self.DEFAULT_B
        if f is None:
            f = self.DEFAULT_F
        if g is None:
            g = self.DEFAULT_G
        super().__init__(initial_state, (a, b, f, g))

    def next_state(self, t, state, parameters):
        """
        Compute the next state of the Lorenz '83 system given the current state and time.

        Args:
            t (float): Current time step (unused in Lorenz '83 equations, but required by solve_ivp).
            state (np.ndarray): Current state of the system, [x, y, z].
            parameters (tuple): Parameters of the Lorenz '83 system (a, b, f, g).

        Returns:
            list: The derivatives [dx/dt, dy/dt, dz/dt].
        """
        a, b, f, g = parameters
        x, y, z = state
        dxdt = -a * x - y**2 - z**2 + a * f
        dydt = -y + x * y - b * x * z + g
        dzdt = -z + b * x * y + x * z
        return [dxdt, dydt, dzdt]
