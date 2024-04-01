from .base_chaotic_attractor import BaseChaoticAttractor
import numpy as np


class ThomasAttractor(BaseChaoticAttractor):
    """
    Represents the Thomas attractor, a chaotic dynamical system.

    Inherits from BaseChaoticAttractor and implements the specific system of differential equations
    that define the Thomas attractor.
    """

    DEFAULT_B = 0.208186
    DEFAULT_INITIAL_STATE = [0.1, 0.1, 0.1]

    def __init__(self, initial_state=None, b=None):
        """
        Initialize the Thomas attractor with default or specified initial state and parameter b.
        """
        if initial_state is None:
            initial_state = self.DEFAULT_INITIAL_STATE
        if b is None:
            b = self.DEFAULT_B
        super().__init__(initial_state, (b,))

    def next_state(self, t, state, parameters):
        """
        Compute the next state of the Thomas system given the current state and time.

        Args:
            t (float): Current time step (unused in Thomas equations, but required by solve_ivp).
            state (np.ndarray): Current state of the system, [x, y, z].
            parameters (tuple): Parameters of the Thomas system (b).

        Returns:
            list: The derivatives [dx/dt, dy/dt, dz/dt].
        """
        (b,) = parameters
        x, y, z = state
        dxdt = np.sin(y) - b * x
        dydt = np.sin(z) - b * y
        dzdt = np.sin(x) - b * z
        return [dxdt, dydt, dzdt]
