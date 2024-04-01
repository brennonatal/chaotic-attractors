from .base_chaotic_attractor import BaseChaoticAttractor


class LorenzAttractor(BaseChaoticAttractor):
    """
    Represents the Lorenz attractor, a chaotic dynamical system.

    Inherits from BaseChaoticAttractor and implements the specific system of differential equations
    that define the Lorenz attractor.
    """

    DEFAULT_SIGMA = 10.0
    DEFAULT_RHO = 28.0
    DEFAULT_BETA = 8.0 / 3.0
    DEFAULT_INITIAL_STATE = [1.1, 2.0, 7.0]

    def __init__(self, initial_state=None, sigma=None, rho=None, beta=None):
        """
        Initialize the Lorenz attractor with default or specified initial state and parameters.
        """
        if initial_state is None:
            initial_state = self.DEFAULT_INITIAL_STATE
        if sigma is None:
            sigma = self.DEFAULT_SIGMA
        if rho is None:
            rho = self.DEFAULT_RHO
        if beta is None:
            beta = self.DEFAULT_BETA
        super().__init__(initial_state, (sigma, rho, beta))

    def next_state(self, t, state, parameters):
        """
        Compute the next state of the Lorenz system given the current state and time.

        Args:
            t (float): Current time step (unused in Lorenz equations, but required by solve_ivp).
            state (np.ndarray): Current state of the system, [x, y, z].
            parameters (tuple): Parameters of the Lorenz system (sigma, rho, beta).

        Returns:
            list: The derivatives [dx/dt, dy/dt, dz/dt].
        """
        sigma, rho, beta = parameters
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]
