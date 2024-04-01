from .base_chaotic_attractor import BaseChaoticAttractor


class LorenzAttractor(BaseChaoticAttractor):
    """
    Represents the Lorenz attractor, a chaotic dynamical system.

    Inherits from BaseChaoticAttractor and implements the specific system of differential equations
    that define the Lorenz attractor.
    """

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
