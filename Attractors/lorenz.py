from .base_chaotic_attractor import BaseChaoticAttractor


class LorenzAttractor(BaseChaoticAttractor):
    """Lorenz '63 attractor (σ, ρ, β)."""

    PARAM_NAMES = ("sigma", "rho", "beta")
    DEFAULT_SIGMA = 10.0
    DEFAULT_RHO = 28.0
    DEFAULT_BETA = 8.0 / 3.0
    DEFAULT_INITIAL_STATE = [1.1, 2.0, 7.0]

    def next_state(self, t, state, parameters):
        sigma, rho, beta = parameters
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
