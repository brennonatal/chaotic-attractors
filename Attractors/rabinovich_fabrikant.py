from .base_chaotic_attractor import BaseChaoticAttractor


class RabinovichFabrikantAttractor(BaseChaoticAttractor):
    """Rabinovich-Fabrikant attractor."""

    PARAM_NAMES = ("alpha", "gamma")
    DEFAULT_ALPHA = 0.14
    DEFAULT_GAMMA = 0.10
    DEFAULT_INITIAL_STATE = [-1.0, 0.0, 0.5]

    def next_state(self, t, state, parameters):
        alpha, gamma = parameters
        x, y, z = state
        return [
            y * (z - 1 + x**2) + gamma * x,
            x * (3 * z + 1 - x**2) + gamma * y,
            -2 * z * (alpha + x * y),
        ]
