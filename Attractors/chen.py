from .base_chaotic_attractor import BaseChaoticAttractor


class ChenAttractor(BaseChaoticAttractor):
    """Chen attractor."""

    PARAM_NAMES = ("alpha", "beta", "delta")
    DEFAULT_ALPHA = 5.0
    DEFAULT_BETA = -10.0
    DEFAULT_DELTA = -0.38
    DEFAULT_INITIAL_STATE = [5.0, 10.0, 10.0]

    def next_state(self, t, state, parameters):
        alpha, beta, delta = parameters
        x, y, z = state
        return [alpha * x - y * z, beta * y + x * z, delta * z + x * y / 3]
