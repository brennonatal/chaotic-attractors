from .base_chaotic_attractor import BaseChaoticAttractor


class RosslerAttractor(BaseChaoticAttractor):
    """Rössler attractor."""

    PARAM_NAMES = ("a", "b", "c")
    DEFAULT_A = 0.2
    DEFAULT_B = 0.2
    DEFAULT_C = 5.7
    DEFAULT_INITIAL_STATE = [10.0, 0.0, 10.0]

    def next_state(self, t, state, parameters):
        a, b, c = parameters
        x, y, z = state
        return [-y - z, x + a * y, b + z * (x - c)]
