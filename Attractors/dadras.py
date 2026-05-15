from .base_chaotic_attractor import BaseChaoticAttractor


class DadrasAttractor(BaseChaoticAttractor):
    """Dadras attractor."""

    PARAM_NAMES = ("a", "b", "c", "d", "e")
    DEFAULT_A = 3.0
    DEFAULT_B = 2.7
    DEFAULT_C = 1.7
    DEFAULT_D = 2.0
    DEFAULT_E = 9.0
    DEFAULT_INITIAL_STATE = [1.1, 2.1, -2.0]

    def next_state(self, t, state, parameters):
        a, b, c, d, e = parameters
        x, y, z = state
        return [y - a * x + b * y * z, c * y - a * x + z, d * x * y - e * z]
