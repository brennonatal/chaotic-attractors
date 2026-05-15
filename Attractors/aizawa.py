from .base_chaotic_attractor import BaseChaoticAttractor


class AizawaAttractor(BaseChaoticAttractor):
    """Aizawa (Langford) attractor."""

    PARAM_NAMES = ("a", "b", "c", "d", "e", "f")
    DEFAULT_A = 0.95
    DEFAULT_B = 0.7
    DEFAULT_C = 0.6
    DEFAULT_D = 3.5
    DEFAULT_E = 0.25
    DEFAULT_F = 0.1
    DEFAULT_INITIAL_STATE = [0.1, 1.0, 0.01]

    def next_state(self, t, state, parameters):
        a, b, c, d, e, f = parameters
        x, y, z = state
        return [
            (z - b) * x - d * y,
            d * x + (z - b) * y,
            c + a * z - (z**3 / 3) - (x**2 + y**2) * (1 + e * z) + f * z * x**3,
        ]
