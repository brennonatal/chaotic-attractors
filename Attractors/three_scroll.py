from .base_chaotic_attractor import BaseChaoticAttractor


class ThreeScrollUnifiedAttractor(BaseChaoticAttractor):
    """Three-Scroll Unified chaotic system."""

    PARAM_NAMES = ("a", "b", "c", "d", "e", "f")
    DEFAULT_A = 32.48
    DEFAULT_B = 45.84
    DEFAULT_C = 1.18
    DEFAULT_D = 0.13
    DEFAULT_E = 0.57
    DEFAULT_F = 14.7
    DEFAULT_INITIAL_STATE = [-0.29, -0.25, -0.59]

    def next_state(self, t, state, parameters):
        a, b, c, d, e, f = parameters
        x, y, z = state
        return [
            a * (y - x) + d * x * z,
            b * x - x * z + f * y,
            c * z + x * y - e * x**2,
        ]
