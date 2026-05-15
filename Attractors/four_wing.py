from .base_chaotic_attractor import BaseChaoticAttractor


class FourWingAttractor(BaseChaoticAttractor):
    """Four-Wing attractor."""

    PARAM_NAMES = ("a", "b", "c")
    DEFAULT_A = 0.2
    DEFAULT_B = 0.01
    DEFAULT_C = -0.4
    DEFAULT_INITIAL_STATE = [1.3, -0.18, 0.01]

    def next_state(self, t, state, parameters):
        a, b, c = parameters
        x, y, z = state
        return [a * x + y * z, b * x + c * y - x * z, -z - x * y]
