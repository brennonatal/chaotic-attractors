from .base_chaotic_attractor import BaseChaoticAttractor


class SprottAttractor(BaseChaoticAttractor):
    """Sprott attractor."""

    PARAM_NAMES = ("a", "b")
    DEFAULT_A = 2.07
    DEFAULT_B = 1.79
    DEFAULT_INITIAL_STATE = [0.63, 0.47, -0.54]

    def next_state(self, t, state, parameters):
        a, b = parameters
        x, y, z = state
        return [y + a * y * z + x * z, 1 - b**2 * x + y * z, x - x**2 - y**2]
