from .base_chaotic_attractor import BaseChaoticAttractor


class HalvorsenAttractor(BaseChaoticAttractor):
    """Halvorsen attractor."""

    PARAM_NAMES = ("a",)
    DEFAULT_A = 1.89
    DEFAULT_INITIAL_STATE = [-1.48, -1.51, 2.04]

    def next_state(self, t, state, parameters):
        (a,) = parameters
        x, y, z = state
        return [
            -a * x - 4 * y - 4 * z - y**2,
            -a * y - 4 * z - 4 * x - z**2,
            -a * z - 4 * x - 4 * y - x**2,
        ]
