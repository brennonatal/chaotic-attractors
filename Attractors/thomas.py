import numpy as np

from .base_chaotic_attractor import BaseChaoticAttractor


class ThomasAttractor(BaseChaoticAttractor):
    """Thomas attractor."""

    PARAM_NAMES = ("b",)
    DEFAULT_B = 0.208186
    DEFAULT_INITIAL_STATE = [0.1, 0.1, 0.1]

    def next_state(self, t, state, parameters):
        (b,) = parameters
        x, y, z = state
        return [np.sin(y) - b * x, np.sin(z) - b * y, np.sin(x) - b * z]
