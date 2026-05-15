from .base_chaotic_attractor import BaseChaoticAttractor


class Lorenz83Attractor(BaseChaoticAttractor):
    """Lorenz '83 attractor."""

    PARAM_NAMES = ("a", "b", "f", "g")
    DEFAULT_A = 0.95
    DEFAULT_B = 7.91
    DEFAULT_F = 4.83
    DEFAULT_G = 4.66
    DEFAULT_INITIAL_STATE = [-0.2, -2.82, 4.66]

    def next_state(self, t, state, parameters):
        a, b, f, g = parameters
        x, y, z = state
        return [
            -a * x - y**2 - z**2 + a * f,
            -y + x * y - b * x * z + g,
            -z + b * x * y + x * z,
        ]
