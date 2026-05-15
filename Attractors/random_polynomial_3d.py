import numpy as np

from .base_chaotic_attractor import BaseChaoticAttractor


# Monomial basis evaluated at (x, y, z), in this order:
#   [1, x, y, z, x², y², z², xy, xz, yz]
BASIS_LEN = 10


def _basis(x, y, z):
    return np.array([1.0, x, y, z, x * x, y * y, z * z, x * y, x * z, y * z])


class RandomPolynomial3D(BaseChaoticAttractor):
    """A 3D ODE whose derivatives are random quadratic polynomials in (x, y, z).

    Each of dx/dt, dy/dt, dz/dt is a linear combination of 10 monomials, so the
    system is fully described by a ``(3, 10)`` coefficient matrix — 30 numbers.
    A single integer ``seed`` is enough to reproduce any discovered attractor.

    Most coefficients are zeroed (Bernoulli with probability ``SPARSITY``)
    because the well-known 3D chaotic systems (Lorenz, Rössler, Chen, …) are
    sparse in the monomial basis — dense random coefficients almost always
    diverge.
    """

    PARAM_NAMES = ("coeffs",)
    DEFAULT_INITIAL_STATE = [0.1, 0.1, 0.1]
    SAMPLE_RANGE = 1.2
    SPARSITY = 0.5  # fraction of nonzero monomial coefficients

    def __init__(self, seed=None, coeffs=None, initial_state=None):
        if coeffs is None:
            rng = np.random.default_rng(seed)
            coeffs = rng.uniform(-self.SAMPLE_RANGE, self.SAMPLE_RANGE, size=(3, BASIS_LEN))
            mask = rng.random(size=(3, BASIS_LEN)) < self.SPARSITY
            coeffs = coeffs * mask
        coeffs = np.asarray(coeffs, dtype=float)
        if coeffs.shape != (3, BASIS_LEN):
            raise ValueError(f"coeffs must be shape (3, {BASIS_LEN}), got {coeffs.shape}")
        self.seed = seed
        super().__init__(initial_state=initial_state, coeffs=coeffs)

    def next_state(self, t, state, parameters):
        (coeffs,) = parameters
        x, y, z = state
        return (coeffs @ _basis(x, y, z)).tolist()
