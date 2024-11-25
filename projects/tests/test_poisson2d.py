import sympy as sp

from ..poisson2d import convergence_rates, x, y


def test_poisson2d() -> None:
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    r, *_ = convergence_rates(ue)
    assert abs(r[-1] - 2) < 1e-2
