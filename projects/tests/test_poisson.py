import numpy as np
import pytest
import sympy as sp

from ..poisson import Poisson, x


@pytest.mark.parametrize(
    "bc, ue_expr, f_expr, L",
    [
        ((0.0, 0.0), sp.sin(sp.pi * x), -(sp.pi**2) * sp.sin(sp.pi * x), 1.0),
        (
            (np.exp(4), np.exp(4 * np.cos(2))),
            sp.exp(4 * sp.cos(x)),
            sp.diff(sp.exp(4 * sp.cos(x)), x, 2),
            2.0,
        ),
    ],
)
def test_poisson(
    bc: tuple[float, float], ue_expr: sp.Function, f_expr: sp.Function, L: float
) -> None:
    """Test the Poisson solver"""
    Ns = [100, 200, 400, 800, 1600]
    errors = []
    sol = Poisson(L=L)
    for N in Ns:
        u = sol(N, bc=bc, f=f_expr)
        error = sol.l2_error(u, ue_expr)
        errors.append(error)

    # Compute convergence rates
    rates = [
        np.log(errors[i - 1] / errors[i]) / np.log(2) for i in range(1, len(errors))
    ]
    expected_rate = 2  # Second-order convergence

    # Assert convergence rates are close to expected_rate
    for rate in rates:
        msg = f"Convergence rate {rate} is not close to {expected_rate}"
        assert abs(rate - expected_rate) < 0.1, msg
