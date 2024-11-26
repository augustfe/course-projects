import jax.numpy as jnp
import pytest
from jax import Array, lax

from ..polynomials import (
    all_chebyshev_polynomials,
    all_legendre_polynomials,
)
from ..utils import Interval, mass_matrix, physical_to_reference, reference_to_physical


@pytest.mark.parametrize(
    "x, reference, physical, expected",
    [
        (
            jnp.array([0.0, 0.5, 1.0]),
            (0.0, 1.0),
            (0.0, 2.0),
            jnp.array([0.0, 1.0, 2.0]),
        ),
        (
            jnp.array([0.0, 0.5, 1.0]),
            (0.0, 1.0),
            (1.0, 2.0),
            jnp.array([1.0, 1.5, 2.0]),
        ),
        (
            jnp.array([0.0, 0.5, 1.0]),
            (0.0, 1.0),
            (-1.0, 1.0),
            jnp.array([-1.0, 0.0, 1.0]),
        ),
    ],
)
def test_reference_to_physical(
    x: Array, reference: Interval, physical: Interval, expected: Array
) -> None:
    phys_x = reference_to_physical(x, reference, physical)
    assert jnp.allclose(
        phys_x, expected
    ), f"Abs diff: {jnp.abs(phys_x - expected).max()}"


@pytest.mark.parametrize(
    "x, reference, physical",
    [
        (jnp.array([0.0, 0.5, 1.0]), (0.0, 1.0), (0.0, 2.0)),
        (jnp.array([0.0, 0.5, 1.0]), (0.0, 1.0), (1.0, 2.0)),
        (jnp.array([0.0, 0.5, 1.0]), (0.0, 1.0), (-1.0, 1.0)),
    ],
)
def test_physical_to_reference_and_back(
    x: Array, reference: Interval, physical: Interval
) -> None:
    x_physical = reference_to_physical(x, reference, physical)
    x_reference = physical_to_reference(x_physical, reference, physical)
    assert jnp.allclose(x, x_reference), f"Abs diff: {jnp.abs(x - x_reference).max()}"


def test_legendre_mass_matrix() -> None:
    """Test the mass matrix for Legendre polynomials."""
    reference = (-1.0, 1.0)
    num_poly = 5
    basis_functions = all_legendre_polynomials(num_poly)
    M = mass_matrix(basis_functions, reference)
    expected = jnp.diag(2 / (2 * jnp.arange(num_poly) + 1))
    assert jnp.allclose(
        M, expected, atol=1e-7
    ), f"Abs diff: {jnp.abs(M - expected).max()}"


def test_chebyshev_mass_matrix() -> None:
    """Test the mass matrix for Chebyshev polynomials."""
    reference = (-1.0, 1.0)
    num_poly = 5
    basis_functions = all_chebyshev_polynomials(num_poly)
    M = mass_matrix(
        basis_functions, reference, weight_func=lambda x: lax.rsqrt(1 - x**2)
    )
    arr = jnp.ones(num_poly).at[0].set(2)
    expected = jnp.diag(arr / 2 * jnp.pi)

    print(jnp.abs(M - expected).max())

    assert jnp.allclose(
        M, expected, atol=1e-3
    ), f"Abs diff: {jnp.abs(M - expected).max()}"
