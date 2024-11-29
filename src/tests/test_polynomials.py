import jax.numpy as jnp
import pytest

from ..polynomials import (
    ScalarFunc,
    all_chebyshev_polynomials,
    all_legendre_polynomials,
    chebyshev_nodes,
    lagrange_polynomial,
)


@pytest.mark.parametrize(
    "nodes, i",
    [
        ([0, 1, 2, 3], 0),
        ([0, 1, 2, 3], 1),
        ([0, 1, 2, 3], 2),
        ([0, 1, 2, 3], 3),
    ],
)
def test_lagrange_is_kronecker_delta(nodes: list[int], i: int) -> None:
    """Test that the Lagrange polynomial is the Kronecker delta at nodes."""
    nodes = jnp.array(nodes)
    p = lagrange_polynomial(nodes, i)
    expected = jnp.eye(len(nodes))[i]

    assert jnp.allclose(p(nodes), expected)


@pytest.mark.parametrize(
    "nodes",
    [
        [-4, -3, -2, -1, 0, 1, 2, 3, 4],
        [-1, 0, 1],
        [-1, 0, 1, 2],
        [-1, 0, 1, 2, 3],
    ],
)
def test_lagrange_polynomials_sum_to_one(nodes: list[int]) -> None:
    """Test that the Lagrange polynomials sum to one."""
    nodes = jnp.array(nodes)
    P = [lagrange_polynomial(nodes, i) for i in range(len(nodes))]
    x = jnp.linspace(nodes[0] - 1, nodes[-1] + 1, 100)
    sum_P = jnp.sum(jnp.array([p(x) for p in P]), axis=0)

    assert jnp.allclose(sum_P, 1.0, atol=1e-4)


@pytest.mark.parametrize("num_nodes", [3, 4, 5, 6])
def test_chebyshev_nodes_in_range(num_nodes: int) -> None:
    """Test that the Chebyshev nodes are in the interval [-1, 1]."""
    nodes = chebyshev_nodes(-1, 1, num_nodes)
    assert jnp.all(nodes >= -1) and jnp.all(nodes <= 1)


@pytest.mark.parametrize("max_nodes", [10])
def test_chebyshev_nodes_are_roots(max_nodes: int) -> None:
    """Test that the Chebyshev nodes are zeros."""
    polynomials = all_chebyshev_polynomials(max_nodes)

    for i, t in enumerate(polynomials[1:], start=1):
        nodes = chebyshev_nodes(-1, 1, i)
        assert jnp.allclose(t(nodes), 0.0, atol=1e-4)


@pytest.mark.parametrize(
    "j, expected",
    [
        (0, lambda x: jnp.ones_like(x)),
        (1, lambda x: x),
        (2, lambda x: 0.5 * (3 * x**2 - 1)),
        (3, lambda x: 0.5 * (5 * x**3 - 3 * x)),
        (4, lambda x: 0.125 * (35 * x**4 - 30 * x**2 + 3)),
    ],
)
def test_legendre_polynomials(j: int, expected: ScalarFunc) -> None:
    """Test that the Legendre polynomials are correct."""

    polynomials = all_legendre_polynomials(j + 1)
    polynomial = polynomials[j]
    x = jnp.linspace(-1, 1, 100)

    assert jnp.allclose(polynomial(x), expected(x), atol=1e-4)
