import jax.numpy as jnp
import pytest

from ..polynomials import lagrange_polynomial


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
