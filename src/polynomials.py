from typing import Protocol

import jax.numpy as jnp
from jax import Array, vmap


class ScalarFunc(Protocol):
    def __call__(self, x: Array) -> Array: ...


def lagrange_polynomial(nodes: Array, i: int) -> ScalarFunc:
    """Generate the i-th Lagrange polynomial.

    Args:
        nodes (Array): Array of nodes
        i (int): Index of the Lagrange polynomial

    Returns:
        Callable: The i-th Lagrange polynomial
    """
    xi = nodes[i]
    others = jnp.delete(nodes, i)

    @vmap
    def p(x: Array) -> Array:
        return jnp.prod((x - others) / (xi - others))

    return p


def all_lagrange_polynomials(nodes: Array) -> list[ScalarFunc]:
    """Generate all Lagrange polynomials for the given nodes.

    Args:
        nodes (Array): Array of nodes

    Returns:
        list[Callable]: List of Lagrange polynomials
    """
    indices = jnp.arange(len(nodes))
    return [lagrange_polynomial(nodes, i) for i in indices]


def chebyshev_nodes(a: float, b: float, N: int) -> Array:
    """Return Chebyshev nodes in [a, b]"""
    k = jnp.arange(N)
    return 0.5 * (a + b) + 0.5 * (b - a) * jnp.cos((2 * k + 1) / (2 * N) * jnp.pi)


def chebyshev_polynomial(k: int) -> ScalarFunc:
    return lambda x: jnp.cos(k * jnp.arccos(x))


def all_chebyshev_polynomials(num_poly: int) -> list[ScalarFunc]:
    """Generate all Chebyshev polynomials for the given nodes.

    Args:
        nodes (Array): Array of nodes

    Returns:
        list[Callable]: List of Chebyshev polynomials
    """
    polynomials = [chebyshev_polynomial(k) for k in range(num_poly)]

    return polynomials


def all_legendre_polynomials(num_poly: int) -> list[ScalarFunc]:
    """Generate all Legendre polynomials for the given nodes.

    Args:
        nodes (Array): Array of nodes

    Returns:
        list[Callable]: List of Legendre polynomials
    """
    polynomials = [lambda x: jnp.ones_like(x), lambda x: x]

    def next_legendre_polynomial(
        p: ScalarFunc, p_prev: ScalarFunc, j: int
    ) -> ScalarFunc:
        return lambda x: ((2 * j + 1) * x * p(x) - j * p_prev(x)) / (j + 1)

    for j in range(2, num_poly):
        p, p_prev = polynomials[-1], polynomials[-2]
        polynomials.append(next_legendre_polynomial(p, p_prev, j - 1))

    return polynomials


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nodes = jnp.linspace(-1, 1, 5)
    x = jnp.linspace(-1, 1, 100)
    P = all_lagrange_polynomials(nodes)
    for i, p in enumerate(P):
        plt.plot(x, p(x), label=f"p_{i}")
        plt.scatter(nodes[i], 1, color="red")

    plt.scatter(nodes, [0] * len(nodes), color="blue")
    plt.legend()
    plt.show()

    T = all_chebyshev_polynomials(5)
    for i, t in enumerate(T):
        plt.plot(x, t(x), label=f"T_{i}")
        nodes = chebyshev_nodes(-1, 1, i)
        plt.plot(nodes, t(nodes), "o")

    plt.legend()
    plt.show()
