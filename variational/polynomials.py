from collections.abc import Callable

import jax.numpy as jnp
from jax import Array, vmap


def lagrange_polynomial(nodes: Array, i: int) -> Callable[[Array], Array]:
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


def all_lagrange_polynomials(nodes: Array) -> list[Callable[[Array], Array]]:
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


def chebyshev_polynomial(
    t: Callable[[Array], Array], t_prev: Callable[[Array], Array]
) -> Callable[[Array], Array]:
    """Generate the next Chebyshev polynomial.

    Args:
        t (Callable): Current Chebyshev polynomial
        t_prev (Callable): Previous Chebyshev polynomial

    Returns:
        Callable: Next Chebyshev polynomial
    """
    return lambda x: 2 * x * t(x) - t_prev(x)


def all_chebyshev_polynomials(num_poly: int) -> list[Callable[[Array], Array]]:
    """Generate all Chebyshev polynomials for the given nodes.

    Args:
        nodes (Array): Array of nodes

    Returns:
        list[Callable]: List of Chebyshev polynomials
    """
    polynomials = [lambda x: jnp.ones_like(x), lambda x: x]
    for _ in range(2, num_poly):
        polynomials.append(chebyshev_polynomial(polynomials[-1], polynomials[-2]))

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
