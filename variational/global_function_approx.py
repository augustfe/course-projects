from collections.abc import Callable

import jax.numpy as jnp
from quadax import quadgk

from .polynomials import lagrange_polynomial


def inner(u: Callable, v: Callable, a: float, b: float) -> float:
    """Return the inner product of u and v"""
    y, _ = quadgk(lambda x: u(x) * v(x), [a, b])
    return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nodes = jnp.linspace(-1, 1, 5)
    x = jnp.linspace(-1, 1, 100)
    P = [lagrange_polynomial(nodes, i) for i in range(len(nodes))]
    for i, p in enumerate(P):
        plt.plot(x, p(x), label=f"p_{i}")
        plt.scatter(nodes[i], 1, color="red")

    plt.scatter(nodes, [0] * len(nodes), color="blue")
    plt.legend()
    plt.show()
