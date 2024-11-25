from collections.abc import Callable

import jax.numpy as jnp
from jax import Array, vmap


def lagrange_polynomial(nodes: Array, i: int) -> Callable:
    """Return the i-th Lagrange polynomial"""
    xi = nodes[i]
    others = jnp.delete(nodes, i)

    @vmap
    def p(x: Array) -> Array:
        return jnp.prod((x - others) / (xi - others))

    return p
