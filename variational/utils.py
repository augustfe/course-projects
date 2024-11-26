from typing import TypeAlias

import jax.numpy as jnp
from jax import Array, lax, vmap
from quadax import quadgk

from .polynomials import ScalarFunc

Interval: TypeAlias = tuple[float, float]


def reference_to_physical(x: Array, reference: Interval, physical: Interval) -> Array:
    """Map reference to physical coordinates.

    Args:
        x (Array): Array of reference coordinates
        reference (Interval): Reference interval (A, B)
        physical (Interval): Physical interval (a, b)

    Returns:
        Array: Array of physical coordinates
    """
    A, B = reference
    a, b = physical
    return a + (b - a) / (B - A) * (x - A)


def physical_to_reference(x: Array, reference: Interval, physical: Interval) -> Array:
    """Map physical to reference coordinates.

    Args:
        x (Array): Array of physical coordinates
        reference (Interval): Reference interval (A, B)
        physical (Interval): Physical interval (a, b)

    Returns:
        Array: Array of reference coordinates
    """
    A, B = reference
    a, b = physical
    return A + (B - A) / (b - a) * (x - a)


def inner(u: ScalarFunc, v: ScalarFunc, a: float, b: float, w: ScalarFunc) -> float:
    """Return the inner product of u and v"""
    y, _ = quadgk(lambda x: u(x) * v(x) * w(x), [a, b])
    return y


def mass_matrix(
    basis_functions: list[ScalarFunc],
    reference: Interval,
    weight_func: ScalarFunc | None = None,
) -> Array:
    """Return the mass matrix

    Parameters
    ----------
    basis_functions : Polynomial
        The basis functions
    reference : Interval
        The reference domain of the basis functions

    Returns
    -------
    Array
        The mass matrix
    """
    A, B = reference
    idxs = jnp.arange(len(basis_functions))

    weight_func = weight_func if weight_func else lambda x: 1

    def matrix_element(i: int, j: int) -> float:
        return inner(
            lambda x: lax.switch(i, basis_functions, x),
            lambda x: lax.switch(j, basis_functions, x),
            A,
            B,
            weight_func,
        )

    compute_all = vmap(vmap(matrix_element, in_axes=(0, None)), in_axes=(None, 0))

    return compute_all(idxs, idxs)


if __name__ == "__main__":
    from polynomials import all_chebyshev_polynomials, all_legendre_polynomials

    n = 5
    reference = (-1.0, 1.0)
    polynomials = all_legendre_polynomials(n)

    M = mass_matrix(polynomials, reference)
    print(M)

    print(mass_matrix(all_chebyshev_polynomials(n), reference))
