from typing import TypeAlias

import jax.numpy as jnp
from jax import Array, lax, vmap
from quadax import quadgk
from tqdm import tqdm, trange

from polynomials import ScalarFunc

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
    basis: str | None = None,
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
    if isinstance(basis, str):
        match basis.lower():
            case "legendre":
                idxs = jnp.arange(len(basis_functions))
                elements = 2 / (2 * idxs + 1)
                return jnp.diag(elements)
            case "chebyshev":
                elements = jnp.ones(len(basis_functions)).at[0].set(2) * jnp.pi / 2
                return jnp.diag(elements)
            case _:
                raise ValueError(f"Unknown basis {basis}")

    A, B = reference
    n = len(basis_functions)

    w = weight_func if weight_func else lambda x: 1
    M = jnp.zeros((n, n))

    total = n * (n + 1) // 2
    pbar = tqdm(total=total, desc="Computing the mass matrix")

    for i in range(n):
        for j in range(i, n):
            res = inner(basis_functions[i], basis_functions[j], A, B, w)
            M = M.at[i, j].set(res)
            M = M.at[j, i].set(res)
            pbar.update(1)

    return M


def b_vector(
    basis_functions: list[ScalarFunc],
    target_func: ScalarFunc,
    reference: Interval,
    weight_func: ScalarFunc | None = None,
) -> Array:
    A, B = reference
    idxs = jnp.arange(len(basis_functions))

    w = weight_func if weight_func else lambda x: 1

    def vec_element(i: int) -> float:
        return inner(basis_functions[i], target_func, A, B, w)

    b = jnp.zeros(len(basis_functions))
    for i in trange(len(basis_functions)):
        b = b.at[i].set(vec_element(i))

    return b


if __name__ == "__main__":
    from polynomials import all_chebyshev_polynomials, all_legendre_polynomials

    n = 5
    reference = (-1.0, 1.0)
    polynomials = all_legendre_polynomials(n)

    M = mass_matrix(polynomials, reference)
    print(M)

    print(mass_matrix(all_chebyshev_polynomials(n), reference))
