import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from tqdm import tqdm

from polynomials import (
    ScalarFunc,
    all_chebyshev_polynomials,
    all_legendre_polynomials,
    chebyshev_nodes,
)
from utils import (
    Interval,
    b_vector,
    mass_matrix,
    physical_to_reference,
    reference_to_physical,
)


def final_func(
    polynomials: list[ScalarFunc],
    coefficients: jnp.ndarray,
    ref_domain: Interval = (-1, 1),
    domain: Interval = (-1, 1),
) -> ScalarFunc:
    """Return the final function."""

    def f(x: jnp.ndarray) -> jnp.ndarray:
        x_ref = physical_to_reference(x, ref_domain, domain)
        values = jnp.array([p(x_ref) for p in polynomials])
        return jnp.dot(coefficients, values)

    return f


def global_func_approx(
    target: ScalarFunc, name: str, domain: Interval, basis: str
) -> None:
    A, B = domain

    N = (2, 3, 5, 10, 20, 30)
    if basis == "chebyshev":
        all_polynomials = all_chebyshev_polynomials(N[-1])
        ref_domain = (-1, 1)
        weight_func = lambda x: 1 / jnp.sqrt(1 - x**2)  # noqa: E731
    elif basis == "legendre":
        all_polynomials = all_legendre_polynomials(N[-1])
        ref_domain = (-1, 1)
        weight_func = None
    else:
        raise ValueError("basis must be either 'chebyshev' or 'legendre'")

    def target_reference(x: Array) -> Array:
        return target(reference_to_physical(x, ref_domain, domain))

    all_b = b_vector(all_polynomials, target_reference, ref_domain, weight_func)
    x = jnp.linspace(A, B, 100)
    for n_poly in N:
        polynomials = all_polynomials[:n_poly]
        b = all_b[:n_poly]
        mass = mass_matrix(polynomials, ref_domain, basis=basis)

        coeffs = jnp.linalg.solve(mass, b)

        f = final_func(polynomials, coeffs, ref_domain, domain)
        plt.plot(x, f(x), label=f"$n = {n_poly}$")

    plt.plot(x, target(x), label=name, linestyle="--")
    plt.legend()
    plt.show()


def approx(basis: str) -> None:
    targets = [
        lambda x: jnp.abs(x),
        lambda x: jnp.exp(jnp.sin(x)),
        lambda x: x**10,
        lambda x: jnp.exp(-((x - 0.5) ** 2)) - jnp.exp(-0.25),
    ]
    names = [
        "$|x|$",
        "$e^{sin(x)}$",
        "$x^{10}$",
        "$e^{-(x - 0.5)^2} - e^{-0.25}$",
    ]
    domains = [
        (-1, 1),
        (0, 2),
        (0, 1),
        (0, 1),
    ]
    for target, name, domain in zip(targets, names, domains, strict=True):
        global_func_approx(target, name, domain, basis)


def lecture_08() -> None:
    approx("legendre")


def lecture_09() -> None:
    approx("chebyshev")


if __name__ == "__main__":
    # lecture_08()
    lecture_09()
