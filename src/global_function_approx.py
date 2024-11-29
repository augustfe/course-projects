import jax.numpy as jnp

from .polynomials import lagrange_polynomial

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
