import numpy as np
import sympy as sp
from numpy.typing import ArrayLike
from scipy import sparse
from sympy.utilities.lambdify import implemented_function

x = sp.Symbol("x")


class Poisson:
    r"""Solve Poisson's equation in 1D

    .. math::

        u''(x) = f(x), \quad x \in [0, L], u(0) = a, u(L) = b,

    where a and b are numbers.

    """

    def __init__(self, L: float = 1.0, N: int | None = None) -> None:
        """Initialize the solver

        Parameters
        ----------
        L : float
            The extent of the domain, which is [0, L]
        N : int, optional
            The number of uniform intervals
        """
        self.L = L
        self.N = N
        self.ue = None
        if isinstance(N, int):
            self.create_mesh(N)

    def D2(self) -> sparse.dia_matrix:
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N + 1, self.N + 1), "lil")
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        D /= self.dx**2
        return D

    def assemble(
        self, bc: tuple[float, float] = (0.0, 0.0), f: sp.Function | None = None
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Assemble coefficient matrix and right hand side vector

        Parameters
        ----------
        bc : 2-tuple of numbers
            The boundary conditions at x=0 and x=L
        f : Sympy Function
            The right hand side as a Sympy function

        Returns
        -------
        A : scipy sparse matrix
            Coefficient matrix
        b : 1D array
            Right hand side vector
        """
        D = self.D2()
        D[0, :4] = 1, 0, 0, 0
        D[-1, -4:] = 0, 0, 0, 1
        b = np.zeros(self.N + 1)
        b[1:-1] = sp.lambdify(x, f)(self.x[1:-1])
        b[0] = bc[0]
        b[-1] = bc[1]
        return D.tocsr(), b

    def create_mesh(self, N: int) -> np.ndarray:
        """Uniform discretization of the line [0, L]

        Parameters
        ----------
        N : int
            The number of uniform intervals

        Returns
        -------
        x : array
            The mesh
        """
        self.N = N
        self.dx = self.L / N
        self.x = np.linspace(0, self.L, self.N + 1)
        return self.x

    def __call__(
        self,
        N: int,
        bc: tuple[float, float] = (0.0, 0.0),
        f: sp.Function | None = None,
    ):
        """Solve Poisson's equation

        Parameters
        ----------
        N : int
            The number of uniform intervals
        bc : 2-tuple of numbers
            The boundary conditions at x=0 and x=L
        f : Sympy Function
            The right hand side as a Sympy function

        Returns
        -------
        The solution as a Numpy array

        """
        if f is None:
            f = implemented_function("f", lambda x: 2)(x)
        self.create_mesh(N)
        A, b = self.assemble(bc=bc, f=f)
        return sparse.linalg.spsolve(A, b)

    def l2_error(self, u: ArrayLike, ue: sp.Function) -> float:
        """Return l2-error

        Parameters
        ----------
        u : array_like
            Numerical solution
        ue : Sympy function
            The analytical solution
        Returns
        -------
        The l2-error as a number

        """
        uj = sp.lambdify(x, ue)(self.x)
        return np.sqrt(self.dx * np.sum((uj - u) ** 2))


if __name__ == "__main__":
    L = 2
    sol = Poisson(L=L)
    ue = sp.exp(4 * sp.cos(x))
    # ue = x**2
    bc = (ue.subs(x, 0), ue.subs(x, L))
    u = sol(100, bc=bc, f=sp.diff(ue, x, 2))
    print("Manufactured solution: ", ue)
    print(f"Boundary conditions: u(0)={bc[0]:2.4f}, u(L)={bc[1]:2.2f}")
    print(f"Discretization: N = {sol.N}")
    print(f"L2-error {sol.l2_error(u, ue)}")
