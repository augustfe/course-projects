import numpy as np
import sympy as sp
from scipy import sparse

from .poisson import Poisson

x, y = sp.symbols("x,y")


class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, Lx] x [0, Ly]

    with homogeneous Dirichlet boundary conditions.
    """

    def __init__(self, Lx: float, Ly: float, Nx: int, Ny: int, ue: sp.Function) -> None:
        """Initialize the solver

        Parameters
        ----------
        Lx : float
            The extent of the domain in x-direction, which is [0, Lx]

        Ly : float
            The extent of the domain in y-direction, which is [0, Ly]

        Nx : int
            The number of uniform intervals in x-direction

        Ny : int
            The number of uniform intervals in y-direction
        """
        self.px = Poisson(Lx, Nx)  # we can reuse some of the code from the 1D case
        self.py = Poisson(Ly, Ny)

        self.ue = ue
        self.create_mesh()

    def create_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Create the 2D mesh

        Returns
        -------
        x, y : tuple of arrays
            The 2D mesh, in sparse format
        """
        self.x, self.y = np.meshgrid(self.px.x, self.py.x, indexing="ij", sparse=True)

        self.dx, self.dy = self.px.dx, self.py.dx
        self.Nx, self.Ny = self.px.N, self.py.N

        return self.x, self.y

    @staticmethod
    def D2(N: int) -> sparse.dia_matrix:
        """Create second order differentiation matrix with Dirichlet conditions.

        Returns
        -------
        The differentiation matrix as a sparse diagonal matrix
        """
        D = sparse.diags_array(
            [1, -2, 1],
            offsets=[-1, 0, 1],
            shape=(N + 1, N + 1),
            format="lil",
        )
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2

        return D

    @staticmethod
    def identity(N: int) -> sparse.dia_matrix:
        """Return identity matrix"""
        return sparse.eye_array(N + 1)

    def laplace(self) -> sparse.csr_matrix:
        """Return a vectorized Laplace operator"""

        D2x = 1.0 / self.dx**2 * sparse.kron(self.D2(self.Nx), self.identity(self.Ny))
        D2y = 1.0 / self.dy**2 * sparse.kron(self.identity(self.Nx), self.D2(self.Ny))

        laplace = D2x + D2y

        return laplace.tocsr()

    def assemble(self, f: sp.Function) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Assemble coefficient matrix A and right hand side vector b

        Parameters
        ----------
        f : Sympy function
            The right hand side function f(x, y)

        Returns
        -------
        A : sparse.csr_matrix
            The coefficient matrix
        """
        bnds = self.get_boundary_indices()

        A = self.laplace().tolil()
        A[bnds] = 0
        A[bnds, bnds] = 1
        A = A.tocsr()

        F = self.mesh_function(f)(self.x, self.y)
        print(F)
        b = F.ravel()

        b[bnds] = sp.lambdify((x, y), self.ue)(self.x, self.y).ravel()[bnds]

        return A, b

    def mesh_function(self, f: sp.Expr) -> sp.Function:
        """Return the mesh function f(x, y)"""
        return sp.lambdify((x, y), f, "numpy")

    def get_boundary_indices(self) -> np.ndarray:
        """Return indices of vectorized matrix that belongs to the boundary"""

        toprow = np.arange(self.Nx + 1)
        bottomrow = np.arange(self.Nx + 1) + self.Ny * (self.Nx + 1)

        leftcol = np.arange(0, self.Nx * self.Ny + 1, self.Ny + 1)
        rightcol = np.arange(self.Nx, self.Nx * self.Ny + 1, self.Ny + 1)

        bnds = np.concatenate([toprow, bottomrow, leftcol, rightcol])

        return bnds

    def l2_error(self, u: np.ndarray, ue: sp.Function) -> float:
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy function
            The analytical solution
        """
        u_exact = sp.lambdify((x, y), ue)(self.x, self.y)

        error = np.sqrt(self.dx * self.dy * np.sum((u - u_exact) ** 2))
        return error

    def __call__(self, f: sp.Function | None = None) -> np.ndarray:
        """Solve Poisson's equation with a given righ hand side function

        Parameters
        ----------
        N : int
            The number of uniform intervals
        f : Sympy function
            The right hand side function f(x, y)

        Returns
        -------
        The solution as a Numpy array

        """
        if f is None:
            f = sp.diff(self.ue, x, 2) + sp.diff(self.ue, y, 2)
        A, b = self.assemble(f=f)

        return sparse.linalg.spsolve(A, b).reshape((self.Nx + 1, self.Ny + 1))


def convergence_rates(
    ue: sp.Function, m: int = 6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute convergence rates for a range of discretizations

    Parameters
    ----------
    m : int
        The number of discretization levels to use

    Returns
    -------
    3-tuple of arrays. The arrays represent:
        0: the orders
        1: the l2-errors
        2: the mesh sizes
    """
    E = np.zeros(m)
    h = np.zeros(m)
    N0 = 8
    for i in range(m):
        sol = Poisson2D(1, 1, N0, N0, ue)
        u = sol()
        E[i] = sol.l2_error(u, ue)
        h[i] = sol.dx
        N0 *= 2

    error = E[:-1] / np.roll(E, -1)[:-1]
    steps = h[:-1] / np.roll(h, -1)[:-1]
    r = np.log(error) / np.log(steps)

    return r, E, h
