"""
In this module we study the vibration equation

    u'' + w^2 u = f, t in [0, T]

where w is a constant and f(t) is a source term assumed to be 0.
We use various boundary conditions.

"""

# ruff: noqa: E741
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

t = sp.Symbol("t")


class VibSolver:
    """
    Solve vibration equation::

        u'' + w**2 u = f,

    """

    order: int

    def __init__(self, Nt: int, T: float, w: float = 0.35, I: float = 1.0) -> None:
        """
        Parameters
        ----------
        Nt : int
            Number of time steps
        T : float
            End time
        I, w : float, optional
            Model parameters
        """
        self.I = I
        self.w = w
        self.T = T
        self.set_mesh(Nt)

    def set_mesh(self, Nt: int) -> None:
        """Create mesh of chose size

        Parameters
        ----------
        Nt : int
            Number of time steps
        """
        self.Nt = Nt
        self.dt = self.T / Nt
        self.t = np.linspace(0, self.T, Nt + 1)

    def ue(self) -> sp.Expr:
        """Return exact solution as sympy function"""
        return self.I * sp.cos(self.w * t)

    def u_exact(self) -> np.ndarray:
        """Exact solution of the vibration equation

        Returns
        -------
        ue : array_like
            The solution at times n*dt
        """
        return sp.lambdify(t, self.ue())(self.t)

    def l2_error(self) -> np.ndarray:
        """Compute the l2 error norm of solver

        Returns
        -------
        float
            The l2 error norm
        """
        u = self()
        ue = self.u_exact()
        return np.sqrt(self.dt * np.sum((ue - u) ** 2))

    def convergence_rates(
        self, m: int = 4, N0: int = 32
    ) -> tuple[list[float], np.ndarray, np.ndarray]:
        """
        Compute convergence rate

        Parameters
        ----------
        m : int
            The number of mesh sizes used
        N0 : int
            Initial mesh size

        Returns
        -------
        r : array_like
            The m-1 computed orders
        E : array_like
            The m computed errors
        dt : array_like
            The m time step sizes
        """
        E = []
        dt = []
        self.set_mesh(N0)  # Set initial size of mesh
        for _ in range(m):
            self.set_mesh(self.Nt + 10)
            E.append(self.l2_error())
            dt.append(self.dt)
        r = [np.log(E[i - 1] / E[i]) / np.log(dt[i - 1] / dt[i]) for i in range(1, m)]
        return r, np.array(E), np.array(dt)

    def test_order(self, m: int = 5, N0: int = 100, tol: float = 0.1) -> None:
        r, *_ = self.convergence_rates(m, N0)
        assert np.allclose(np.array(r), self.order, atol=tol)

    def __call__(self) -> np.ndarray:
        raise NotImplementedError


class VibHPL(VibSolver):
    """
    Second order accurate recursive solver

    Boundary conditions u(0)=I and u'(0)=0
    """

    order = 2

    def __call__(self) -> np.ndarray:
        u = np.zeros(self.Nt + 1)
        u[0] = self.I
        u[1] = u[0] - 0.5 * self.dt**2 * self.w**2 * u[0]
        for n in range(1, self.Nt):
            u[n + 1] = 2 * u[n] - u[n - 1] - self.dt**2 * self.w**2 * u[n]
        return u


class VibFD2(VibSolver):
    """
    Second order accurate solver using boundary conditions::

        u(0)=I and u(T)=I

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """

    order = 2

    def __init__(self, Nt: int, T: float, w: float = 0.35, I: float = 1) -> None:
        VibSolver.__init__(self, Nt, T, w, I)
        T = T * w / np.pi
        assert T.is_integer() and T % 2 == 0

    def __call__(self) -> np.ndarray:
        b = np.zeros(self.Nt + 1)
        b[0] = self.I
        b[-1] = self.I

        g = 2 - self.dt**2 * self.w**2
        A = diags([1, -g, 1], [-1, 0, 1], (self.Nt + 1, self.Nt + 1), "lil")
        A[0, :3] = [1, 0, 0]
        A[-1, -3:] = [0, 0, 1]
        A = A.tocsr()

        u = spsolve(A, b)
        return u


class VibFD3(VibSolver):
    """
    Second order accurate solver using mixed Dirichlet and Neumann boundary
    conditions::

        u(0)=I and u'(T)=0

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """

    order = 2

    def __init__(self, Nt: int, T: float, w: float = 0.35, I: float = 1.0) -> None:
        VibSolver.__init__(self, Nt, T, w, I)
        T = T * w / np.pi
        assert T.is_integer() and T % 2 == 0

    def __call__(self) -> np.ndarray:
        b = np.zeros(self.Nt + 1)
        b[0] = self.I
        b[-1] = self.I

        g = 2 - self.dt**2 * self.w**2
        A = diags([1, -g, 1], [-1, 0, 1], (self.Nt + 1, self.Nt + 1), "lil")
        A[0, :3] = [1, 0, 0]
        A[-1, -3:] = [0, 1, 0]
        A = A.tocsr()

        u = spsolve(A, b)
        return u


class VibFD4(VibFD2):
    """
    Fourth order accurate solver using boundary conditions::

        u(0)=I and u(T)=I

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """

    order = 4

    def __call__(self) -> np.ndarray:
        u = np.zeros(self.Nt + 1)
        return u


def test_order() -> None:
    w = 0.35
    VibHPL(8, 2 * np.pi / w, w).test_order()
    VibFD2(8, 2 * np.pi / w, w).test_order()
    VibFD3(8, 2 * np.pi / w, w).test_order()
    # VibFD4(8, 2 * np.pi / w, w).test_order(N0=20)


if __name__ == "__main__":
    vib2 = VibFD2(8, 2 * np.pi / 0.35, 0.35)
    vib3 = VibFD3(8, 2 * np.pi / 0.35, 0.35)
    u2 = vib2()
    u3 = vib3()

    ue = vib2.u_exact()
    print(ue)
    plt.plot(vib2.t, ue, "b-", vib2.t, u2, "r-", vib3.t, u3, "g-")
    plt.show()

    # print(res)
    test_order()
