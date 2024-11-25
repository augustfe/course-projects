# ruff: noqa: E741
import matplotlib.pyplot as plt
import numpy as np


def solver(
    I: float, a: float, T: float, dt: float, theta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Solve u'=-a*u, u(0)=I, for t in (0,T] with steps of dt."""
    dt = float(dt)  # avoid integer division
    Nt = int(round(T / dt))  # no of time intervals
    T = Nt * dt  # adjust T to fit time step dt
    u = np.zeros(Nt + 1)  # array of u[n] values
    t = np.linspace(0, T, Nt + 1)  # time mesh

    u[0] = I  # assign initial condition
    for n in range(0, Nt):  # n=0,1,...,Nt-1
        u[n + 1] = (1 - (1 - theta) * a * dt) / (1 + theta * dt * a) * u[n]
    return u, t


def u_exact(
    t: np.ndarray,
    I: float,
    a: float,
) -> np.ndarray:
    return I * np.exp(-a * t)


def plot_numerical_and_exact(
    theta: float,
    I: float,  # pylint: disable=invalid-name
    a: float,
    T: float,
    dt: float,
) -> None:
    """Compare the numerical and exact solution in a plot."""
    u, t = solver(I=I, a=a, T=T, dt=dt, theta=theta)

    t_e = np.linspace(0, T, 1001)  # fine mesh for u_e
    u_e = u_exact(t_e, I, a)

    plt.plot(
        t,
        u,
        "r--o",
        t_e,
        u_e,
        "b-",  # red dashes w/circles
    )  # blue line for exact sol.
    plt.legend(["numerical", "exact"])
    plt.xlabel("$t$")
    plt.ylabel("$u$")
    plt.title(f"theta={theta:g}, dt={dt:g}")
    plt.savefig(f"plot_{theta}_{dt:g}.png")
