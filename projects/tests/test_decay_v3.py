import numpy as np

from ..decay_v3 import solver


def test_solver_three_steps() -> None:
    """Compare three steps with known manual computations."""
    theta = 0.8
    a = 2
    initial = 0.1
    dt = 0.8
    u_by_hand = np.array([initial, 0.0298245614035, 0.00889504462912, 0.00265290804728])

    Nt = 3  # number of time steps
    u, _ = solver(I=initial, a=a, T=Nt * dt, dt=dt, theta=theta)

    tol = 1e-12  # tolerance for comparing floats
    diff = abs(u - u_by_hand).max()
    success = diff < tol
    assert success
