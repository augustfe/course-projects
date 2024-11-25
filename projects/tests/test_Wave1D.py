from numpy.linalg import norm

from ..Wave1D import Wave1D


def test_pulse_bcs() -> None:
    sol = Wave1D(100, cfl=1, L0=2, c0=1)
    data = sol(100, bc=0, ic=0, save_step=100)
    assert norm(data[0] + data[100]) < 1e-12
    data = sol(100, bc=0, ic=1, save_step=100)
    assert norm(data[0] + data[100]) < 1e-12
    data = sol(100, bc=1, ic=0, save_step=100)
    assert norm(data[0] - data[100]) < 1e-12
    data = sol(100, bc=1, ic=1, save_step=100)
    assert norm(data[0] - data[100]) < 1e-12
    data = sol(100, bc=2, ic=0, save_step=100)
    assert norm(data[100]) < 1e-12
    data = sol(100, bc=2, ic=1, save_step=100)
    assert norm(data[100]) < 1e-12
    data = sol(100, bc=3, ic=0, save_step=100)
    assert norm(data[0] - data[100]) < 1e-12
    data = sol(100, bc=3, ic=1, save_step=100)
    assert norm(data[0] - data[100]) < 1e-12
