import numpy as np
from typing import Callable


def mesh_function(f: Callable[[float], float], t: np.ndarray) -> np.ndarray:
    u = np.zeros(t.shape)
    for i, ti in enumerate(t):
        u[i] = f(ti)
    return u


def func(t: float) -> float:
    if 0 <= t <= 3:
        return np.exp(-t)
    elif 3 < t <= 4:
        return np.exp(-3 * t)
    else:
        raise ValueError("t must be in [0, 4]")


def test_mesh_function() -> None:
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)
