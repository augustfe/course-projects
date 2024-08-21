import numpy as np
from typing import Callable


def mesh_function(f: Callable[[np.ndarray], np.ndarray], t: np.ndarray):
    pass


def func(t: np.ndarray):
    pass


def test_mesh_function() -> None:
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)
