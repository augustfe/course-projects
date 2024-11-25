from collections.abc import Callable

import numpy as np


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
