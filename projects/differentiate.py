import numpy as np


def differentiate(u: np.ndarray, dt: float) -> np.ndarray:
    du = np.zeros(u.shape)
    du[0] = (u[1] - u[0]) / dt
    for i in range(1, len(u) - 1):
        du[i] = (u[i + 1] - u[i - 1]) / (2 * dt)
    du[-1] = (u[-1] - u[-2]) / dt
    return du


def differentiate_vector(u: np.ndarray, dt: float) -> np.ndarray:
    du = np.zeros(u.shape)
    du[0] = (u[1] - u[0]) / dt
    du[1:-1] = (u[2:] - u[:-2]) / (2 * dt)
    du[-1] = (u[-1] - u[-2]) / dt
    return du
