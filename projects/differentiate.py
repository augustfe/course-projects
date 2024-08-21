import numpy as np


def differentiate(u: np.ndarray, dt: float) -> np.ndarray:
    pass


def differentiate_vector(u: np.ndarray, dt: float) -> np.ndarray:
    pass


def test_differentiate() -> None:
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    assert np.allclose(du1, du2)


if __name__ == "__main__":
    test_differentiate()
