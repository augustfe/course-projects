import numpy as np

from ..VibFD import VibFD2, VibFD3, VibFD4, VibHPL


def test_order() -> None:
    w = 0.35
    VibHPL(8, 2 * np.pi / w, w).test_order()
    VibFD2(8, 2 * np.pi / w, w).test_order()
    VibFD3(8, 2 * np.pi / w, w).test_order()
    VibFD4(8, 2 * np.pi / w, w).test_order(N0=20)
