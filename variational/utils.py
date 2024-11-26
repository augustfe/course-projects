from typing import TypeAlias

from jax import Array

Interval: TypeAlias = tuple[float, float]


def reference_to_physical(x: Array, reference: Interval, physical: Interval) -> Array:
    """Map reference to physical coordinates.

    Args:
        x (Array): Array of reference coordinates
        reference (Interval): Reference interval (A, B)
        physical (Interval): Physical interval (a, b)

    Returns:
        Array: Array of physical coordinates
    """
    A, B = reference
    a, b = physical
    return a + (b - a) / (B - A) * (x - A)


def physical_to_reference(x: Array, reference: Interval, physical: Interval) -> Array:
    """Map physical to reference coordinates.

    Args:
        x (Array): Array of physical coordinates
        reference (Interval): Reference interval (A, B)
        physical (Interval): Physical interval (a, b)

    Returns:
        Array: Array of reference coordinates
    """
    A, B = reference
    a, b = physical
    return A + (B - A) / (b - a) * (x - a)
