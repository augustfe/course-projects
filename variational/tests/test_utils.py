import jax.numpy as jnp
import pytest
from jax import Array

from ..utils import Interval, physical_to_reference, reference_to_physical


@pytest.mark.parametrize(
    "x, reference, physical, expected",
    [
        (
            jnp.array([0.0, 0.5, 1.0]),
            (0.0, 1.0),
            (0.0, 2.0),
            jnp.array([0.0, 1.0, 2.0]),
        ),
        (
            jnp.array([0.0, 0.5, 1.0]),
            (0.0, 1.0),
            (1.0, 2.0),
            jnp.array([1.0, 1.5, 2.0]),
        ),
        (
            jnp.array([0.0, 0.5, 1.0]),
            (0.0, 1.0),
            (-1.0, 1.0),
            jnp.array([-1.0, 0.0, 1.0]),
        ),
    ],
)
def test_reference_to_physical(
    x: Array, reference: Interval, physical: Interval, expected: Array
) -> None:
    assert jnp.allclose(reference_to_physical(x, reference, physical), expected)


@pytest.mark.parametrize(
    "x, reference, physical",
    [
        (jnp.array([0.0, 0.5, 1.0]), (0.0, 1.0), (0.0, 2.0)),
        (jnp.array([0.0, 0.5, 1.0]), (0.0, 1.0), (1.0, 2.0)),
        (jnp.array([0.0, 0.5, 1.0]), (0.0, 1.0), (-1.0, 1.0)),
    ],
)
def test_physical_to_reference_and_back(
    x: Array, reference: Interval, physical: Interval
) -> None:
    x_physical = reference_to_physical(x, reference, physical)
    x_reference = physical_to_reference(x_physical, reference, physical)
    assert jnp.allclose(x, x_reference)
