import numpy as np


def is_trivial_bar(bar):
    return np.isclose(bar[0], bar[1])


def ensure_smaller_or_equal(value : float, *coboundary_values) -> float:
    """
    Return `value` if it's smaller than the minimum of all the other passed values or close to it.
    Otherwise, return the minimum of the other passed values
    """
    minimum_above = min(coboundary_values, default=float('inf'))
    if (np.isclose(value, minimum_above) and value != minimum_above) or value > minimum_above:
        return minimum_above
    else:
        return value


def ensure_weights_monotonicity_and_equal_values(weight_function : dict[tuple[int, ...], float],
                                                 co_boundary: dict[tuple[int, ...], set]) -> None:
    for simplex in sorted(weight_function, key=len, reverse=True):
        weight_function[simplex] = ensure_smaller_or_equal(
            weight_function[simplex],
            *(weight_function[co_face] for co_face in co_boundary[simplex])
        )