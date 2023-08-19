import numpy as np
from chromatic_tda.utils.singleton import singleton


@singleton
class FloatingPointUtils:

    def is_trivial_bar(self, bar):
        return np.isclose(bar[0], bar[1])

    def ensure_smaller_or_equal(self, value : float, *coboundary_values) -> float:
        """Return `value` if it's smaller than the minimum of all the other passed values or close to it.
        Otherwise return the minimum of the other passed values"""
        minimum_above = min(coboundary_values, default=float('inf'))
        if (np.isclose(value, minimum_above) and value != minimum_above) or value > minimum_above:
            return minimum_above
        else:
            return value

    def ensure_weights_monotonicity_and_equal_values(self,
                                                     weight_function : dict[tuple[int, ...], float],
                                                     co_boundary: dict[tuple[int, ...], set]) -> None:
        for simplex in sorted(weight_function, key=len, reverse=True):
            weight_function[simplex] = self.ensure_smaller_or_equal(
                weight_function[simplex],
                *(weight_function[co_face] for co_face in co_boundary[simplex])
            )


class FloatComparisonWrapper:
    def __init__(self, num):
        self.value = num

    def __lt__(self, other):
        return self.__ne__(other) and self.value < other.value

    def __gt__(self, other):
        return self.__ne__(other) and self.value > other.value

    def __eq__(self, other):
        return np.isclose(self.value, other.value)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        return self.__eq__(other) or self.value <= other.value

    def __ge__(self, other):
        return self.__eq__(other) or self.value >= other.value

