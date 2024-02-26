import numpy as np


class FloatingPointUtils:

    @staticmethod
    def is_trivial_bar(bar):
        return np.isclose(bar[0], bar[1])

    @staticmethod
    def ensure_smaller_or_equal(value: float, *coboundary_values) -> float:
        """
        Return `value` if it's smaller than the minimum of all the other passed values or close to it.
        Otherwise, return the minimum of the other passed values
        """
        minimum_above = min(coboundary_values, default=float('inf'))
        if (np.isclose(value, minimum_above) and value != minimum_above) or value > minimum_above:
            return minimum_above
        else:
            return value

    @staticmethod
    def ensure_weights_monotonicity_and_equal_values(weight_function: dict[tuple[int, ...], float],
                                                     co_boundary: dict[tuple[int, ...], set]) -> None:
        for simplex in sorted(weight_function, key=len, reverse=True):
            weight_function[simplex] = FloatingPointUtils.ensure_smaller_or_equal(
                weight_function[simplex],
                *(weight_function[co_face] for co_face in co_boundary[simplex])
            )

    @staticmethod
    def flag_duplicates_from_reference(reference: np.ndarray, to_check: np.ndarray) -> np.ndarray[bool, ...]:
        """Return a boolean list of len(to_check) describing whether the corresponding element in to_check
        is close to some element of the reference."""
        return np.array([any(np.isclose(ref, x).all() for ref in reference) for x in to_check], dtype=bool)
