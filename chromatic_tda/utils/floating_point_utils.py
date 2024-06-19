import numpy as np
import numpy.typing as npt
import math

from chromatic_tda.utils.timing import TimingUtils


class FloatingPointUtils:

    @staticmethod
    def is_close(a, b) -> bool:
        """
        Return True if a should be considered equal to b.
        The default method used to compare whether two floats are close.
        """
        if abs(b) > .01:
            return bool(np.isclose(a, b))
        else:
            return bool(np.isclose(a, b, atol=1e-12))

    @staticmethod
    def is_all_close(avec, bvec):
        """
        Return true if lists avec and bvec are close, i.e., if all elements are close.
        """
        if len(avec) != len(bvec):
            raise ValueError("The parameters need to be of the same length.")
        return all(FloatingPointUtils.is_close(a, b) for a, b in zip(avec, bvec))

    @staticmethod
    def is_trivial_bar(bar) -> bool:
        return FloatingPointUtils.is_close(bar[0], bar[1])

    @staticmethod
    def ensure_smaller_or_equal(value: float, *coboundary_values) -> float:
        """
        Return `value` if it's smaller than the minimum of all the other passed values or close to it.
        Otherwise, return the minimum of the other passed values
        """
        minimum_above = min(coboundary_values, default=float('inf'))
        if (FloatingPointUtils.is_close(value, minimum_above) and value != minimum_above) or value > minimum_above:
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
    def flag_duplicates_from_reference(reference: npt.NDArray, to_check: npt.NDArray) -> npt.NDArray[bool]:
        """Return a boolean list of len(to_check) describing whether the corresponding element in to_check
        is close to some element of the reference."""
        TimingUtils().start("Flag Duplicates Among Vectors")
        flags = np.array([any(FloatingPointUtils.is_all_close(ref, x) for ref in reference) for x in to_check],
                         dtype=bool)
        TimingUtils().stop("Flag Duplicates Among Vectors")
        return flags
