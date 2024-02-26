import numpy as np

from chromatic_tda.algorithms.miniball import MiniballAlgorithm
from chromatic_tda.core.core_chromatic_alpha_complex import CoreChromaticAlphaComplex
from chromatic_tda.utils.floating_point_utils import FloatingPointUtils
from chromatic_tda.utils.geometry_utils import GeometryUtils


class RadiusFunctionConstructor:
    @staticmethod
    def construct_radius_function(alpha_complex: CoreChromaticAlphaComplex):
        ...

    @staticmethod
    def find_smallest_circumstack_of_simplex(alpha_complex: CoreChromaticAlphaComplex, simplex: tuple):
        ...

    @staticmethod
    def find_smallest_circumstack(*point_sets: np.ndarray) -> tuple[np.ndarray, tuple[float, ...]]:
        """For arguments B_0, ..., B_k, return the center z and radii r_0, ..., r_k defining a stack of spheres
        S_0, ..., S_k passing through B_0, ..., B_k, respectively."""

        lengths = [len(point_set) for point_set in point_sets]
        points = np.concatenate(point_sets)
        z, ker = GeometryUtils.construct_equispace(*point_sets)
        points_reflected = GeometryUtils.reflect_points_through_affine_space(z, ker, *points)
        points_reflected_filtered = np.concatenate([
            refl[~FloatingPointUtils.flag_duplicates_from_reference(orig, refl)]
            for orig, refl in zip(point_sets, RadiusFunctionConstructor.split_points(points_reflected, lengths))
        ])  # filter away duplicates

        center, _ = MiniballAlgorithm.find_smallest_enclosing_ball(np.concatenate([points, points_reflected_filtered]))

        return center, tuple(np.linalg.norm(point_set[0] - center) if len(point_set) > 0 else 0.
                             for point_set in point_sets)

    @staticmethod
    def split_points(points: np.ndarray, lengths) -> list[np.ndarray, ...]:
        splits = np.zeros(len(lengths), dtype=int)
        for i, ln in enumerate(lengths):
            splits[i] = splits[i - 1] + ln

        return np.split(points, splits)[:-1]