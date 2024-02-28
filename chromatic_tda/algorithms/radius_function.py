import numpy as np

from chromatic_tda.algorithms.chromatic_subcomplex_utils import ChromaticComplexUtils
from chromatic_tda.algorithms.miniball_interface import MiniballAlgorithm
from chromatic_tda.core.core_chromatic_alpha_complex import CoreChromaticAlphaComplex
from chromatic_tda.utils.floating_point_utils import FloatingPointUtils
from chromatic_tda.utils.geometry_utils import GeometryUtils


class RadiusFunctionConstructor:
    @staticmethod
    def construct_radius_function(alpha_complex: CoreChromaticAlphaComplex) -> dict[tuple[int, ...], float]:
        radius_function = {}
        for dim in range(alpha_complex.simplicial_complex.dimension, 0, -1):
            simplices: list[tuple[int, ...]] = alpha_complex.simplicial_complex.get_simplices_of_dim(dim)
            for simplex in simplices:
                center, radii = RadiusFunctionConstructor.find_smallest_circumstack_of_simplex(alpha_complex, simplex)
                extra_vertices = alpha_complex.simplicial_complex.get_extra_vertices_of_cofaces(simplex)
                if RadiusFunctionConstructor.is_stack_empty_of_vertices(alpha_complex, extra_vertices, center, radii):
                    radius_function[simplex] = max(radii.values())
                else:
                    co_faces = alpha_complex.simplicial_complex.co_boundary[simplex]
                    radius_function[simplex] = min(radius_function[co_face] for co_face in co_faces)
        for simplex in alpha_complex.simplicial_complex.get_simplices_of_dim(0):
            radius_function[simplex] = 0.

        return radius_function

    @staticmethod
    def is_stack_empty_of_vertices(alpha_complex: CoreChromaticAlphaComplex, vertices, center: np.ndarray, radii: dict)\
            -> bool:
        """Return True if the distance from center is greater or close to the corresponding color radius
        for all given vertices."""
        for v in vertices:
            distance = np.linalg.norm(alpha_complex.points[v] - center)
            radius = radii.get(alpha_complex.internal_labeling[v], 0)
            if distance < radius and not np.isclose(distance, radius):
                return False
        return True

    @staticmethod
    def find_smallest_circumstack_of_simplex(alpha_complex: CoreChromaticAlphaComplex, simplex: tuple)\
            -> tuple[np.ndarray, dict]:
        split = ChromaticComplexUtils.split_simplex_by_labels(simplex, alpha_complex.internal_labeling)
        labels, vertex_sets = zip(*split.items())
        point_sets = [alpha_complex.points[vertex_set] for vertex_set in vertex_sets]
        center, radii = RadiusFunctionConstructor.find_smallest_circumstack(*point_sets)
        return center, {lab: rad for lab, rad in zip(labels, radii)}

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
        radii = tuple(np.linalg.norm(point_set[0] - center) if len(point_set) > 0 else 0. for point_set in point_sets)

        return center, radii

    @staticmethod
    def split_points(points: np.ndarray, lengths) -> list[np.ndarray, ...]:
        splits = np.zeros(len(lengths), dtype=int)
        for i, ln in enumerate(lengths):
            splits[i] = splits[i - 1] + ln

        return np.split(points, splits)[:-1]