import itertools
from typing import Optional

import numpy as np
import numpy.typing as npt
from collections.abc import Iterable

from chromatic_tda.algorithms.chromatic_subcomplex_utils import ChromaticComplexUtils
from chromatic_tda.core.core_chromatic_alpha_complex import CoreChromaticAlphaComplex
from chromatic_tda.core.stack import StackOfSpheres
from chromatic_tda.utils.floating_point_utils import FloatingPointUtils
from chromatic_tda.utils.geometry_utils import GeometryUtils
from chromatic_tda.utils.timing import TimingUtils


class RadiusFunctionConstructor:
    @staticmethod
    def construct_sq_radius_function(alpha_complex: CoreChromaticAlphaComplex,
                                     use_morse_optimization: bool) -> dict[tuple[int, ...], float]:
        TimingUtils().start("Rad :: Construct Radius Function")

        radius_function = {}
        for dim in range(alpha_complex.simplicial_complex.dimension, 0, -1):
            simplices: set[tuple[int, ...]] = alpha_complex.simplicial_complex.dim_simplex_dict[dim]
            for simplex in simplices:
                if radius_function.get(simplex, None) is not None:
                    continue  # if radius already found at an earlier step, skip the simplex
                circumstack = RadiusFunctionConstructor.find_smallest_circumstack_of_simplex(
                    alpha_complex, simplex)
                extra_vertices = alpha_complex.simplicial_complex.get_extra_vertices_of_cofaces(simplex)
                if RadiusFunctionConstructor.is_stack_empty_of_vertices(alpha_complex, extra_vertices, circumstack):
                    radius_function[simplex] = circumstack.maximum_radius
                    if use_morse_optimization:
                        RadiusFunctionConstructor.morse_optimization_fill_in_interval(radius_function, alpha_complex,
                                                                                      simplex, circumstack)
                else:
                    co_faces = alpha_complex.simplicial_complex.co_boundary[simplex]
                    radius_function[simplex] = min(radius_function[co_face] for co_face in co_faces)
        for simplex in alpha_complex.simplicial_complex.get_simplices_of_dim(0):
            radius_function[simplex] = 0.

        TimingUtils().stop("Rad :: Construct Radius Function")
        return radius_function

    @staticmethod
    def is_stack_empty_of_vertices(alpha_complex: CoreChromaticAlphaComplex, vertices, stack: StackOfSpheres) -> bool:
        """Return True if the squared distance from center is greater or close to the corresponding color squared radius
        for all given vertices."""
        TimingUtils().start("Rad :: Check Stack Emptiness")

        for v in vertices:
            distance = np.square(alpha_complex.points[v] - stack.center).sum()
            radius = stack.radii.get(alpha_complex.internal_labeling[v], 0)
            if distance < radius and not FloatingPointUtils.is_close(distance, radius):
                return False

        TimingUtils().stop("Rad :: Check Stack Emptiness")
        return True

    @staticmethod
    def find_smallest_circumstack_of_simplex(alpha_complex: CoreChromaticAlphaComplex,
                                             simplex: tuple) -> StackOfSpheres:
        split = ChromaticComplexUtils.split_simplex_by_labels(simplex, alpha_complex.internal_labeling)
        labels, vertex_sets = zip(*split.items())
        point_sets = [alpha_complex.points[vertex_set] for vertex_set in vertex_sets]
        center, radii = RadiusFunctionConstructor.find_smallest_circumstack(*point_sets)
        circumstack = StackOfSpheres(center, {lab: rad for lab, rad in zip(labels, radii)})
        return circumstack

    @staticmethod
    def find_smallest_circumstack(*point_sets: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """For arguments B_0, ..., B_k, return the center z and radii r_0, ..., r_k defining a stack of spheres
        S_0, ..., S_k passing through B_0, ..., B_k, respectively."""
        TimingUtils().start("Rad :: Find Smallest Circumstack")
        if any(len(point_set) == 0 for point_set in point_sets):
            raise ValueError("Point sets need to be non-empty.")
        center, radii2 = RadiusFunctionConstructor.find_smallest_circumstack_weighted_circumspheres(*point_sets)
        TimingUtils().stop("Rad :: Find Smallest Circumstack")
        return center, radii2

    # @staticmethod
    # def find_smallest_circumstack_center_miniball(*point_sets: npt.NDArray) -> npt.NDArray:
    #     TimingUtils().start("Rad :: Find Smallest Circumstack :: Miniball")
    #     lengths = [len(point_set) for point_set in point_sets]
    #     equispace = GeometryUtils.construct_equispace(*point_sets)
    #     points = np.concatenate(point_sets)
    #     points_reflected = GeometryUtils.reflect_points_through_affine_space(equispace, *points)
    #     points_reflected_filtered = np.concatenate([
    #         refl[~FloatingPointUtils.flag_duplicates_from_reference(orig, refl)]
    #         for orig, refl in zip(point_sets, RadiusFunctionConstructor.split_points(points_reflected, lengths))
    #     ])  # filter away duplicates
    #     center, rad = MiniballAlgorithm.find_smallest_enclosing_ball(
    #         np.concatenate([points, points_reflected_filtered]))
    #     TimingUtils().stop("Rad :: Find Smallest Circumstack :: Miniball")
    #
    #     return center

    @staticmethod
    def find_smallest_circumstack_weighted_circumspheres(*point_sets: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        TimingUtils().start("Rad :: Find Smallest Circumstack :: Weighted Circumspheres")
        equispace = GeometryUtils.construct_equispace(*point_sets)

        points = np.array([point_set[0] for point_set in point_sets])  # each point set projects to a single point

        if equispace.dimension < 0:
            center = equispace.shift
            radii2 = np.square(points - center).sum(axis=1)
            return center, radii2

        points_projected = GeometryUtils.orthogonal_projection_onto_affine_space(equispace, *points)
        weights = np.square(points - points_projected).sum(axis=1)
        candidate = (np.array([]), np.array([]), float('inf'))  # center, radii, max_radius
        for k in range(1, equispace.dimension + 2):  # if dim = 2, we want up to 3 points, and we need range(1, 4)
            for choice in itertools.combinations(range(len(point_sets)), k):
                center, _ = GeometryUtils.circumsphere_of_weighted_points(points_projected[list(choice)],
                                                                          weights[list(choice)])
                radii2 = np.square(points - center).sum(axis=1)
                rad2 = max(radii2)
                if rad2 < candidate[2]:
                    candidate = (center, radii2, rad2)

        TimingUtils().stop("Rad :: Find Smallest Circumstack :: Weighted Circumspheres")

        return candidate[0], candidate[1]

    @staticmethod
    def split_points(points: npt.NDArray, lengths) -> list[npt.NDArray, ...]:
        TimingUtils().start("Rad :: Split Points Into Colors")

        splits = np.zeros(len(lengths), dtype=int)
        for i, ln in enumerate(lengths):
            splits[i] = splits[i - 1] + ln

        TimingUtils().stop("Rad :: Split Points Into Colors")
        return np.split(points, splits)[:-1]

    @staticmethod
    def compute_kkt_solution(points: Iterable[npt.NDArray], labels: Iterable, circumstack: StackOfSpheres) \
            -> Optional[npt.NDArray]:
        """Return vector of lambdas ordered as simplex if valid solution exists, otherwise return None.
        WARNING: `circumstack` is assumed to be a circumstack of `points`, `labels`, condition not checked
        WARNING: Emptiness is *not* checked.
        WARNING: `points` is assumed to be all points that `circumstack` goes through --- if not the case,
                 solution might not exist, and None is returned"""
        TimingUtils().start("Rad :: Morse :: Compute KKT Solution")
        labels_to_int = {lab: i for i, lab in enumerate(circumstack.radii.keys())}
        number_of_labels = len(labels_to_int)
        point_part = np.array([
            np.concatenate([np.eye(1, number_of_labels + 1, k=labels_to_int[lab]).flatten(), pt])
            for pt, lab in zip(points, labels)
        ])
        color_part = np.zeros(shape=(len(circumstack.maximum_labels), point_part.shape[1]))
        for i, lab in enumerate(circumstack.maximum_labels):
            color_part[i, labels_to_int[lab]] = -1
            color_part[i, number_of_labels] = 1
        a_mat = np.concatenate([point_part, color_part]).transpose()
        b_vec = np.concatenate([np.eye(1, number_of_labels + 1, k=number_of_labels).flatten(), circumstack.center])

        x, res, rk, s = np.linalg.lstsq(a_mat, b_vec, rcond=None)

        TimingUtils().stop("Rad :: Morse :: Compute KKT Solution")
        if len(res) > 0 and not FloatingPointUtils.is_close(res[0], 0):
            return None
        else:
            return x[:len(point_part)]

    @staticmethod
    def generate_radius_function_interval(simplex: tuple, coefficients: Iterable) -> list[tuple, ...]:
        """Return a list of simplices [min, simplex] - simplex, where min are all the vertices for which
        the corresponding coefficient is strictly positive."""
        TimingUtils().start("Rad :: Morse :: Generate Interval Simplices")
        minimal_simplex = {v for v, coef in zip(simplex, coefficients) if coef > 0}
        non_positive_vertices = set(simplex) - minimal_simplex
        interval = [tuple(sorted(minimal_simplex.union(extra)))
                    for k in range(len(non_positive_vertices))
                    for extra in itertools.combinations(non_positive_vertices, k)]
        TimingUtils().stop("Rad :: Morse :: Generate Interval Simplices")
        return interval

    @staticmethod
    def morse_optimization_fill_in_interval(radius_function: dict[tuple, float],
                                            alpha_complex: CoreChromaticAlphaComplex,
                                            simplex: tuple,
                                            circumstack: StackOfSpheres) -> None:
        points, labels = zip(*[(alpha_complex.points[v], alpha_complex.internal_labeling[v])
                               for v in simplex])
        lambdas = RadiusFunctionConstructor.compute_kkt_solution(points, labels, circumstack)
        if lambdas is not None:
            interval = RadiusFunctionConstructor.generate_radius_function_interval(simplex, lambdas)
            for interval_simplex in interval:
                radius_function[interval_simplex] = circumstack.maximum_radius
