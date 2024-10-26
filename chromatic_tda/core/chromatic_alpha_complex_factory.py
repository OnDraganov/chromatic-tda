from typing import Optional

import numpy as np
import numpy.typing as npt
import random
from scipy.spatial import Delaunay

from chromatic_tda.algorithms.radius_function import RadiusFunctionConstructor
from chromatic_tda.utils.boundary_matrix_utils import BoundaryMatrixUtils
from chromatic_tda.core.core_chromatic_alpha_complex import CoreChromaticAlphaComplex
from chromatic_tda.algorithms.legacy_radius_function_utils import LegacyRadiusFunctionUtils
from chromatic_tda.core.simplicial_complex_factory import CoreSimplicialComplexFactory
from chromatic_tda.utils.timing import TimingUtils


class CoreChromaticAlphaComplexFactory:

    def __init__(self, points, labels):
        self.points = np.array(points, dtype='d')
        self.labels = labels
        self.check_input()
        self.alpha_complex = None

    def create_instance(self, lift_perturbation: Optional[float],
                        point_perturbation: Optional[float],
                        use_morse_optimization: bool = True,
                        legacy_radius_function: bool = False) -> CoreChromaticAlphaComplex:
        """
        Compute the chromatic alpha complex of given points and labels.
        """
        TimingUtils().start("AlphFac :: Create Alf Instance")
        self.alpha_complex = CoreChromaticAlphaComplex()

        self.init_points(self.points, point_perturbation)
        self.init_labels(self.labels)
        self.build_alpha_complex_structure(lift_perturbation=lift_perturbation)
        self.add_radius_function(use_morse_optimization=use_morse_optimization,
                                 legacy_radius_function=legacy_radius_function)
        TimingUtils().start("AlphFac :: Create Alf Instance")

        return self.alpha_complex

    def init_points(self, points, point_perturbation: Optional[float]) -> None:
        if point_perturbation:
            self.alpha_complex.points = np.array(self.perturb_points(points, point_perturbation))
        else:
            self.alpha_complex.points = np.array(points)
        self.alpha_complex.points_dimension = (self.alpha_complex.points.shape[1]
                                               if len(self.alpha_complex.points.shape) > 1
                                               else 0)

    def init_labels(self, labels) -> None:
        sorted_labels = sorted(set(labels))
        self.alpha_complex.input_labels_to_internal_labels_dict = {lab: i for i, lab in enumerate(sorted_labels)}
        self.alpha_complex.internal_labels_to_input_labels_dict = {
            i: lab for lab, i in self.alpha_complex.input_labels_to_internal_labels_dict.items()
        }
        self.alpha_complex.labels_number = len(self.alpha_complex.input_labels_to_internal_labels_dict)
        self.alpha_complex.internal_labeling = [self.alpha_complex.input_labels_to_internal_labels_dict[lab]
                                                for lab in labels]

    def build_alpha_complex_structure(self, lift_perturbation: float, make_co_boundary: bool = True) -> None:
        """
        Compute the simplicial complex for a CoreChromaticAlphaComplex
        with initialised points and labels. Adds the structure directly to the given alpha_complex.

        Parameters
        ----------
        alpha_complex       CoreChromaticAlphaComplex with initialised points and labels
        lift_perturbation   amount by which to perturb the lifting (in order to make Delaunay computation easier)
        """
        TimingUtils().start("AlphFac :: Build Alpha Complex Structure")

        colorful_max_simplices = self.compute_chromatic_delaunay(lift_perturbation)
        TimingUtils().start("AlphFac :: Build Chro Del From Max Simplices")
        self.alpha_complex.simplicial_complex = CoreSimplicialComplexFactory().create_instance(colorful_max_simplices)
        TimingUtils().stop("AlphFac :: Build Chro Del From Max Simplices")

        if make_co_boundary:
            self.alpha_complex.simplicial_complex.co_boundary = BoundaryMatrixUtils.make_co_boundary(
                self.alpha_complex.simplicial_complex.boundary)

        TimingUtils().stop("AlphFac :: Build Alpha Complex Structure")

    def compute_chromatic_delaunay(self, lift_perturbation: float) -> list[tuple[int, ...]]:
        """
        Parameters
        ----------
        alpha_complex       CoreChromaticAlphaComplex with initialised points and labels
        lift_perturbation   amount by which to perturb the lifting (in order to make Delaunay computation easier)

        Returns
        -------
        List of maximal simplices of the chromatic Delaunay complex.
        """
        TimingUtils().start("AlphFac :: Compute Chromatic Delaunay")

        simplices = Delaunay(self.chromatic_lift(lift_perturbation)).simplices
        colorful_maximal_simplices = self.filter_colorful_simplices(simplices)

        TimingUtils().stop("AlphFac :: Compute Chromatic Delaunay")

        return [tuple(sorted(int(v) for v in simplex)) for simplex in colorful_maximal_simplices]

    def filter_colorful_simplices(self, simplices):
        """Generator. Given labels and simplices, yields only those simplices that span all colors."""
        all_labels = set(self.alpha_complex.internal_labeling)
        for simplex in simplices:
            if set(self.alpha_complex.internal_labeling[i] for i in simplex) == all_labels:
                yield simplex

    @staticmethod
    def perturb_points(points, point_perturbation):
        return [[p + point_perturbation * (random.random() - .5) for p in pt] for pt in points]

    def chromatic_lift(self, lift_perturbation) -> npt.NDArray:
        """
        Add extra coordinates to lift points to the chromatic simplex. Here we choose one-hot embedding without
        the first coordinate. That is, 0 --> (0,0,0,...), 1 --> (1,0,0,...), 2 --> (0,1,0,...), etc.
        """
        pts_lift: npt.NDArray = np.array([
            np.concatenate((point, [1 if i == label else 0 for i in range(1, self.alpha_complex.labels_number)]))
            for point, label in zip(self.alpha_complex.points, self.alpha_complex.internal_labeling)])
        if lift_perturbation:
            prefix = [0] * self.alpha_complex.points_dimension
            for pt in pts_lift:
                pt += prefix + [lift_perturbation * random.random() for _ in range(1, self.alpha_complex.labels_number)]

        return pts_lift

    def add_radius_function(self, use_morse_optimization: bool, legacy_radius_function: bool):
        if legacy_radius_function:
            LegacyRadiusFunctionUtils().compute_radius_function(self.alpha_complex)
        else:
            sq_radius_function = RadiusFunctionConstructor.construct_sq_radius_function(
                self.alpha_complex, use_morse_optimization=use_morse_optimization)
            self.alpha_complex.simplicial_complex.set_simplex_weights(
                {simplex: np.sqrt(rad2) for simplex, rad2 in sq_radius_function.items()})

    def check_input(self):
        if len(self.points) != len(self.labels):
            raise ValueError("The length of points has to equal the length of labels.")


class CoreChromaticAlphaComplexTorus2DFactory(CoreChromaticAlphaComplexFactory):

    def __init__(self, points, labels, xrange=None, yrange=None,
                 suppress_wrapping_check=False, suppress_boundary_consistency_check=False):
        super().__init__(points, labels)
        if self.points.shape[1] != 2:
            raise ValueError(f"ChromaticAlphaComplexTorus2D expects 2-dimensional"
                             f" point sets ({self.alpha_complex.points_dimension}-dimensional given).")
        if xrange is None or yrange is None:
            raise TypeError("ChromaticAlphaComplexTorus2D missing required argument xrange or yrange")
        self.check_frame(xrange, yrange)
        self.xrange = np.array(xrange)
        self.yrange = np.array(yrange)
        self.suppress_wrapping_check = suppress_wrapping_check
        self.suppress_boundary_consistency_check = suppress_boundary_consistency_check
        self.xshift, self.yshift = self.get_shifts()
        self.n = len(points)

    def create_instance(self, lift_perturbation: Optional[float],
                        point_perturbation: Optional[float],
                        use_morse_optimization: bool = True,
                        legacy_radius_function: bool = False) -> CoreChromaticAlphaComplex:
        """
        Compute the chromatic alpha complex of given points and labels.
        """
        TimingUtils().start("AlphFac :: Create Alf Instance Torus")
        self.alpha_complex = CoreChromaticAlphaComplex()

        self.init_points_torus(self.points, point_perturbation=None)
        self.init_labels(list(self.labels) * 9)
        self.build_alpha_complex_structure_torus(lift_perturbation=lift_perturbation)
        self.add_radius_function(use_morse_optimization=use_morse_optimization,
                                 legacy_radius_function=legacy_radius_function)
        self.restrict_to_torus_simplices()

        TimingUtils().start("AlphFac :: Create Alf Instance Torus")

        return self.alpha_complex

    def check_frame(self, xrange: npt.NDArray, yrange: npt.NDArray):
        """Check whether the points fit into the frame."""
        if (xrange[0] >= xrange[1]) or (yrange[0] >= yrange[1]):
            raise ValueError("xrange and yrange need to be non-trivial intervals (a, b) with a < b.")
        if ((xrange[0] <= self.points[:, 0]) * (self.points[:, 0] <= xrange[1])
                * (yrange[0] <= self.points[:, 1]) * (self.points[:, 1] <= yrange[1])).all():
            return True
        else:
            raise ValueError("The points do not fit into the given (xrange, yrange) frame.")

    def get_shifts(self):
        """Return the shifting constants"""
        return self.xrange[1] - self.xrange[0], self.yrange[1] - self.yrange[0]

    def get_3x3grid(self, points):
        return np.concatenate((
            points,
            points + [0, self.yshift],
            points + [self.xshift, self.yshift],
            points + [self.xshift, 0],
            points + [self.xshift, - self.yshift],
            points + [0, - self.yshift],
            points + [- self.xshift, - self.yshift],
            points + [- self.xshift, 0],
            points + [- self.xshift, self.yshift],
        ))

    def init_points_torus(self, points, point_perturbation: Optional[float]) -> None:
        if point_perturbation:
            points = np.array(self.perturb_points(points, point_perturbation))
        else:
            points = np.array(points)
        self.alpha_complex.points = self.get_3x3grid(points)
        self.alpha_complex.points_dimension = 2

    def build_alpha_complex_structure_torus(self, lift_perturbation: float, make_co_boundary: bool = True) -> None:
        colorful_max_simplices = self.compute_chromatic_delaunay(lift_perturbation)
        colorful_max_simplices = self.purge_outer_simplices(colorful_max_simplices)

        if not self.suppress_boundary_consistency_check:
            if not self.check_fibers_of_maximal_simplices(colorful_max_simplices):
                self._error_boundary_consistency()

        self.alpha_complex.simplicial_complex = CoreSimplicialComplexFactory().create_instance(colorful_max_simplices)

        if make_co_boundary:
            self.alpha_complex.simplicial_complex.co_boundary = BoundaryMatrixUtils.make_co_boundary(
                self.alpha_complex.simplicial_complex.boundary)

    def purge_outer_simplices(self, simplices):
        """Return list of only those simplices that have at least one vertex in the inner square of the 3x3 grid.
        The check is done by vertex index: a vertex is in the inner region iff its index is less than the length of
        the original given point set."""
        return [simplex for simplex in simplices if any(v < self.n for v in simplex)]

    def restrict_to_torus_simplices(self):
        """Restricts the alpha_complex with computed radius function to just the torus simplices."""
        torus_simplices_transform = {simplex: self.transform_simplex_to_torus(simplex)
                                     for simplex in self.alpha_complex.simplicial_complex.simplex_weights.keys()
                                     if self.is_torus_simplex(simplex)}
        if not self.suppress_wrapping_check:  # keyword parameter of the factory
            if not self.check_unique_preimages(torus_simplices_transform):
                self._error_wrapping()
        self.alpha_complex.simplicial_complex = CoreSimplicialComplexFactory().create_instance(
            {
                simplex_transformed: self.alpha_complex.simplicial_complex.simplex_weights[simplex]
                for simplex, simplex_transformed in torus_simplices_transform.items()
            }
        )

    def is_torus_simplex(self, simplex):
        """Return True if the given simplex is the single representative of a torus simplex."""
        if len(simplex) == 1:
            return simplex[0] < self.n
        else:
            return min(simplex) == min([v % self.n for v in simplex])

    def transform_simplex_to_torus(self, simplex):
        """Given a simplex on 3x3 grid, return the simplex on the torus using only original vertices."""
        return tuple(sorted(v % self.n for v in simplex))

    @staticmethod
    def check_unique_preimages(torus_simplices_transform):
        return len(set(torus_simplices_transform.keys())) == len(set(torus_simplices_transform.values()))

    def _error_wrapping(self):
        raise ValueError("Multiple simplices wrap around the torus to became the same simplex. "
                         "This can happen if there is not enough points of any one color. "
                         "The computed six-pack would NOT be the desired six-pack on torus! "
                         "You can suppress this error by passing suppress_wrapping_check=True "
                         "to ChromaticAlphaComplex.")

    def check_fibers_of_maximal_simplices(self, max_simplices):
        """Return True if, in the set of maximal simplices mapped to the same center simplex, each center vertex
        is present exactly once."""
        fibers = {}
        for simplex in max_simplices:
            simplex_center = self.transform_simplex_to_torus(simplex)
            if simplex_center not in fibers:
                fibers[simplex_center] = set()
            fibers[simplex_center].add(simplex)

        for simplex_center, fiber in fibers.items():
            vertex_counter = [0] * len(simplex_center)
            for i, v in enumerate(simplex_center):
                for fiber_simplex in fiber:
                    if v in fiber_simplex:
                        vertex_counter[i] += 1
            if any(count != 1 for count in vertex_counter):
                return False
        return True

    def _error_boundary_consistency(self):
        raise ValueError("Inconsistency in the Delaunay complex at the boundary of the torus region. "
                         "Likely due to position close to non-generic. Try perturbing the point set, "
                         "e.g., by passing `point_perturbation=1e-5` argument to the constructor. "
                         "You can suppress this error by passing suppress_boundary_consistency_check=True "
                         "to ChromaticAlphaComplex.")
