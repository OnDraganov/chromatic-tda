# from abc import ABC, abstractmethod
import numpy as np
import random
from scipy.spatial import Delaunay

from chromatic_tda.utils.singleton import singleton
from chromatic_tda.core.core_chromatic_alpha_complex import CoreChromaticAlphaComplex
from chromatic_tda.utils import boundary_matrix_utils
from chromatic_tda.algorithms.radius_function_utils import RadiusFunctionUtils
from chromatic_tda.core.simplicial_complex_factory import CoreSimplicialComplexFactory
from chromatic_tda.utils.timing import TimingUtils


@singleton
class CoreChromaticAlphaComplexFactory:

    def create_instance(self, points, labels, lift_perturbation, point_perturbation) -> CoreChromaticAlphaComplex:
        """
        Compute the chromatic alpha complex of given points and labels.
        At most three different labels allowed
        """
        alpha_complex = CoreChromaticAlphaComplex()

        self.init_points(alpha_complex, points, point_perturbation)
        self.init_labels(alpha_complex, labels)
        self.build_alpha_complex_structure(alpha_complex, lift_perturbation=lift_perturbation)

        return alpha_complex

    def init_points(self, alpha_complex: CoreChromaticAlphaComplex, points, point_perturbation) -> None:
        if point_perturbation:
            alpha_complex.points = np.array(self.perturb_points(points, point_perturbation))
        else:
            alpha_complex.points = np.array(points)
        alpha_complex.points_dimension = alpha_complex.points.shape[1] if len(alpha_complex.points.shape) > 1 else 0

    @staticmethod
    def init_labels(alpha_complex: CoreChromaticAlphaComplex, labels) -> None:
        alpha_complex.input_labels_to_internal_labels_dict = {lab: i for i, lab in enumerate(sorted(set(labels)))}
        alpha_complex.internal_labels_to_input_labels_dict = {
            i: lab for lab, i in alpha_complex.input_labels_to_internal_labels_dict.items()
        }
        alpha_complex.labels_number = len(alpha_complex.input_labels_to_internal_labels_dict)
        alpha_complex.internal_labels = [alpha_complex.input_labels_to_internal_labels_dict[lab] for lab in labels]

    @staticmethod
    def perturb_points(points, point_perturbation):
        return [[p + point_perturbation * (random.random() - .5) for p in pt] for pt in points]

    def build_alpha_complex_structure(self, alpha_complex: CoreChromaticAlphaComplex, lift_perturbation) -> None:
        """
        Compute the simplicial complex and radius weight function for a CoreChromaticAlphaComplex
        with initialised points and labels. Adds the structure directly to the given alpha_complex.

        Parameters
        ----------
        alpha_complex       CoreChromaticAlphaComplex with initialised points and labels
        lift_perturbation   amount by which to perturb the lifting (in order to make Delaunay computation easier)
        """
        TimingUtils().start("Build Alpha Complex Structure")

        if alpha_complex.points.shape[1] != 2:
            raise ValueError("Points has to be an iterable of two-dimensional points.")

        if alpha_complex.labels_number > 3:
            raise ValueError(f"There can be at most 3 different labels, {alpha_complex.labels_number} given.")
        if len(alpha_complex.points) != len(alpha_complex.internal_labels):
            raise ValueError("The list of labels must have the same length as the list of points.")

        colorful_maximal_simplices = self.compute_chromatic_delaunay(alpha_complex, lift_perturbation)
        TimingUtils().start("Build Chro Del from Max Simplices")
        alpha_complex.simplicial_complex = CoreSimplicialComplexFactory().create_instance(colorful_maximal_simplices)
        TimingUtils().stop("Build Chro Del from Max Simplices")

        alpha_complex.simplicial_complex.co_boundary = boundary_matrix_utils.make_co_boundary(
            alpha_complex.simplicial_complex.boundary)

        RadiusFunctionUtils().compute_radius_function(alpha_complex)

        TimingUtils().stop("Build Alpha Complex Structure")

    def compute_chromatic_delaunay(self, alpha_complex: CoreChromaticAlphaComplex, lift_perturbation):
        """
        Parameters
        ----------
        alpha_complex       CoreChromaticAlphaComplex with initialised points and labels
        lift_perturbation   amount by which to perturb the lifting (in order to make Delaunay computation easier)

        Returns
        -------
        List of maximal simplices of the chromatic Delaunay complex.
        """
        TimingUtils().start("Compute Chromatic Delaunay")

        del_complex = Delaunay(self.chromatic_lift(alpha_complex, lift_perturbation))
        all_labels = set(alpha_complex.internal_labels)
        colorful_maximal_simplices = [simplex for simplex in del_complex.simplices
                                      if set(alpha_complex.internal_labels[i] for i in simplex) == all_labels]

        TimingUtils().stop("Compute Chromatic Delaunay")

        return colorful_maximal_simplices

    @staticmethod
    def chromatic_lift(alpha_complex: CoreChromaticAlphaComplex, lift_perturbation):
        """
        Add extra coordinates to lift points to the chromatic simplex. Here we choose one-hot embedding without
        the first coordinate. That is, 0 --> (0,0,0,...), 1 --> (1,0,0,...), 2 --> (0,1,0,...), etc.
        """
        pts_lift = np.array([
            np.concatenate((point, [1 if i == label else 0 for i in range(1, alpha_complex.labels_number)]))
            for point, label in zip(alpha_complex.points, alpha_complex.internal_labels)])
        if lift_perturbation:
            prefix = [0] * alpha_complex.points_dimension
            for pt in pts_lift:
                pt += prefix + [lift_perturbation * random.random() for _ in range(1, alpha_complex.labels_number)]

        return pts_lift
