# from abc import ABC, abstractmethod
import numpy as np
import random
from scipy.spatial import Delaunay

from chromatic_tda.algorithms.radius_function import RadiusFunctionConstructor
from chromatic_tda.utils.boundary_matrix_utils import BoundaryMatrixUtils
from chromatic_tda.core.core_chromatic_alpha_complex import CoreChromaticAlphaComplex
from chromatic_tda.algorithms.OLD_radius_function_utils import RadiusFunctionUtils
from chromatic_tda.core.simplicial_complex_factory import CoreSimplicialComplexFactory
from chromatic_tda.utils.timing import TimingUtils


class CoreChromaticAlphaComplexFactory:

    def __init__(self, points, labels):
        self.points = points
        self.labels = labels
        self.check_input()
        self.alpha_complex = None

    def create_instance(self, lift_perturbation: float, point_perturbation: float,
                        radius_method: str = 'new') -> CoreChromaticAlphaComplex:
        """
        Compute the chromatic alpha complex of given points and labels.
        At most three different labels allowed
        """
        TimingUtils().start("AlphFac :: Create Alf Instance")
        self.alpha_complex = CoreChromaticAlphaComplex()

        self.init_points(point_perturbation)
        self.init_labels()
        self.build_alpha_complex_structure(lift_perturbation=lift_perturbation)
        self.add_radius_function(radius_method)
        TimingUtils().start("AlphFac :: Create Alf Instance")

        return self.alpha_complex

    def init_points(self, point_perturbation: float | None) -> None:
        if point_perturbation:
            self.alpha_complex.points = np.array(self.perturb_points(self.points, point_perturbation))
        else:
            self.alpha_complex.points = np.array(self.points)
        self.alpha_complex.points_dimension = (self.alpha_complex.points.shape[1]
                                               if len(self.alpha_complex.points.shape) > 1
                                               else 0)

    def init_labels(self) -> None:
        sorted_labels = sorted(set(self.labels))
        self.alpha_complex.input_labels_to_internal_labels_dict = {lab: i for i, lab in enumerate(sorted_labels)}
        self.alpha_complex.internal_labels_to_input_labels_dict = {
            i: lab for lab, i in self.alpha_complex.input_labels_to_internal_labels_dict.items()
        }
        self.alpha_complex.labels_number = len(self.alpha_complex.input_labels_to_internal_labels_dict)
        self.alpha_complex.internal_labeling = [self.alpha_complex.input_labels_to_internal_labels_dict[lab]
                                                for lab in self.labels]

    def build_alpha_complex_structure(self, lift_perturbation: float) -> None:
        """
        Compute the simplicial complex and radius weight function for a CoreChromaticAlphaComplex
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

        self.alpha_complex.simplicial_complex.co_boundary = BoundaryMatrixUtils.make_co_boundary(
            self.alpha_complex.simplicial_complex.boundary)

        TimingUtils().stop("AlphFac :: Build Alpha Complex Structure")

    def compute_chromatic_delaunay(self, lift_perturbation: float):
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

        del_complex = Delaunay(self.chromatic_lift(lift_perturbation))
        all_labels = set(self.alpha_complex.internal_labeling)
        colorful_maximal_simplices = [simplex for simplex in del_complex.simplices
                                      if set(self.alpha_complex.internal_labeling[i] for i in simplex) == all_labels]

        TimingUtils().stop("AlphFac :: Compute Chromatic Delaunay")

        return colorful_maximal_simplices

    @staticmethod
    def perturb_points(points, point_perturbation):
        return [[p + point_perturbation * (random.random() - .5) for p in pt] for pt in points]

    def chromatic_lift(self, lift_perturbation):
        """
        Add extra coordinates to lift points to the chromatic simplex. Here we choose one-hot embedding without
        the first coordinate. That is, 0 --> (0,0,0,...), 1 --> (1,0,0,...), 2 --> (0,1,0,...), etc.
        """
        pts_lift = np.array([
            np.concatenate((point, [1 if i == label else 0 for i in range(1, self.alpha_complex.labels_number)]))
            for point, label in zip(self.alpha_complex.points, self.alpha_complex.internal_labeling)])
        if lift_perturbation:
            prefix = [0] * self.alpha_complex.points_dimension
            for pt in pts_lift:
                pt += prefix + [lift_perturbation * random.random() for _ in range(1, self.alpha_complex.labels_number)]

        return pts_lift

    def add_radius_function(self, method):
        if method == 'old':
            RadiusFunctionUtils().compute_radius_function(self.alpha_complex)
        elif method == 'new':
            self.alpha_complex.simplicial_complex.set_simplex_weights(
                RadiusFunctionConstructor.construct_radius_function(self.alpha_complex))

    def check_input(self):
        if len(self.points) != len(self.labels):
            raise ValueError("The length of points has to equal the length of labels.")
