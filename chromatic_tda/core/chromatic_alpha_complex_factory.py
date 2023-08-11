# from abc import ABC, abstractmethod
import numpy as np
import random
from scipy.spatial import Delaunay

from chromatic_tda.utils.singleton import singleton
from chromatic_tda.core.core_chromatic_alpha_complex import CoreChromaticAlphaComplex
from chromatic_tda.utils.boundary_matrix_utils import BoundaryMatrixUtils
from chromatic_tda.algorithms.radius_function_utils import RadiusFunctionUtils
from chromatic_tda.core.simplicial_complex_factory import CoreSimplicialComplexFactory
from chromatic_tda.utils.timing import TimingUtils



@singleton
class CoreChromaticAlphaComplexFactory():

    def create_instance(self, points, labels, lift_perturbation, point_perturbation, **kwargs) -> CoreChromaticAlphaComplex:
        alpha_complex = CoreChromaticAlphaComplex()
        self.build_alpha_complex(alpha_complex, points, labels, lift_perturbation, point_perturbation, **kwargs)
        return alpha_complex

    def build_alpha_complex(self, alpha_complex: CoreChromaticAlphaComplex, points, labels, lift_perturbation,
                            point_perturbation, **kwargs):
        """
        Compute the chromatic alpha complex of given points.
        At most three different labels allowed
        """
        TimingUtils().start("Build Alpha Complex")

        if point_perturbation:
            alpha_complex.points = np.array([[p + point_perturbation * (random.random() - .5) for p in pt] for pt in points])
        else:
            alpha_complex.points = np.array(points)
            
        if alpha_complex.points.shape[1] != 2:
            raise ValueError("Points has to be an iterable of two-dimensional points.")

        alpha_complex.input_labels_to_internal_labels_dict = {lab : i for i, lab in enumerate(sorted(set(labels)))}
        alpha_complex.internal_labels_to_input_labels_dict = {i : lab for lab, i in alpha_complex.input_labels_to_internal_labels_dict.items()}
        alpha_complex.labels_number = len(alpha_complex.input_labels_to_internal_labels_dict)
        alpha_complex.internal_labels = [alpha_complex.input_labels_to_internal_labels_dict[lab] for lab in labels]

        if alpha_complex.labels_number > 3:
            raise ValueError(f"There can be at most 3 different labels, {alpha_complex.labels_number} given.")
        if len(alpha_complex.points) != len(alpha_complex.internal_labels):
            raise ValueError("The list of labels must have the same length as the list of points.")

        self.construct_chromatic_delaunay(alpha_complex, lift_perturbation)
        alpha_complex.simplicial_complex.co_boundary = BoundaryMatrixUtils().make_co_boundary(alpha_complex.simplicial_complex.boundary)
        RadiusFunctionUtils().compute_radius_function(alpha_complex, **kwargs)
#        RadiusFunctionParallelUtils(alpha_complex).compute_radius_function_in_parallel(**kwargs)

        TimingUtils().stop("Build Alpha Complex")

    def construct_chromatic_delaunay(self, alpha_complex: CoreChromaticAlphaComplex, lift_perturbation = None):
        TimingUtils().start("Construct Chromatic Delaunay")

        del_complex = Delaunay( self.chromatic_lift(alpha_complex, lift_perturbation))
        all_labels = set(alpha_complex.internal_labels)
        simplices = [simplex for simplex in del_complex.simplices if set(alpha_complex.internal_labels[i] for i in simplex) == all_labels]
        alpha_complex.simplicial_complex = CoreSimplicialComplexFactory().create_instance(simplices)

        TimingUtils().stop("Construct Chromatic Delaunay")

    def chromatic_lift(self, alpha_complex: CoreChromaticAlphaComplex, lift_perturbation = None):
        """
        Add extra coordinates to lift points to the chromatic simplex. Here we choose one-hot embedding without
        the first coordinate. That is, 0 --> (0,0,0,...), 1 --> (1,0,0,...), 2 --> (0,1,0,...), etc.
        """
        pts_lift = np.array([
            np.concatenate((point, [1 if i == label else 0 for i in range(1, alpha_complex.labels_number)]))
            for point, label in zip(alpha_complex.points, alpha_complex.internal_labels)])
        if lift_perturbation:
            for i in range(len(pts_lift)):
                pts_lift[i] += [0, 0] + [lift_perturbation * random.random() for i in range(1, alpha_complex.labels_number)]

        return pts_lift
