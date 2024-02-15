import numpy as np
from collections.abc import KeysView, ValuesView

from chromatic_tda.utils.singleton import singleton

from itertools import combinations
from chromatic_tda.core.core_simplicial_complex import CoreSimplicialComplex
from chromatic_tda.utils import boundary_matrix_utils, simplex_utils


@singleton
class CoreSimplicialComplexFactory:

    def create_instance(self, simplices) -> CoreSimplicialComplex:
        simplicial_complex = CoreSimplicialComplex()
        self.build_complex(simplicial_complex, simplices)

        return simplicial_complex

    def build_complex(self, simplicial_complex: CoreSimplicialComplex, simplices):
        if isinstance(simplices, dict):
            self.build_complex_from_dictionary(simplicial_complex, simplices)
        elif isinstance(simplices, (list, tuple, set, np.ndarray, KeysView, ValuesView)):
            self.build_complex_from_list(simplicial_complex, simplices)
        else:
            raise TypeError(f"Cannot build complex from type {type(simplices)}.")

    def build_complex_from_dictionary(self, simplicial_complex: CoreSimplicialComplex, simplex_weights_dict):
        self.build_complex_from_list(simplicial_complex, simplex_weights_dict.keys())
        simplicial_complex.set_simplex_weights(simplex_weights_dict)

    def build_complex_from_list(self, simplicial_complex: CoreSimplicialComplex, simplex_list):
        simplicial_complex.dimension = self.find_maximal_dimension(simplex_list)
        simplicial_complex.dim_simplex_dict = self.build_dimension_dictionary(
            simplex_list, max_dimension=simplicial_complex.dimension)
        self.add_boundary_and_missing_simplices(simplicial_complex)
        simplicial_complex.set_simplex_weights({})

    @staticmethod
    def find_maximal_dimension(simplex_list):
        return max(simplex_utils.dimension(simplex) for simplex in simplex_list)

    @staticmethod
    def build_dimension_dictionary(simplex_list, max_dimension):
        dim_simplex_dict = {dim : set() for dim in range(max_dimension, -1, -1)}
        for simplex in simplex_list:
            simplex_dim = simplex_utils.dimension(simplex)
            dim_simplex_dict[simplex_dim].add(tuple(sorted(simplex)))
        return dim_simplex_dict

    @staticmethod
    def add_boundary_and_missing_simplices(simplicial_complex: CoreSimplicialComplex):
        for dim in range(simplicial_complex.dimension, 0, -1):
            for simplex in simplicial_complex.dim_simplex_dict[dim]:
                simplicial_complex.boundary[simplex] = set(combinations(simplex, dim))
            simplicial_complex.dim_simplex_dict[dim - 1] = simplicial_complex.dim_simplex_dict[dim - 1].union(
                *[simplicial_complex.boundary[simplex] for simplex in simplicial_complex.dim_simplex_dict[dim]])
        for vertex in simplicial_complex.dim_simplex_dict[0]:
            simplicial_complex.boundary[vertex] = set()

    @staticmethod
    def create_restricted_instance(complex: CoreSimplicialComplex, restricted_simplices) -> CoreSimplicialComplex:
        """Return a new SimplicialComplex restricted to given simplices."""
        new_complex : CoreSimplicialComplex = CoreSimplicialComplex()
        restricted_simplices_set : set = set(restricted_simplices)

        new_complex.dim_simplex_dict = {d : complex.dim_simplex_dict[d] & restricted_simplices_set
                                        for d in complex.dim_simplex_dict}
        new_complex.clear_empty_dimensions()
        new_complex.dimension = max(new_complex.dim_simplex_dict)

        new_complex.boundary = {simplex: complex.boundary[simplex] & restricted_simplices_set
                                for simplex in set(complex.boundary) & restricted_simplices_set}
        new_complex.co_boundary = boundary_matrix_utils.make_co_boundary(new_complex.boundary)
        new_complex.simplex_weights = {simplex : complex.simplex_weights[simplex] for simplex in new_complex.boundary}
        new_complex.sub_complex = complex.sub_complex & restricted_simplices_set

        return new_complex
