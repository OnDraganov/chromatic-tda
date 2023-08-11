from chromatic_tda.utils.singleton import singleton

from itertools import combinations
from chromatic_tda.core.core_simplicial_complex import CoreSimplicialComplex
from chromatic_tda.utils.simplex_utils import SimplexUtils
from chromatic_tda.utils.boundary_matrix_utils import BoundaryMatrixUtils

@singleton
class CoreSimplicialComplexFactory():

    def create_instance(self, simplices) -> CoreSimplicialComplex:
        simplicial_complex = CoreSimplicialComplex()
        self.build_complex(simplicial_complex, simplices)
        return simplicial_complex

    def build_complex(self, simplicial_complex: CoreSimplicialComplex, simplices):
        if str(type(simplices)) in (
                "<class 'list'>", "<class 'tuple'>", "<class 'set'>",
                "<class 'numpy.ndarray'>", "<class 'dict_keys'>"):
            self._build_complex_from_list(simplicial_complex, simplices)

        elif str(type(simplices)) == "<class 'dict'>":
            self._build_complex_dict(simplicial_complex, simplices)
        else:
            raise TypeError(f"Cannot build complex from type {type(simplices)}.")

    def _build_complex_from_list(self, complex: CoreSimplicialComplex, simplex_list):
        complex.dimension = max(SimplexUtils().dimension(simplex) for simplex in simplex_list)

        for dim in range(0, complex.dimension+1):
            complex.dim_simplex_dict[dim] = set()

        # create dict for dim -> simplex mapping
        for simplex in simplex_list:
            simplex_dim = SimplexUtils().dimension(simplex)
            complex.dim_simplex_dict[simplex_dim].add(tuple(sorted(simplex)))

        # creating boundary map (simplex -> it's lower dimension simplices)
        for dim in range(complex.dimension, 0, -1):
            for simplex in complex.dim_simplex_dict[dim]:
                complex.boundary[simplex] = set(combinations(simplex, dim))
            # update dim to simplex dict with boundary maps
            complex.dim_simplex_dict[dim-1] = complex.dim_simplex_dict[dim-1].union(*[complex.boundary[simplex] for simplex in complex.dim_simplex_dict[dim]])

        for vertex in complex.dim_simplex_dict[0]:
            complex.boundary[vertex] = set()        

    def _build_complex_dict(self, complex: CoreSimplicialComplex, simplex_weights_dict):
        self._build_complex_from_list(complex, simplex_weights_dict.keys())
        complex.set_simplex_weighs(simplex_weights_dict)

    def create_restricted_instance(self, complex: CoreSimplicialComplex, restricted_simplices) -> CoreSimplicialComplex:
        """Return a new SimplicialComplex restricted to given simplices."""
        new_complex : CoreSimplicialComplex = CoreSimplicialComplex()
        restricted_simplices_set : set = set(restricted_simplices)

        new_complex.dim_simplex_dict = { d : complex.dim_simplex_dict[d] & restricted_simplices_set for d in complex.dim_simplex_dict}
        new_complex.clear_empty_dimensions()

        new_complex.boundary = BoundaryMatrixUtils().get_restricted_boundary(complex, restricted_simplices_set)
        new_complex.co_boundary = BoundaryMatrixUtils().make_co_boundary(new_complex.boundary)
        new_complex.simplex_weights = { simplex : complex.simplex_weights[simplex] for simplex in new_complex.boundary}
        new_complex.sub_complex = complex.sub_complex & restricted_simplices_set

        return new_complex
