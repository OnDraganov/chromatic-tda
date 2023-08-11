import numpy as np
import itertools

from chromatic_tda.core.core_simplicial_complex import CoreSimplicialComplex
from chromatic_tda.utils.geometrical_utils import GeometricalUtils
from chromatic_tda.core.simplicial_complex_factory import CoreSimplicialComplexFactory


class CoreChromaticAlphaComplex():

    points : np.ndarray
    input_labels_to_internal_labels_dict : dict
    internal_labels_to_input_labels_dict : dict
    labels_number : int
    internal_labels : list
    simplicial_complex : CoreSimplicialComplex
    sq_rad : dict

    def __init__(self) -> None:
        self.input_labels_to_internal_labels_dict = {}
        self.labels_number = 0
        self.internal_labels = []
        self.sq_rad = {}

    def __len__(self) -> int:
        return len(self.simplicial_complex)

    def __contains__(self, item) -> bool:
        return item in self.simplicial_complex

    def _process_sub_complex_parameter_string(self, parameter : str):
        parameter = parameter.lower().strip()
        if parameter == 'all':
            list_of_sets_form = [set(self.input_labels_to_internal_labels_dict.keys())]
        elif parameter.endswith('chromatic'):
            chromaticity = parameter.replace('chromatic', '').replace('-', '')
            words_to_numbers = {'mono': 1, 'one': 1, 'bi': 2, 'two': 2, 'tri': 3, 'three': 3, 'tetra': 4, 'four': 4}
            if chromaticity in words_to_numbers:
                chromaticity = words_to_numbers[chromaticity]
            elif chromaticity.isnumeric() and chromaticity != '0':
                chromaticity = int(chromaticity)
            else:
                raise ValueError(f"The simplex restriction color parameter given `{parameter}` is invalid")
            list_of_sets_form = [set(x) for x in itertools.combinations(self.input_labels_to_internal_labels_dict.keys(), chromaticity)]
        else:
            list_of_sets_form = [{int(ch) for ch in w} for w in parameter.split(',')]

        return list_of_sets_form

    def _process_sub_complex_parameter(self, parameter, allow_unused_labels):
        if isinstance(parameter, str):
            list_of_sets_form = self._process_sub_complex_parameter_string(parameter)
        else:
            list_of_sets_form = [set(color_set) for color_set in parameter]

        if allow_unused_labels:
            list_of_sets_form = [color_set & set(self.input_labels_to_internal_labels_dict.keys()) for color_set in list_of_sets_form]
        else:
            for lab in set().union(*list_of_sets_form):
                if lab not in self.input_labels_to_internal_labels_dict.keys():
                    raise ValueError(f"There is no point labeled by `{lab}`. "
                                     f"To suppress this error, pass allow_unused_labels=True to get_complex function")
        list_of_sets_form_internal = [{self.input_labels_to_internal_labels_dict[lab] for lab in color_set}
                                      for color_set in list_of_sets_form]
        return list_of_sets_form_internal

    def simplex_labels_internal(self, simplex):
        return {self.internal_labels[v] for v in simplex}

    def simplex_labels_input(self, simplex):
        return {self.internal_labels_to_input_labels_dict[lab] for lab in self.simplex_labels_internal(simplex)}

    def get_complex(self, sub_complex, full_complex, relative, allow_unused_labels) -> CoreSimplicialComplex:
        if full_complex is None or full_complex == '':
            complex_simplices = set(self.simplicial_complex.boundary)
        else:
            complex_allowed_labels = self._process_sub_complex_parameter(full_complex, allow_unused_labels)
            complex_simplices = set()
            for simplex in self.simplicial_complex.boundary:
                simplex_labels = self.simplex_labels_internal(simplex)
                if any( simplex_labels.issubset(s) for s in complex_allowed_labels ):
                    complex_simplices.add(simplex)

        if relative is None or relative == '':
            relative_simplices = set()
        else:
            relative_labels_to_drop = self._process_sub_complex_parameter(relative, allow_unused_labels)
            relative_simplices = set()
            for simplex in self.simplicial_complex.boundary:
                simplex_labels = self.simplex_labels_internal(simplex)
                if any( simplex_labels.issubset(s) for s in relative_labels_to_drop ):
                    relative_simplices.add(simplex)

        restricted_complex : CoreSimplicialComplex = CoreSimplicialComplexFactory().create_restricted_instance(self.simplicial_complex, complex_simplices - relative_simplices)

        if sub_complex is None or sub_complex == '':
            restricted_complex.set_sub_complex(set())
        else:
            sub_complex_allowed_labels = self._process_sub_complex_parameter(sub_complex, allow_unused_labels)
            sub_complex_simplices = set()
            for simplex in restricted_complex.boundary: # already relativized in definition of new-complex
                simplex_labels = self.simplex_labels_internal(simplex)
                if any( simplex_labels.issubset(s) for s in sub_complex_allowed_labels ):
                    sub_complex_simplices.add(simplex)
            restricted_complex.set_sub_complex(sub_complex_simplices)

        return restricted_complex

    def star_vertices(self, simplex) -> set:
        return set().union(*self.simplicial_complex.co_boundary[simplex]) - set(simplex)

    def is_empty_stack(self, center, simplex) -> bool:
        """
        Return True if the stack centered at `center` is empty.
        Assume that the center given is a circum-center for each
        monochromatic face, and check whether the vertices in
        co-boundary of simplex are inside the circum-centers of
        the corresponding colors. Colors not present in simplex
        are treated as automatically satisfied.
        """
        vertices_to_check = self.star_vertices(simplex)
        for mono_chrom_face in self.split_simplex(simplex): # for every color
            if len(mono_chrom_face) > 0: # if the color is present in simplex
                radius = GeometricalUtils().sq_dist(center, self.points[mono_chrom_face[0]])
                if any( (self.internal_labels[mono_chrom_face[0]] == self.internal_labels[v] and
                         GeometricalUtils().sq_dist(center, self.points[v]) < radius)
                        for v in vertices_to_check ):
                    return False
        return True

    def split_simplex(self, simplex) -> list:
        return [tuple(i for i in simplex if self.internal_labels[i] == 0),
                tuple(i for i in simplex if self.internal_labels[i] == 1),
                tuple(i for i in simplex if self.internal_labels[i] == 2)]

    def split_simplex_sort_by_size(self, simplex) -> list:
        return sorted(self.split_simplex(simplex), key=len, reverse=True)

    def write(self) -> None:
        print()
        print(f"**** Delaunay Complex (dimension = {self.simplicial_complex.get_dimension()}) ****")
        for (p,l) in zip(self.points, self.internal_labels):
            print(f"{p} -> {l}")
        print(42*"*")
        print()
