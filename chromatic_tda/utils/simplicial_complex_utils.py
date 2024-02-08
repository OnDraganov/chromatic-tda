import itertools

from chromatic_tda.core.core_simplicial_complex import CoreSimplicialComplex
from chromatic_tda.core.simplicial_complex_factory import CoreSimplicialComplexFactory
from chromatic_tda.utils.singleton import singleton


@singleton
class SimplicialComplexUtils:

    def get_chromatic_subcomplex(self, sub_complex, full_complex, relative,
                                 simplicial_complex: CoreSimplicialComplex, internal_labeling,
                                 labels_user_to_internal=None, allow_unused_labels=False) -> CoreSimplicialComplex:
        if full_complex is None or full_complex == '' or full_complex.lower().strip() == 'all':
            complex_simplices = set(simplicial_complex.boundary)
        else:
            pattern = self.read_pattern_input(full_complex, labels_user_to_internal, check_labels=not allow_unused_labels)
            if labels_user_to_internal is not None:
                pattern = self.pattern_translate_user_to_internal(pattern, labels_user_to_internal)
            complex_simplices = self.select_simplices_with_chromatic_pattern(
                simplicial_complex.boundary, internal_labeling, pattern)

        if relative is None or relative == '':
            relative_simplices = set()
        else:
            pattern = self.read_pattern_input(relative, labels_user_to_internal, check_labels=not allow_unused_labels)
            if labels_user_to_internal is not None:
                pattern = self.pattern_translate_user_to_internal(pattern, labels_user_to_internal)
            relative_simplices = self.select_simplices_with_chromatic_pattern(
                simplicial_complex.boundary, internal_labeling, pattern)

        if sub_complex is None or sub_complex == '':
            sub_complex_simplices = set(self.simplicial_complex.boundary)
        else:
            pattern = self.read_pattern_input(sub_complex, labels_user_to_internal, check_labels=not allow_unused_labels)
            if labels_user_to_internal is not None:
                pattern = self.pattern_translate_user_to_internal(pattern, labels_user_to_internal)
            sub_complex_simplices = self.select_simplices_with_chromatic_pattern(
                simplicial_complex.boundary, internal_labeling, pattern)

        restricted_complex : CoreSimplicialComplex = CoreSimplicialComplexFactory().create_restricted_instance(
            simplicial_complex, complex_simplices - relative_simplices)
        restricted_complex.set_sub_complex(sub_complex_simplices - relative_simplices)

        return restricted_complex

    def select_simplices_with_chromatic_pattern(self, simplices, labeling, pattern):
        return set(simplex for simplex in simplices if self.simplex_satisfies_pattern(simplex, labeling, pattern))

    @staticmethod
    def simplex_satisfies_pattern(simplex, labeling, pattern):
        """
        Return true iff the colors of the simplex are subset of one of the patterns.
        simplex ... sorted tuple of vertices
        labeling ... list or dictionary giving a label to each possible vertex
        pattern ... collection of sets of labels
        """
        simplex_labels = {labeling[v] for v in simplex}
        return any(simplex_labels.issubset(s) for s in pattern)

    @staticmethod
    def pattern_translate_user_to_internal(pattern, labels_user_to_internal):
        return [set(labels_user_to_internal[lab] for lab in face) for face in pattern]

    @staticmethod
    def read_pattern_input_string(parameter: str, labels=None):
        parameter = parameter.lower().strip()
        if parameter.endswith('chromatic'):
            chromaticity = parameter.replace('chromatic', '').replace('-', '')
            words_to_numbers = {'mono': 1, 'one': 1, 'bi': 2, 'two': 2, 'tri': 3, 'three': 3, 'tetra': 4, 'four': 4}
            if chromaticity in words_to_numbers:
                chromaticity = words_to_numbers[chromaticity]
            elif chromaticity.isnumeric() and chromaticity != '0':
                chromaticity = int(chromaticity)
            else:
                raise ValueError(f"The color pattern `{parameter}` is invalid")
            if labels is None:
                labels = list(range(chromaticity))
            if labels and chromaticity > len(labels):
                chromaticity = len(labels)  # 4-chromatic subcomplex of 3-colored should still be the full complex
            pattern_list_of_sets = [set(x) for x in itertools.combinations(labels, chromaticity)]
        else:
            pattern_list_of_sets = [{int(ch) if ch.isnumeric() else ch for ch in w} for w in parameter.split(',')]

        return pattern_list_of_sets

    def read_pattern_input(self, parameter, labels=None, check_labels=True):
        if isinstance(parameter, str):
            pattern_list_of_sets = self.read_pattern_input_string(parameter)
        else:
            pattern_list_of_sets = [set(color_set) for color_set in parameter]

        if labels is not None and check_labels:  # check that the pattern only uses the given labels
            for lab in set().union(*pattern_list_of_sets):
                if lab not in labels:
                    raise ValueError(f"There is no point labeled by `{lab}`. "
                                     f"To suppress this error, pass allow_unused_labels=True to get_complex function")

        return pattern_list_of_sets
