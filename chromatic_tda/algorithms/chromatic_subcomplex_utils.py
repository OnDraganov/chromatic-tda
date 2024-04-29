import itertools

from chromatic_tda.core.core_simplicial_complex import CoreSimplicialComplex
from chromatic_tda.core.simplicial_complex_factory import CoreSimplicialComplexFactory


class ChromaticComplexUtils:
    @staticmethod
    def get_chromatic_subcomplex(sub_complex, full_complex, relative,
                                 simplicial_complex: CoreSimplicialComplex, internal_labeling,
                                 labels_user_to_internal=None, allow_unused_labels=False) -> CoreSimplicialComplex:
        list_of_input_labels = ChromaticComplexUtils.construct_list_of_labels(
            internal_labeling=internal_labeling, labels_user_to_internal=labels_user_to_internal)
        if (full_complex is None or full_complex == '' or
                (isinstance(full_complex, str) and full_complex.lower().strip() == 'all')):
            complex_simplices = set(simplicial_complex.boundary)
        else:
            pattern = ChromaticComplexUtils.read_pattern_input(full_complex, list_of_input_labels,
                                                               check_labels=not allow_unused_labels)
            if labels_user_to_internal is not None:
                pattern = ChromaticComplexUtils.pattern_translate(pattern=pattern,
                                                                  translation_function=labels_user_to_internal)
            complex_simplices = ChromaticComplexUtils.select_simplices_with_chromatic_pattern(
                simplices=simplicial_complex.boundary.keys(), labeling_function=internal_labeling, pattern=pattern)

        if relative is None or relative == '':
            relative_simplices = set()
        elif isinstance(relative, str) and relative.lower().strip() == 'all':
            relative_simplices = set(simplicial_complex.boundary)
        else:
            pattern = ChromaticComplexUtils.read_pattern_input(relative, list_of_input_labels,
                                                               check_labels=not allow_unused_labels)
            if labels_user_to_internal is not None:
                pattern = ChromaticComplexUtils.pattern_translate(pattern=pattern,
                                                                  translation_function=labels_user_to_internal)
            relative_simplices = ChromaticComplexUtils.select_simplices_with_chromatic_pattern(
                simplices=simplicial_complex.boundary.keys(), labeling_function=internal_labeling, pattern=pattern)

        if sub_complex is None or sub_complex == '':
            sub_complex_simplices = set()
        elif isinstance(sub_complex, str) and sub_complex.lower().strip() == 'all':
            sub_complex_simplices = set(simplicial_complex.boundary)
        else:
            pattern = ChromaticComplexUtils.read_pattern_input(sub_complex, list_of_input_labels,
                                                               check_labels=not allow_unused_labels)
            if labels_user_to_internal is not None:
                pattern = ChromaticComplexUtils.pattern_translate(pattern=pattern,
                                                                  translation_function=labels_user_to_internal)
            sub_complex_simplices = ChromaticComplexUtils.select_simplices_with_chromatic_pattern(
                simplices=simplicial_complex.boundary.keys(), labeling_function=internal_labeling, pattern=pattern)

        restricted_complex: CoreSimplicialComplex = CoreSimplicialComplexFactory().create_restricted_instance(
            simplicial_complex, complex_simplices - relative_simplices)
        restricted_complex.set_sub_complex(sub_complex_simplices - relative_simplices)

        return restricted_complex

    @staticmethod
    def construct_list_of_labels(internal_labeling, labels_user_to_internal):
        if labels_user_to_internal is not None:
            return set(labels_user_to_internal.keys())
        if isinstance(internal_labeling, dict):
            return set(internal_labeling.values())
        return set(internal_labeling)

    @staticmethod
    def read_pattern_input(parameter, labels=None, check_labels=True):
        if isinstance(parameter, str):
            pattern_list_of_sets = ChromaticComplexUtils.read_pattern_input_string(parameter, labels=labels)
        else:
            pattern_list_of_sets = [set(color_set) for color_set in parameter]

        if labels is not None and check_labels:  # check that the pattern only uses the given labels
            for lab in set().union(*pattern_list_of_sets):
                if lab not in labels:
                    raise ValueError(f"There is no point labeled by `{lab}`. "
                                     f"To suppress this error, pass allow_unused_labels=True to get_complex function")

        return pattern_list_of_sets

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
                raise ValueError(
                    f"'k-chromatic' option given with labels=None. List of labels is needed for this option.")
            if labels and chromaticity > len(labels):
                chromaticity = len(labels)  # 4-chromatic subcomplex of 3-colored should still be the full complex
            pattern_list_of_sets = [set(x) for x in itertools.combinations(labels, chromaticity)]
        else:
            pattern_list_of_sets = [{int(ch) if ch.isnumeric() else ch for ch in w} for w in parameter.split(',')]

        return pattern_list_of_sets

    @staticmethod
    def simplex_satisfies_pattern(simplex, labeling_function, pattern):
        """
        Return true iff the colors of the simplex are subset of one of the patterns.
        simplex ... sorted tuple of vertices
        labeling ... list or dictionary giving a label to each possible vertex
        pattern ... collection of sets of labels
        """
        simplex_labels = {labeling_function[v] for v in simplex}
        return any(simplex_labels.issubset(s) for s in pattern)

    @staticmethod
    def select_simplices_with_chromatic_pattern(simplices, labeling_function, pattern):
        return set(simplex for simplex in simplices
                   if ChromaticComplexUtils.simplex_satisfies_pattern(simplex, labeling_function, pattern))

    @staticmethod
    def pattern_translate(pattern, translation_function):
        return [set(translation_function[lab] for lab in face if lab in translation_function) for face in pattern]

    @staticmethod
    def split_simplex_by_labels(simplex: tuple, labeling) -> dict:
        simplex_split = {}
        for v in simplex:
            lab = labeling[v]
            if lab not in simplex_split:
                simplex_split[lab] = []
            simplex_split[lab].append(v)

        return simplex_split
