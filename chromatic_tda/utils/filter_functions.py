from chromatic_tda.core.core_simplicial_complex import CoreSimplicialComplex


class FilterFunctions():
    def __init__(self, simplicial_complex: CoreSimplicialComplex):
        self.complex = simplicial_complex

    def filter_function_rad(self):
        """Filter by radius, then dimension, then lexicographic."""
        return lambda s: (self.complex.simplex_weights[s], len(s), s)

    def filter_function_rad_sub_first(self):
        """Put all sub_complex simplices first, and then the rest.
        Filter within those blocks by radius, then dimension, then
        lexicographic."""
        return lambda s: (s not in self.complex.sub_complex, self.complex.simplex_weights[s], len(s), s)

    def filter_function_lex(self):
        """Filter by dimension, then lexicographic."""
        return lambda s: (len(s), s)

    def filter_function_lex_sub_first(self):
        """Put all sub_complex simplices first, and then the rest.
        Filter within those blocks by dimension, then lexicographic."""
        return lambda s: (s not in self.complex.sub_complex, len(s), s)

    def filter_function_dim_rad(self):
        """Filter by dimension, then radius, then lexicographic."""
        return lambda s: (len(s), self.complex.simplex_weights[s], s)
