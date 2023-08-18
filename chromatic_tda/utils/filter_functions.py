from chromatic_tda.core.core_simplicial_complex import CoreSimplicialComplex


class FilterFunctions:
    """Functions to order tuples according to float values and subset-relation."""
    def __init__(self, weight_function : dict, sub_complex : set):
        self.weight_function = weight_function
        self.sub_complex = sub_complex
        ordered_simplices = list(weight_function)
        ordered_simplices.sort(key= lambda simplex: self.weight_function[simplex])
        ordered_simplices.sort(key= lambda simplex: set(simplex))
        self.total_filtration = {simplex : index for index, simplex in enumerate(ordered_simplices)}

    def filter_function_rad(self):
        """Filter by radius."""
        return lambda simplex: self.total_filtration[simplex]

    def filter_function_rad_sub_first(self):
        """Filter by radius, but put all sub-complex simplices first, and then the rest."""
        return lambda simplex: (simplex not in self.sub_complex, self.total_filtration[simplex])
