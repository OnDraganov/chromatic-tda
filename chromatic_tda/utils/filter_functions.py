class FilterFunctions:
    """Functions to order tuples according to float values and subset-relation."""
    def __init__(self, weight_function : dict, sub_complex : set):
        self.weight_function = weight_function
        self.sub_complex = sub_complex
        self.total_filtration = {simplex : index for index, simplex in enumerate(
            sorted(weight_function.keys(), key= lambda simplex: (weight_function[simplex], len(simplex), simplex))
        )}

    def filter_function_rad(self):
        """Filter by radius."""
        return lambda simplex: self.total_filtration[simplex]

    def filter_function_rad_sub_first(self):
        """Filter by radius, but put all sub-complex simplices first, and then the rest."""
        return lambda simplex: (simplex not in self.sub_complex, self.total_filtration[simplex])
