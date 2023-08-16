from chromatic_tda.algorithms.persistence_algorithm import PersistenceAlgorithm
from chromatic_tda.core.core_simplicial_complex import CoreSimplicialComplex
from chromatic_tda.core.simplicial_complex_factory import CoreSimplicialComplexFactory


class SimplicialComplex:

    GROUPS = ['kernel', 'sub_complex', 'image', 'complex', 'cokernel', 'relative']

    def __init__(self, data):
        self.core_complex: CoreSimplicialComplex

        if isinstance(data, CoreSimplicialComplex):
            self.core_complex = data
        else:
            self.core_complex = CoreSimplicialComplexFactory().create_instance(data)

    def compute_persistence(self):
        PersistenceAlgorithm(simplicial_complex=self.core_complex).compute_persistence()

    def dimension(self) -> int:
        """Return the dimension of the simplicial complex"""
        return self.core_complex.get_dimension()

    def bars(self, group : str, dim : int = None, only_finite : bool = False, return_as : str = 'dict'):
        """Return persistence bars for a given group as. If dimension is given, bars are returned as a list of
         (birth, death) pairs. If no dimension is given, all dimensions are returned in one of the following formats:
            - dict ... a dictionary {dimension : list_of_bars}
            - list ... a list of tuples (dimension, (birth, death))
        Default return format is `dict`.

        Keyword arguments:
            dim ... Dimension. If an integer is given, returns a list of bars of only that dimension (default: None).
            only_finite ... If True, only return the finite bars (default: False).
            return_as ... Either 'dict' or 'list'. Switch return format as described above. (default: 'dict')

        Remark: If compute_persistence was not called on the SimplicialComplex yet, it is called.
        """
        if group not in self.GROUPS:
            raise ValueError(f'The group needs to be one of the following: '+', '.join(self.GROUPS))
        if len(self.core_complex.birth_death) == 0:
            self.compute_persistence()
        bars = self.core_complex.get_bars_list(group, dim=dim, only_finite=only_finite)
        if dim is not None or return_as == 'list':
            return bars
        elif return_as == 'dict':
            bars_dict = {}
            for dim, bar in bars:
                if dim not in bars_dict:
                    bars_dict[dim] = []
                bars_dict[dim].append(bar)
            return bars_dict
        else:
            raise ValueError('The argument `return_as` can only be "dict" or "list".')

    def bars_six_pack(self, only_finite : bool = False, return_as : str = 'dict') -> dict:
        """Return the whole six-pack as a dictionary of the form { group : bars_of_the_group }. Bars of one group
        are returned in one of the following formats:
            - dict ... a dictionary {dimension : list_of_bars}
            - list ... a list of tuples (dimension, (birth, death))
        Default return format is `dict`.

        Keyword arguments:
            only_finite ... If True, only return the finite bars (default: False).
            return_as ... Either 'dict' or 'list'. Switch return format as described above. (default: 'dict')

        Remark: If compute_persistence was not called on the SimplicialComplex yet, it is called.
        """
        return {group : self.bars(group, dim=None, only_finite=only_finite, return_as=return_as)
                for group in self.GROUPS}

    def bars_finite_1norm(self, group : str, dim : int):
        """Return the 1-norm of finite bars of the given group and dimension"""
        return sum([death - birth for birth, death in self.bars(group, dim=dim, only_finite=True)])

    def diagrams_list(self, group_dimension_pattern, only_finite=False):
        """Return list of persistence diagrams defined by the `group_dimension_pattern`. The pattern is a list of
        pairs (group, dimension). The method respects the order given.
        """
        return [self.bars(group, dim, only_finite=only_finite) for group, dim in group_dimension_pattern]

    # def get_1norm_share(self, group_numerator : str, group_denominator : str, dim : int):
    #     """Return the ratio of finite bars 1-norms of group_numerator and group_denominator"""
    #     numerator = self.bars_finite_1norm(group_numerator, dim)
    #     denominator = self.bars_finite_1norm(group_denominator, dim)
    #     return numerator/denominator if denominator else 0
    
    def simplices(self) -> list:
        """Return list of all simplices sorted by dimension and then lexicographically (w.r.t. vertex indices)."""
        return self.core_complex.get_simplices()

    def weight_function(self) -> dict:
        """Return the weight/radius function as a dictionary {simplex : weight}."""
        return self.core_complex.get_simplex_weights()

    def simplices_of_dim(self, dim : int) -> list:
        """Return list of all simplices ."""
        return self.core_complex.get_simplices_of_dim(dim)

    def sub_complex_simplices(self) -> list:
        """Return list of all simplices of the sub_complex, sorted by dimension and then
        lexicographically (w.r.t. vertex indices)."""
        return self.core_complex.get_sub_complex_simplices()

    def set_simplex_weights(self, weight_function, default_value = 0):
        """Set the weights of simplices. The `weight_function` is a dictionary {simplex : weight}.
        The `default_value` is used for all simplices not present in `weight_function`.
        USE WITH CAUTION: The method is meant for creating a new simplicial complex with your own filtration. If you
        change this, you need to manually run `compute_persistence` again."""
        self.core_complex.set_simplex_weights(weight_function, default_value)

    def set_sub_complex(self, simplices):
        """Set the sub-complex by passing its maximal (generating) simplices.
        USE WITH CAUTION: The method is meant for creating a new simplicial complex with your own filtration. If you
        change this, you need to manually run `compute_persistence` again."""
        self.core_complex.set_sub_complex(simplices)

