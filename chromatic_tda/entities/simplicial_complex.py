import numpy as np

from chromatic_tda.algorithms.chromatic_subcomplex_utils import ChromaticComplexUtils
from chromatic_tda.algorithms.persistence_algorithm import PersistenceAlgorithm
from chromatic_tda.core.core_simplicial_complex import CoreSimplicialComplex
from chromatic_tda.core.simplicial_complex_factory import CoreSimplicialComplexFactory


class SimplicialComplex:

    GROUPS = ['kernel', 'sub_complex', 'image', 'complex', 'cokernel', 'relative']
    core_complex: CoreSimplicialComplex

    def __init__(self, data):
        if isinstance(data, CoreSimplicialComplex):
            self.core_complex = data
        else:
            self.core_complex = CoreSimplicialComplexFactory().create_instance(data)

    def __iter__(self):
        yield from self.core_complex

    def __len__(self) -> int:
        return len(self.core_complex)

    def __contains__(self, element) -> bool:
        return element in self.core_complex

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
            raise ValueError(f'The group needs to be one of the following: ' + ', '.join(self.GROUPS))
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

    def weight_function(self, simplex=None):
        """If simplex is given, return the weight/radius of the simplex.
        If no simplex is given, return the weight/radius function as a dictionary {simplex : weight}."""
        if simplex is None:
            return self.core_complex.get_weight_function_copy()
        return self.core_complex.get_simplex_weight(simplex)

    def simplices_of_dim(self, dim : int) -> list:
        """Return list of all simplices ."""
        return self.core_complex.get_simplices_of_dim(dim)

    def simplices_sub_complex(self) -> list:
        """Return list of all simplices of the sub_complex, sorted by dimension and then
        lexicographically (w.r.t. vertex indices)."""
        return self.core_complex.get_sub_complex_simplices()

    def is_in_sub_complex(self, simplex) -> bool:
        """Return True iff the given simplex is in the sub-complex."""
        return simplex in self.core_complex.sub_complex

    def set_simplex_weights(self, weight_function : dict, default_value : float = 0):
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

    def get_chromatic_subcomplex(self, labeling, sub_complex=None, full_complex=None, relative=None,
                                 allow_unused_labels=False):
        """Ignoring the current subcomplex, return a new SimplicialComplex as a chromatic subcomplex given by
        the parameters (see docstring of ChromaticAlphaComplex.get_simplicial_complex).
        Argument labeling is a dictionary or a list such that labeling[vertex] = label;
        A previously defined sub_complex of the complex is NOT preserved."""
        return SimplicialComplex(ChromaticComplexUtils.get_chromatic_subcomplex(
            internal_labeling=labeling, sub_complex=sub_complex, full_complex=full_complex, relative=relative,
            simplicial_complex=self.core_complex, allow_unused_labels=allow_unused_labels)
        )
