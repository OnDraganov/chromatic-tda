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

    def restricted_complex(self, simplices):
        simplicial_complex = SimplicialComplex(
            CoreSimplicialComplexFactory().create_restricted_instance(self.core_complex, simplices)
        )
        return simplicial_complex

    def bars(self, group : str, dim : int = None, only_finite : bool = False) -> list:
        """Return persistence bars for a given group as a list with elements of the form (dimension, (birth, death)).
        Keyword arguments:
            dim ... Dimension. If an integer is given, returns a list of bars of only that dimension (default: None).
            only_finite ... If True, only return the finite bars (default: False).

        Remark: If compute_persistence was not called on the SimplicialComplex yet, it is called.
        """
        if group not in self.GROUPS:
            raise ValueError(f'The group needs to be one of the following: '+', '.join(self.GROUPS))
        if len(self.core_complex.birth_death) == 0:
            self.compute_persistence()
        return self.core_complex.get_bars_list(group, dim, only_finite)

    def bars_dict(self, group : str, only_finite : bool = False) -> dict:
        """Return persistence bars for a given group as a dictionary of the form { dimension : list of (birth, death) }.
        Keyword arguments:
            only_finite ... If True, only return the finite bars (default: False).

        Remark: If compute_persistence was not called on the SimplicialComplex yet, it is called.
        """
        bars = {}
        for dim, bar in self.bars(group, only_finite=only_finite):
            if dim not in bars:
                bars[dim] = []
            bars[dim].append(bar)
        return bars

    def bars_six_pack(self, only_finite : bool = False) -> dict:
        """Return the whole six-pack as a dictionary of the form { group : list of (dimension, (birth, death)) }.
        Keyword arguments:
            only_finite ... If True, only return the finite bars (default: False).

        Remark: If compute_persistence was not called on the SimplicialComplex yet, it is called.
        """
        return {group : self.bars(group, dim=None, only_finite=only_finite) for group in self.GROUPS}

    def bars_finite_1norm(self, group : str, dim : int):
        """Return the 1-norm of finite bars of the given group and dimension"""
        return sum([death - birth for birth, death in self.bars(group, dim=dim, only_finite=True)])

    def get_1norm_share(self, group_numerator : str, group_denominator : str, dim : int):
        """Return the ratio of finite bars 1-norms of group_numerator and group_denominator"""
        numerator = self.bars_finite_1norm(group_numerator, dim)
        denominator = self.bars_finite_1norm(group_denominator, dim)
        return numerator/denominator if denominator else 0
    
    def simplices(self) -> list:
        """Return list of all simplices sorted by dimension and then lexicographically (w.r.t. vertex indices)."""
        return self.core_complex.get_simplices()

    def simplices_of_dim(self, dim : int) -> list:
        """Return list of all simplices ."""
        return self.core_complex.get_simplices_of_dim(dim)

    def sub_complex_simplices(self) -> list:
        """Return list of all simplices of the sub_complex, sorted by dimension and then
        lexicographically (w.r.t. vertex indices)."""
        return self.core_complex.get_sub_complex_simplices()

    def vertices(self) -> list:
        """Return list of all vertices."""
        return self.core_complex.get_vertices()

    def set_simplex_weights(self, radius_function, default_value = 0):
        self.core_complex.set_simplex_weights(radius_function, default_value)

    def set_sub_complex(self, simplices):
        self.core_complex.set_sub_complex(simplices)

    def write(self):
        self.core_complex.write()

