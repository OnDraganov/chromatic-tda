import numpy as np

from chromatic_tda.utils.boundary_matrix_utils import BoundaryMatrixUtils
from chromatic_tda.utils.floating_point_utils import FloatingPointUtils


class CoreSimplicialComplex:
    dim_simplex_dict: dict[int, set]
    simplex_weights: dict[tuple, float]
    boundary: dict[tuple, set]
    co_boundary: dict[tuple, set]
    sub_complex: set
    persistence_data: dict
    birth_death: dict
    dimension: int

    def __init__(self) -> None:
        self.clear()

    def __iter__(self):
        yield from self.get_simplices()

    def __len__(self) -> int:
        return len(self.boundary)

    def __contains__(self, item) -> bool:
        return item in self.boundary

    def clear(self) -> None:
        self.dim_simplex_dict = {}
        self.simplex_weights = {}  # radius values of simplices

        self.boundary = {}  # example: {(1,2,3) : {(1,2), (1,3), (2,3)}, (1,2) : {(1), (2)}, ...}
        self.co_boundary = {}  # example:  {(1,2) : {(1,2,3), (1,2,4)}, (1,3) : {(1,2,3), (1,3,4)}, ...}

        self.sub_complex = set()
        self.persistence_data = {}
        self.birth_death = {}

        self.dimension = 0

    def clear_empty_dimensions(self) -> None:
        clear_dims = []
        for dim in self.dim_simplex_dict:
            if len(self.dim_simplex_dict[dim]) == 0:
                clear_dims.append(dim)
        for dim in clear_dims:
            self.dim_simplex_dict.pop(dim)

    def get_bars_list(self, group: str, dim: int = None, only_finite: bool = False) -> list:
        """
        Return list of pairs representing bars of a given dimension.
        If `dim` is not given, returns all bars in form `(dim, (birth, death))`.
        The groups to choose from (`group`) are:
            complex, sub_complex, image, kernel, cokernel, relative
        """
        if self.birth_death is None:
            raise ValueError("Persistence not yet computed, run `compute_persistence` first.")
        if group not in self.birth_death:
            raise ValueError(f"Persistence for `{group}` not computed. Did you run `compute_persistence`?")

        dim_bars_finite: list[tuple[int, tuple[float, float]]] = [
            (len(s) - 1 if group != 'kernel' else len(s) - 2,
             (self.simplex_weights[s], self.simplex_weights[t])) for s, t in self.birth_death[group]['pairs']
        ]
        if only_finite:
            dim_bars_infinite = []
        else:
            dim_bars_infinite: list[tuple[int, tuple[float, float]]] = [
                (len(s) - 1 if group != 'kernel' else len(s) - 2,
                 (self.simplex_weights[s], np.inf)) for s in self.birth_death[group]['essential']
            ]
        dim_bars = dim_bars_finite + dim_bars_infinite

        if dim is None:
            return sorted([b for b in dim_bars if not FloatingPointUtils.is_trivial_bar(b[1])])
        else:
            bars: list[tuple] = [bar for bar_dim, bar in dim_bars if bar_dim == dim]
            return sorted([b for b in bars if not FloatingPointUtils.is_trivial_bar(b)])

    def get_simplices(self) -> list:
        """Return list of all simplices sorted by dimension and then lexicographically."""
        return sorted(self.boundary, key=lambda s: (len(s), s))

    def get_simplices_of_dim(self, dim: int) -> list:
        return sorted(self.dim_simplex_dict.get(dim, []))

    def get_simplices_of_dim_count(self, dim: int) -> int:
        return len(self.dim_simplex_dict.get(dim, []))

    def get_sub_complex_simplices(self) -> list:
        """Return list of all simplices of the sub_complex, sorted by dimension and then lexicographically."""
        return sorted(self.sub_complex, key=lambda s: (len(s), s))

    def set_simplex_weights(self, weight_function: dict, default_value: float = 0) -> None:
        """
        Add weights/radii of simplices from a dictionary `simplex : weight`.
        Add default_value to all simplices not present in the weight function.

        Note: Monotonicity is NOT checked.
        """
        self.simplex_weights = {simplex: default_value for simplex in self.boundary}
        for simplex, weight in weight_function.items():
            simplex_tuple = tuple(sorted(simplex))
            if simplex_tuple not in self.boundary:
                raise ValueError(f"Simplex {simplex_tuple} is not in the simplicial complex, cannot add its weight.")
            if weight < default_value:
                raise Warning(f"Simplex {simplex_tuple} given weight lower than default.")
            self.simplex_weights[simplex_tuple] = weight
        co_boundary = (self.co_boundary if self.co_boundary
                       else BoundaryMatrixUtils.make_co_boundary(self.boundary))
        FloatingPointUtils.ensure_weights_monotonicity_and_equal_values(self.simplex_weights, co_boundary)

    def get_weight_function_copy(self) -> dict:
        """Return copy of {simplex : weight} dictionary."""
        return {simplex: weight for simplex, weight in self.simplex_weights.items()}

    def get_simplex_weight(self, simplex: tuple[int, ...]) -> float:
        """Return the weight of a simplex."""
        if simplex in self.simplex_weights:
            return self.simplex_weights[simplex]
        else:
            raise KeyError(f'{simplex} is not a simplex in the simplicial complex (or has no wight attached to it).')

    def get_extra_vertices_of_cofaces(self, simplex: tuple[int, ...]) -> list[int, ...]:
        return [(set(co_face) - set(simplex)).pop() for co_face in self.co_boundary[simplex]]

    def set_sub_complex(self, simplices) -> None:
        """Sets a sub_complex generated by given list of simplices."""
        simplices = set(tuple(sorted(s)) for s in simplices)
        queue = sorted(simplices, key=lambda s: (len(s), s))
        while queue:
            simplex = queue.pop()
            if simplex not in self.boundary:
                raise KeyError(f"Simplex {simplex} is not in the simplicial complex, cannot add into sub-complex.")
            simplex_boundary = self.boundary[simplex]
            for sub_simplex in simplex_boundary:
                if sub_simplex not in simplices:
                    simplices.add(sub_simplex)
                    queue.append(sub_simplex)
        self.sub_complex = simplices

    def set_total_sub_complex(self, vertices) -> None:
        """Sets a sub_complex as the total sub_complex given by a list of vertices."""
        vertices = set(vertices)
        self.sub_complex = set(s for s in self.boundary if set(s).issubset(vertices))

    def get_dimension(self) -> int:
        return max(self.dim_simplex_dict, default=-1)

    def write(self) -> None:
        print()
        print(f"*** Simplicial Complex (dimension = {self.get_dimension()})")
        for dim in sorted(self.dim_simplex_dict):
            print(
                f"Simplices with dimension {dim} ({self.get_simplices_of_dim_count(dim)}): {self.get_simplices_of_dim(dim)}")
