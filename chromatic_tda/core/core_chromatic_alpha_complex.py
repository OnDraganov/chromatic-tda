import numpy as np
import numpy.typing as npt

from chromatic_tda.algorithms.chromatic_subcomplex_utils import ChromaticComplexUtils
from chromatic_tda.core.core_simplicial_complex import CoreSimplicialComplex
from chromatic_tda.utils.legacy_geometrical_utils import sq_dist


class CoreChromaticAlphaComplex:
    points: npt.NDArray
    points_dimension: int
    input_labels_to_internal_labels_dict: dict
    internal_labels_to_input_labels_dict: dict
    labels_number: int
    internal_labeling: list
    simplicial_complex: CoreSimplicialComplex

    def __init__(self) -> None:
        self.input_labels_to_internal_labels_dict = {}
        self.labels_number = 0
        self.internal_labeling = []
        self.sq_rad = {}

    def __iter__(self):
        yield from self.simplicial_complex

    def __len__(self) -> int:
        return len(self.simplicial_complex)

    def __contains__(self, element) -> bool:
        return element in self.simplicial_complex

    def copy_points(self):
        """Return a copy of the points"""
        return np.array(self.points)

    def simplex_labels_internal(self, simplex):
        return {self.internal_labeling[v] for v in simplex}

    def simplex_labels_input(self, simplex):
        """Return set of labels of the vertices of the given simplex"""
        return {self.internal_labels_to_input_labels_dict[lab] for lab in self.simplex_labels_internal(simplex)}

    def get_simplicial_complex(self, sub_complex, full_complex, relative, allow_unused_labels) -> CoreSimplicialComplex:
        return ChromaticComplexUtils.get_chromatic_subcomplex(
            sub_complex=sub_complex, full_complex=full_complex, relative=relative,
            simplicial_complex=self.simplicial_complex, internal_labeling=self.internal_labeling,
            labels_user_to_internal=self.input_labels_to_internal_labels_dict,
            allow_unused_labels=allow_unused_labels
        )

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
        for mono_chrom_face in self.OLD_split_simplex(simplex):  # for every color
            if len(mono_chrom_face) > 0:  # if the color is present in simplex
                radius = sq_dist(center, self.points[mono_chrom_face[0]])
                if any((self.internal_labeling[mono_chrom_face[0]] == self.internal_labeling[v] and
                        sq_dist(center, self.points[v]) < radius)
                       for v in vertices_to_check):
                    return False
        return True

    def OLD_split_simplex(self, simplex) -> list:
        return [tuple(i for i in simplex if self.internal_labeling[i] == 0),
                tuple(i for i in simplex if self.internal_labeling[i] == 1),
                tuple(i for i in simplex if self.internal_labeling[i] == 2)]

    def OLD_split_simplex_sort_by_size(self, simplex) -> list:
        return sorted(self.OLD_split_simplex(simplex), key=len, reverse=True)

    def write(self) -> None:
        print()
        print(f"**** Delaunay Complex (dimension = {self.simplicial_complex.get_dimension()}) ****")
        for (p, l) in zip(self.points, self.internal_labeling):
            print(f"{p} -> {l}")
        print(42 * "*")
        print()
