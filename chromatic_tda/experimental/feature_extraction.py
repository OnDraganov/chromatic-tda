import numpy as np

from chromatic_tda import SimplicialComplex
from chromatic_tda.core.core_simplicial_complex import CoreSimplicialComplex
from chromatic_tda.utils.floating_point_utils import FloatingPointUtils

BLANK = ''

class FeatureExtractor:
    """A class created with a simplicial pair to extract persistent pairs."""

    def __init__(self, simplicial_complex: SimplicialComplex|CoreSimplicialComplex):
        if isinstance(simplicial_complex, CoreSimplicialComplex):
            self.simplicial_complex = simplicial_complex
        elif isinstance(simplicial_complex, SimplicialComplex):
            self.simplicial_complex = simplicial_complex.core_complex
        else:
            raise TypeError()


    def persistence_pairs(self, group: str, dim: int, sorted_by='persistence', bar_of_interest=None):
        """Return list of finite nontrivial birth-death pairs. Final means they have both birth and death,
        nontrivial means their filtration values differ.
        Parameters
        ----------
            group     ... the persistence group (sub_complex, image, complex, cokernel, relative, kernel)
            dim       ... dimension/degree of the feature
            sorted_by ... how should the list be sorted
                            - persistence ... decreasing order by length of the bar
                            - proximity   ... increasing order by proximity to the bar_of_interest
            bar_of_interest ... only used when sorted_by='proximity', the point w.r.t. which the proximity is computed,
                                it is a pair of floats describing the birth and death value
        Returns
        -------
        Ordered list of birth-death pairs.
        """
        if group not in SimplicialComplex.GROUPS:
            raise ValueError(f'Only following groups are allowed: ' + ', '.join(SimplicialComplex.GROUPS))
        if self.simplicial_complex.birth_death is None:
            raise ValueError("Persistence not yet computed, run `compute_persistence` on the simplicial complex first.")

        pairs = [(birth, death) for birth, death in self.simplicial_complex.birth_death[group]['pairs']
                 if (len(death) == dim + 2 and
                     not FloatingPointUtils.is_trivial_bar((self.simplicial_complex.get_simplex_weight(birth),
                                                            self.simplicial_complex.get_simplex_weight(death)))
                     )
                 ]

        if sorted_by == 'persistence':
            key = lambda pair: self.simplicial_complex.get_simplex_weight(pair[1]) - self.simplicial_complex.get_simplex_weight(pair[0])
            reverse = True
        elif sorted_by == 'proximity':
            if bar_of_interest is None:
                raise ValueError('If sorted_by="proximity", then bar_of_interest needs to be given')
            key = lambda pair: np.linalg.norm(np.array(bar_of_interest)
                                              - (self.simplicial_complex.get_simplex_weight(pair[0]), self.simplicial_complex.get_simplex_weight(pair[1])))
            reverse = False
        else:
            raise ValueError('sorted_by can only be "persistence" or "proximity"')

        return sorted(pairs, key=key, reverse=reverse)

    def extract_feature(self, death_simplex, group):
        """
        Return a representative of the feature killed by the given simplex.

        Parameters
        ----------
        death_simplex ... the death simplex of the feature
        group         ... the persistence group for which the feature should be extracted

        Returns
        -------
        Set of simplices.
        """
        if group == 'complex':
            return self.simplicial_complex.persistence_data[group]['reduced_matrix'][death_simplex]
        if group == 'sub_complex':
            return self.simplicial_complex.persistence_data[group]['reduced_matrix'][death_simplex]
        if group == 'image':
            return self.simplicial_complex.persistence_data[group]['reduced_matrix'][death_simplex]
        if group == 'kernel':
            death_column =  self.simplicial_complex.persistence_data[group]['reduced_matrix'][death_simplex]
            return self.simplicial_complex.chain_boundary(death_column & self.simplicial_complex.sub_complex)
        if group == 'relative':
            raise NotImplementedError("The cokernel feature extractor is not implemented.")
        if group == 'cokernel':
            raise NotImplementedError("The cokernel feature extractor is not implemented.")