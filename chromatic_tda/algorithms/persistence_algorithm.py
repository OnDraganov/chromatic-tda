from chromatic_tda.core.core_simplicial_complex import CoreSimplicialComplex
from chromatic_tda.utils.filter_functions import FilterFunctions
from chromatic_tda.algorithms.reduce_matrix import ReduceMatrixAlgorithm


class PersistenceAlgorithm:  # why is this not a singleton? with complex as a parameter for each function?
    complex : CoreSimplicialComplex
    filters : FilterFunctions

    def __init__(self, simplicial_complex : CoreSimplicialComplex) -> None:
        self.complex = simplicial_complex
        self.filters = FilterFunctions(simplicial_complex)

    def _compute_birth_death_complex(self) -> None:
        Rf = self.complex.persistence_data['complex']['R']
        low_inv_f = self.complex.persistence_data['complex']['low_inv']

        birth = set(s for s in Rf if len(Rf[s]) == 0)
        death = set(s for s in Rf if len(Rf[s]) != 0)
        pairs = set((k, v) for k, v in low_inv_f.items())
        killed = set(pair[0] for pair in pairs)
        essential = birth - killed
        
        self.complex.birth_death['complex'] = {
            'birth' : birth,
            'death' : death,
            'essential' : essential,
            'pairs' : pairs
        }

    def _compute_birth_death_sub_complex(self) -> None:
        Rg = self.complex.persistence_data['sub_complex']['R']
        low_inv_g = self.complex.persistence_data['sub_complex']['low_inv']

        birth = set(s for s in Rg if len(Rg[s]) == 0)
        death = set(s for s in Rg if len(Rg[s]) != 0)
        pairs = set((k, v) for k, v in low_inv_g.items())
        killed = set(pair[0] for pair in pairs)
        essential = birth - killed

        self.complex.birth_death['sub_complex'] = {
            'birth' : birth,
            'death' : death,
            'essential' : essential,
            'pairs' : pairs
        }

    def _compute_birth_death_image(self) -> None:
        Rg = self.complex.persistence_data['sub_complex']['R']
        low_inv_im = self.complex.persistence_data['image']['low_inv']

        birth = set(s for s in Rg if len(Rg[s]) == 0)
        pairs = set((k, v) for k, v in low_inv_im.items() if k in self.complex.sub_complex)
        death = set(pair[1] for pair in pairs)
        killed = set(pair[0] for pair in pairs)
        essential = birth - killed

        self.complex.birth_death['image'] = {
            'birth' : birth,
            'death' : death,
            'essential' : essential,
            'pairs' : pairs
        }

    def _compute_birth_death_kernel(self) -> None:
        Rf = self.complex.persistence_data['complex']['R']
        Rg = self.complex.persistence_data['sub_complex']['R']
        low_inv_im = self.complex.persistence_data['image']['low_inv']
        low_inv_ker = self.complex.persistence_data['kernel']['low_inv']

        birth = set(v for k, v in low_inv_im.items() if (v not in self.complex.sub_complex and
                                                         k in self.complex.sub_complex))
        pairs = set((k, v) for k, v in low_inv_ker.items() if (v in self.complex.sub_complex and  # tau in L
                                                               len(Rg[v]) != 0 and  # tau negative in Rg
                                                               len(Rf[v]) == 0))  # tau positive in Rf
        death = set(pair[1] for pair in pairs)
        killed = set(pair[0] for pair in pairs)
        essential = birth - killed

        self.complex.birth_death['kernel'] = {
            'birth' : birth,
            'death' : death,
            'essential' : essential,
            'pairs' : pairs
        }

    def _compute_birth_death_cokernel(self) -> None:
        Rim = self.complex.persistence_data['image']['R']
        low_im = {v:k for k,v in self.complex.persistence_data['image']['low_inv'].items()}
        Rg = self.complex.persistence_data['sub_complex']['R']
        low_inv_cok = self.complex.persistence_data['cokernel']['low_inv']

        birth = set(s for s in Rim if len(Rim[s]) == 0 and (s not in self.complex.sub_complex or len(Rg[s]) != 0))
        pairs = set((k, v) for k, v in low_inv_cok.items()
                    if len(Rim[v]) != 0 and low_im[v] not in self.complex.sub_complex)
        death = set(pair[1] for pair in pairs)
        killed = set(pair[0] for pair in pairs)
        essential = birth - killed
        self.complex.birth_death['cokernel'] = {
            'birth' : birth,
            'death' : death,
            'essential' : essential,
            'pairs' : pairs
        }

    def _compute_persistence_relative(self) -> None:
        R = self.complex.persistence_data['relative']['R']
        low_inv = self.complex.persistence_data['relative']['low_inv']

        birth = set(s for s in R if len(R[s]) == 0)
        death = set(s for s in R if len(R[s]) != 0)
        pairs = set((k, v) for k, v in low_inv.items())
        killed = set(pair[0] for pair in pairs)
        essential = birth - killed

        self.complex.birth_death['relative'] = {
            'birth' : birth,
            'death' : death,
            'essential' : essential,
            'pairs' : pairs
        }

    def _compute_persistence_data(self) -> None:
        self.complex.persistence_data = {}
        
        # TODO (low priority): Only compute what is needed. The prerequisites graph is as follows:
        #   complex    -R--> image
        #   complex    -R--> relative
        #   complex    -RV-> kernel
        #   complex    -R--> co_kernel
        #   sub_complex -RV-> co_kernel
        # Now user can give a list of desired groups, we add prerequisites to the list
        # and then only compute the data from the list. ! Also need to change compute_persistence !

        self.complex.persistence_data['complex'] = ReduceMatrixAlgorithm().reduce(
            self.complex.boundary,
            order_function=self.filters.filter_function_rad(),
            return_V=True)

        self.complex.persistence_data['sub_complex'] = ReduceMatrixAlgorithm().reduce(
            {simplex : self.complex.boundary[simplex] for simplex in self.complex.sub_complex},  # boundary matrix of sub-complex
            order_function=self.filters.filter_function_rad(),
            return_V=True)

        self.complex.persistence_data['image'] = ReduceMatrixAlgorithm().reduce(
            self.complex.persistence_data['complex']['R'],  # instead of (self.complex.boundary) to have more performance
            order_function=self.filters.filter_function_rad(),
            order_function_row=self.filters.filter_function_rad_sub_first(),
            return_V=False)

        R = self.complex.persistence_data['complex']['R'] # should be V_im, but coincides on cycles with V_f
        V = self.complex.persistence_data['complex']['V'] # should be R_im, but coincides on cycles with R_f
        self.complex.persistence_data['kernel'] = ReduceMatrixAlgorithm().reduce(
            {simplex : V[simplex] for simplex in V if len(R[simplex]) == 0},  # columns of V which represent cycles
            order_function=self.filters.filter_function_rad(),
            order_function_row=self.filters.filter_function_rad_sub_first(),
            return_V=False)

        Vg = self.complex.persistence_data['sub_complex']['V']
        Rg = self.complex.persistence_data['sub_complex']['R']
        Df = self.complex.persistence_data['complex']['R']  # We can take R rather than D, because we only replace
                                                            # cycle columns, so all reductions are still valid.
        D_cok = {simplex : Vg[simplex] if (simplex in Rg and len(Rg[simplex]) == 0) else Df[simplex] for simplex in Df}
        self.complex.persistence_data['cokernel'] = ReduceMatrixAlgorithm().reduce(
            D_cok,
            order_function=self.filters.filter_function_rad(),
            return_V=False)

        matrix_relative = {s : {t for t in self.complex.persistence_data['complex']['R'][s] if t not in self.complex.sub_complex}
                                  for s in self.complex.persistence_data['complex']['R']    if s not in self.complex.sub_complex}
        self.complex.persistence_data['relative'] = ReduceMatrixAlgorithm().reduce(
            matrix_relative,
            order_function=self.filters.filter_function_rad(),
            return_V=False)

    def compute_persistence(self) -> None:
        """
        Compute the following persistence homology:
            complex -- PH of the whole complex
            sub_complex -- PH only of the sub_complex
            kernel -- kernel PH
            image  -- image PH
            cokernel -- co_kernel PH
        """
        self._compute_persistence_data()
        self.complex.birth_death = {}
        self._compute_birth_death_complex()
        self._compute_birth_death_sub_complex()
        self._compute_birth_death_image()
        self._compute_birth_death_kernel()
        self._compute_birth_death_cokernel()
        self._compute_persistence_relative()

    def get_birth_death_from_matrix(self, R, low_inv) -> dict:
        """Given a matrix R a low_inv function,
        return birth, death and essential simplices."""
        birth = set(s for s in R if len(R[s]) == 0)
        death = set(s for s in R if len(R[s]) != 0)
        essential = set(s for s in birth if s not in low_inv.keys())

        return {
            'birth' : birth,
            'death' : death,
            'essential' : essential
        }
