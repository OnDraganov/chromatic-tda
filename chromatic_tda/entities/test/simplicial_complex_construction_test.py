import unittest
import numpy as np

from chromatic_tda.entities.simplicial_complex import SimplicialComplex


class ComplexTest(unittest.TestCase):

    def test_custom_complex_without_weights_list(self) -> None:
        simplicial_complex = SimplicialComplex([(0, 2, 1), (3, 1, 2)])  # out of order to check sorting

        assert set(simplicial_complex.simplices()) == {(0,), (1,), (2,), (3,), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3),
                                                       (0, 1, 2), (1, 2, 3)}
        assert set(simplicial_complex.simplices_of_dim(0)) == {(0,), (1,), (2,), (3,)}
        assert set(simplicial_complex.simplices_of_dim(1)) == {(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)}
        assert set(simplicial_complex.simplices_of_dim(2)) == {(0, 1, 2), (1, 2, 3)}

    def test_custom_complex_without_weights_ndarray(self) -> None:
        simplicial_complex = SimplicialComplex(np.array([(0, 2, 1), (3, 1, 2)]))  # out of order to check sorting

        assert set(simplicial_complex.simplices()) == {(0,), (1,), (2,), (3,), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3),
                                                       (0, 1, 2), (1, 2, 3)}
        assert set(simplicial_complex.simplices_of_dim(0)) == {(0,), (1,), (2,), (3,)}
        assert set(simplicial_complex.simplices_of_dim(1)) == {(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)}
        assert set(simplicial_complex.simplices_of_dim(2)) == {(0, 1, 2), (1, 2, 3)}

    def test_custom_complex_with_weights(self) -> None:
        simplicial_complex = SimplicialComplex({
            (1, 2): 1,
            (3,): 2,
            (1, 3): 2,
            (2, 3): 2,
            (1, 2, 3): 3,
            (0, 1, 2): 4
        })

        assert simplicial_complex.weight_function() == {(0, 1, 2): 4,
                                                        (1, 2, 3): 3,
                                                        (0, 1): 0,
                                                        (2, 3): 2,
                                                        (0, 2): 0,
                                                        (1, 2): 1,
                                                        (1, 3): 2,
                                                        (1,): 0,
                                                        (2,): 0,
                                                        (0,): 0,
                                                        (3,): 2}
