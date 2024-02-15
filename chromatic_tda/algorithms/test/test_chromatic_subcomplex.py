import unittest

from chromatic_tda.entities.simplicial_complex import SimplicialComplex


class ComplexTest(unittest.TestCase):

    def test_chromatic_subcomplex_mono_chromatic(self) -> None:
        simplicial_complex = SimplicialComplex([(0, 1, 2), (1, 2, 3)])

        chromatic_subcomplex = simplicial_complex.get_chromatic_subcomplex([0, 1, 1, 0],
                                                                           sub_complex='mono-chromatic',
                                                                           full_complex=None,
                                                                           relative=None)

        assert set(chromatic_subcomplex.simplices_sub_complex()) == {(0,), (3,), (1,), (2,), (1, 2)}
