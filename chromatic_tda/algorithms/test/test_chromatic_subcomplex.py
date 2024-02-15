import unittest

from chromatic_tda.entities.simplicial_complex import SimplicialComplex


class ComplexTest(unittest.TestCase):

    def test_chromatic_subcomplex_mono_chromatic(self) -> None:
        simplicial_complex = SimplicialComplex([(0, 1, 2), (1, 2, 3)])

        chromatic_subcomplex = simplicial_complex.get_chromatic_subcomplex([0, 1, 1, 0], sub_complex='mono-chromatic')

        assert set(chromatic_subcomplex.simplices()) == {(0,), (1,), (2,), (3,), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3),
                                                         (0, 1, 2), (1, 2, 3)}
        assert set(chromatic_subcomplex.simplices_sub_complex()) == {(0,), (3,), (1,), (2,), (1, 2)}

    def test_chromatic_subcomplex_complex_mono_chromatic(self) -> None:
        simplicial_complex = SimplicialComplex([(0, 1, 2), (1, 2, 3)])

        chromatic_subcomplex = simplicial_complex.get_chromatic_subcomplex([0, 1, 1, 0], full_complex='mono-chromatic')

        assert set(chromatic_subcomplex.simplices()) == {(0,), (3,), (1,), (2,), (1, 2)}
        assert set(chromatic_subcomplex.simplices_sub_complex()) == set()

    def test_chromatic_subcomplex_relative_mono_chromatic(self) -> None:
        simplicial_complex = SimplicialComplex([(0, 1, 2), (1, 2, 3)])
        simplicial_complex.set_sub_complex([(0, 1, 2)])

        chromatic_subcomplex = simplicial_complex.get_chromatic_subcomplex([0, 1, 1, 0], relative='mono-chromatic')

        assert set(chromatic_subcomplex.simplices()) == {(0, 1), (0, 2), (1, 3), (2, 3), (0, 1, 2), (1, 2, 3)}
        assert set(chromatic_subcomplex.simplices_sub_complex()) == set()

    def test_chromatic_subcomplex_bi_to_tri_over_mono(self) -> None:
        simplicial_complex = SimplicialComplex([(0, 1, 2, 3, 4), (2, 3, 5)])
        chromatic_subcomplex = simplicial_complex.get_chromatic_subcomplex([0, 1, 2, 3, 3, 2],
                                                                           sub_complex='bi-chromatic',
                                                                           full_complex='tri-chromatic',
                                                                           relative='mono-chromatic')

        assert set(chromatic_subcomplex.simplices()) == {(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3),
                                                         (2, 4), (3, 5), (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3),
                                                         (0, 2, 4), (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4),
                                                         (2, 3, 4), (2, 3, 5), (0, 1, 3, 4), (0, 2, 3, 4), (1, 2, 3, 4)}
        assert set(chromatic_subcomplex.simplices_sub_complex()) == {(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3),
                                                                     (1, 4), (2, 3), (2, 4), (3, 5), (0, 3, 4),
                                                                     (1, 3, 4), (2, 3, 4), (2, 3, 5)}

    def test_chromatic_subcomplex_custom_labels(self) -> None:
        simplicial_complex = SimplicialComplex([(0, 1, 2), (1, 2, 3)])

        chromatic_subcomplex = simplicial_complex.get_chromatic_subcomplex(['blue', 'red', 'red', 'blue'],
                                                                           sub_complex='mono-chromatic')

        assert set(chromatic_subcomplex.simplices()) == {(0,), (1,), (2,), (3,), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3),
                                                         (0, 1, 2), (1, 2, 3)}
        assert set(chromatic_subcomplex.simplices_sub_complex()) == {(0,), (3,), (1,), (2,), (1, 2)}

    def test_chromatic_subcomplex_01_2(self) -> None:
        simplicial_complex = SimplicialComplex([(0, 1, 2), (1, 2, 3), (2, 3, 4), (1, 3, 5), (4, 5)])

        chromatic_subcomplex = simplicial_complex.get_chromatic_subcomplex([0, 1, 1, 0, 2, 2],
                                                                           sub_complex='01,2')
        assert set(chromatic_subcomplex.simplices_sub_complex()) == {(0,), (1,), (2,), (3,), (4,), (5,), (0, 1), (0, 2),
                                                                     (1, 2), (1, 3), (2, 3), (4, 5), (0, 1, 2),
                                                                     (1, 2, 3)}

    def test_chromatic_subcomplex_ab_c(self) -> None:
        simplicial_complex = SimplicialComplex([(0, 1, 2), (1, 2, 3), (2, 3, 4), (1, 3, 5), (4, 5)])

        chromatic_subcomplex = simplicial_complex.get_chromatic_subcomplex(['a', 'b', 'b', 'a', 'c', 'c'],
                                                                           sub_complex='ab,c')
        assert set(chromatic_subcomplex.simplices_sub_complex()) == {(0,), (1,), (2,), (3,), (4,), (5,), (0, 1), (0, 2),
                                                                     (1, 2), (1, 3), (2, 3), (4, 5), (0, 1, 2),
                                                                     (1, 2, 3)}

    def test_chromatic_subcomplex_dictionary_labels(self) -> None:
        simplicial_complex = SimplicialComplex([(0, 1, 2), (1, 2, 3)])

        chromatic_subcomplex = simplicial_complex.get_chromatic_subcomplex({
            0: '0', 1: '1', 2: '1', 3: '0'
        }, sub_complex='mono-chromatic')

        assert set(chromatic_subcomplex.simplices()) == {(0,), (1,), (2,), (3,), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3),
                                                         (0, 1, 2), (1, 2, 3)}
        assert set(chromatic_subcomplex.simplices_sub_complex()) == {(0,), (3,), (1,), (2,), (1, 2)}
