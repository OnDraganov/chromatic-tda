import unittest
import numpy as np

from chromatic_tda.utils.linear_algebra_utils import LinAlgUtils


class LinearAlgebraTest(unittest.TestCase):
    a_mat = np.array([
        [7.56248052, 3.01108843, 0.56261590, 0.32744947, 0.23429895, 0.18101955, 0.70902646, 0.10934863, 0.9035789],
        [4.34061090, 1.47696429, 0.91088442, 0.99761031, 0.65215561, 0.42838886, 0.62356712, 0.97818994, 0.9252501],
        [1.58343582, 0.62058533, 0.24465723, 0.92581940, 0.98481243, 0.94776809, 0.39252661, 0.84119163, 0.8511891],
        [2.80497347, 2.49306469, 0.97397928, 0.55287025, 0.77001101, 0.59266880, 0.71048821, 0.43799465, 0.0107083]
    ])
    b_vec = np.array([0.57413191, 0.79050492, 2.42991792, 8.25227247])

    def test_solution_full_rank_4x9(self):
        a_mat = LinearAlgebraTest.a_mat
        b_vec = LinearAlgebraTest.b_vec
        x, ker = LinAlgUtils.solve(a_mat, b_vec)

        assert np.isclose(a_mat @ x, b_vec).all()
        assert len(ker) == 5
        for v in ker:
            assert np.isclose(a_mat @ (x + v), b_vec).all()

    def test_solution_singular_8x9_stack(self):
        a_mat = np.concatenate([LinearAlgebraTest.a_mat, LinearAlgebraTest.a_mat])
        b_vec = np.concatenate([LinearAlgebraTest.b_vec, LinearAlgebraTest.b_vec])
        x, ker = LinAlgUtils.solve(a_mat, b_vec)

        assert np.isclose(a_mat @ x, b_vec).all()
        assert len(ker) == 5
        for v in ker:
            assert np.isclose(a_mat @ (x + v), b_vec).all()

    def test_solution_singular_7x9(self):
        coef_mat = np.array([
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [0.3005, 0.6383, 0.6640, 0.9111]
        ])
        a_mat = np.concatenate([LinearAlgebraTest.a_mat, coef_mat @ LinearAlgebraTest.a_mat])
        b_vec = np.concatenate([LinearAlgebraTest.b_vec, coef_mat @ LinearAlgebraTest.b_vec])
        x, ker = LinAlgUtils.solve(a_mat, b_vec)

        assert np.isclose(a_mat @ x, b_vec).all()
        assert len(ker) == 5
        for v in ker:
            assert np.isclose(a_mat @ (x + v), b_vec).all()

    def test_solution_singular_4x12(self):
        coef_mat = np.array([
            [0.0262, 0.8473, 0.5723, 0.4596, 0.2175, 0.7477, 0.4135, 0.7733, 0.8522],
            [0.0933, 0.9487, 0.2949, 0.5585, 0.1369, 0.3590, 0.7447, 0.9125, 0.3629],
            [0.4308, 0.2242, 2.2360, 0.7456, 0.6092, 0.7564, 5.9848, 0.6998, 0.8237]
        ]).transpose()
        a_mat = np.concatenate([LinearAlgebraTest.a_mat, LinearAlgebraTest.a_mat @ coef_mat], axis=1)
        b_vec = LinearAlgebraTest.b_vec
        x, ker = LinAlgUtils.solve(a_mat, b_vec)

        assert np.isclose(a_mat @ x, b_vec).all()
        assert len(ker) == 8
        for v in ker:
            assert np.isclose(a_mat @ (x + v), b_vec).all()

    def test_solution_full_rank_9x4(self):
        a_mat = LinearAlgebraTest.a_mat.transpose()
        b_vec = a_mat @ [1, 2, 3, 4]

        x, ker = LinAlgUtils.solve(a_mat, b_vec)

        assert np.isclose(x, [1, 2, 3, 4]).all()
        assert len(ker) == 0

    def test_solution_singular_9x7(self):
        coef_mat = np.array([
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [0.3005, 0.6383, 0.6640, 0.9111]
        ])
        a_mat = np.concatenate([LinearAlgebraTest.a_mat, coef_mat @ LinearAlgebraTest.a_mat]).transpose()
        b_vec = a_mat @ [1, 2, 3, 4, 0, 0, 0]

        x, ker = LinAlgUtils.solve(a_mat, b_vec)

        assert np.isclose(a_mat @ x, b_vec).all()
        assert len(ker) == 3
        for v in ker:
            assert np.isclose(a_mat @ (x + v), b_vec).all()

    def test_orthogonalize_4x9(self):
        a_mat = self.a_mat
        ortho = LinAlgUtils.orthogonalize_rows(a_mat)
        rank_ortho = (~np.isclose(np.linalg.svd(ortho).S, 0)).sum()
        rank_ortho_a_mat = (~np.isclose(np.linalg.svd(np.concatenate([a_mat, ortho])).S, 0)).sum()

        assert ortho.shape == (4, 9)
        assert rank_ortho == 4 and rank_ortho_a_mat == 4  # row spaces match
        assert np.isclose(ortho @ ortho.transpose(), np.identity(4)).all()  # rows are orthonormal

    def test_orthogonalize_2x9(self):
        a_mat = self.a_mat[:2]
        ortho = LinAlgUtils.orthogonalize_rows(a_mat)
        rank_ortho = (~np.isclose(np.linalg.svd(ortho).S, 0)).sum()
        rank_ortho_a_mat = (~np.isclose(np.linalg.svd(np.concatenate([a_mat, ortho])).S, 0)).sum()

        assert ortho.shape == (2, 9)
        assert rank_ortho == 2 and rank_ortho_a_mat == 2  # row spaces match
        assert np.isclose(ortho @ ortho.transpose(), np.identity(2)).all()  # rows are orthonormal

    def test_orthogonalize_singular_7x9(self):
        coef_mat = np.array([
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [0.3005, 0.6383, 0.6640, 0.9111]
        ])
        a_mat = np.concatenate([LinearAlgebraTest.a_mat, coef_mat @ LinearAlgebraTest.a_mat])
        ortho = LinAlgUtils.orthogonalize_rows(a_mat)
        rank_ortho = (~np.isclose(np.linalg.svd(ortho).S, 0)).sum()
        rank_ortho_a_mat = (~np.isclose(np.linalg.svd(np.concatenate([a_mat, ortho])).S, 0)).sum()

        assert ortho.shape == (4, 9)
        assert rank_ortho == 4 and rank_ortho_a_mat == 4  # row spaces match
        assert np.isclose(ortho @ ortho.transpose(), np.identity(4)).all()  # rows are orthonormal
