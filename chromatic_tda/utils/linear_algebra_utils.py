import numpy as np
import numpy.typing as npt
from numpy.linalg._linalg import SVDResult

from chromatic_tda.utils.floating_point_utils import FloatingPointUtils
from chromatic_tda.utils.timing import TimingUtils


class LinAlgUtils:
    @staticmethod
    def solve(a_matrix: npt.NDArray, b_vector: npt.NDArray, check_solution=False) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Solve a general linear equation Ax=b using singular value decomposition.
        Particular solution computed via pseudo-inverse A^+ as A^+ @ b. This makes sense even if the equation
        has no solution, and returns the least squares solution. By default, no warning is given -- see check_solution.

        :param a_matrix:        matrix A
        :param b_vector:        right side b
        :param check_solution:  if True, Ax == b check is performed for the particular solution x,
                                and error raised if it fails

        :return: A tuple (x, kernel) where x is a particular solution, and kernel is an orthonormal basis of Ker(A)
        (vectors in rows).
        """
        TimingUtils().start("LinAlg :: Solve Linear Equation With Kernel")
        svd : SVDResult = np.linalg.svd(a_matrix)  # svd.U @ diagonal from svd.S @ svd.Vh == A
        rank : int = LinAlgUtils.count_nonzero(svd.S)

        pseudoinverse = LinAlgUtils.pseudoinverse_from_svd(svd)
        x = pseudoinverse @ b_vector
        kernel = svd.Vh[rank :]  # the vectors that Vh sends to E_{rank+1}, ..., E_{n}: Vhh @ E_i is i-th row of Vh

        if check_solution and not LinAlgUtils.check_solution(a_matrix, b_vector, x):
            raise np.linalg.LinAlgError("There is no exact solution to given Ax=b.")

        TimingUtils().stop("LinAlg :: Solve Linear Equation With Kernel")
        return x, kernel

    @staticmethod
    def orthogonalize_rows(array: npt.NDArray) -> npt.NDArray:
        """Given m x n array A with m <= n,
        return array whose rows are orthonormal basis of the row-space of A.
        WARNING: for rank(A) < m might return smaller space if R in np.linalg.qr would skip a pivot and use it later"""
        TimingUtils().start("LinAlg :: Orthogonalize Rows")
        m, n = array.shape
        if m > n:
            raise ValueError("Only m x n arrays with m <= n allowed")
        qr = np.linalg.qr(array.transpose(), mode='reduced')
        non_zero_r_diagonal = ~np.array([FloatingPointUtils.is_close(x, 0) for x in qr.R.diagonal()], dtype=bool)
        TimingUtils().stop("LinAlg :: Orthogonalize Rows")
        return qr.Q.transpose()[non_zero_r_diagonal]

    @staticmethod
    def count_nonzero(array: npt.NDArray) -> int:
        return int(np.prod(array.shape)) - sum(int(FloatingPointUtils.is_close(0, x)) for x in array)

    @staticmethod
    def check_solution(a_matrix: npt.NDArray, b_vector: npt.NDArray, x_vector: npt.NDArray):
        return FloatingPointUtils.is_all_close(a_matrix @ x_vector, b_vector)

    @staticmethod
    def pseudoinverse_from_svd(svd: SVDResult) -> npt.NDArray:
        TimingUtils().start("LinAlg :: Pseudoinverse From SVD")
        pseudo_s = np.zeros(shape=(svd.U.shape[1], svd.Vh.shape[0]))
        np.fill_diagonal(pseudo_s, LinAlgUtils.pseudoinverse_of_diagonal(svd.S))
        pseudoinverse = (svd.U @ pseudo_s @ svd.Vh).transpose()
        TimingUtils().stop("LinAlg :: Pseudoinverse From SVD")
        return pseudoinverse

    @staticmethod
    def pseudoinverse_of_diagonal(diagonal: npt.NDArray) -> npt.NDArray:
        return np.array([s ** -1 if not FloatingPointUtils.is_close(s, 0) else 0 for s in diagonal])
