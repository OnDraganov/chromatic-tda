import numpy as np


class LinAlgUtils:
    @staticmethod
    def solve(a_matrix: np.ndarray, b_vector: np.ndarray, check_solution=False) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve a general linear equation Ax=b using singular value decomposition.
        Particular solution computed via pseudo-inverse A^+ as A^+ @ b. This makes sense even if the equation
        has no solution, and returns the least squares solution. By default, no warning is given -- see check_solution.

        :param a_matrix:        the matrix A
        :param b_vector:        the right side b
        :param check_solution:  if True, Ax == b check is performed for the particular solution x,
                                and error raised if it fails

        :return: A tuple (x, kernel) where x is a particular solution, and kernel is an orthonormal basis of Ker(A).
        """
        svd : np.linalg.linalg.SVDResult = np.linalg.svd(a_matrix)  # svd.U @ diagonal from svd.S @ svd.Vh == A
        rank : int = LinAlgUtils.count_nonzero(svd.S)

        pseudoinverse = LinAlgUtils.pseudoinverse_from_svd(svd)
        x = pseudoinverse @ b_vector
        kernel = svd.Vh[rank + 1 :]  # the vectors that Vh sends to E_{rank+1}, ..., E_{n}: Vhh @ E_i is i-th row of Vh

        if check_solution and not LinAlgUtils.check_solution(a_matrix, b_vector, x):
            raise np.linalg.LinAlgError("There is no exact solution to given Ax=b.")

        return x, kernel

    @staticmethod
    def count_nonzero(array: np.ndarray) -> int:
        return np.prod(array.shape) - np.isclose(array, 0).sum()

    @staticmethod
    def check_solution(a_matrix: np.ndarray, b_vector: np.ndarray, x_vector: np.ndarray):
        return np.isclose(a_matrix @ x_vector, b_vector)

    @staticmethod
    def pseudoinverse_from_svd(svd: np.linalg.linalg.SVDResult) -> np.ndarray:
        pseudo_s = np.zeros(shape=(svd.U.shape[1], svd.Vh.shape[0]))
        np.fill_diagonal(pseudo_s, LinAlgUtils.pseudoinverse_of_diagonal(svd.S))
        return (svd.U @ pseudo_s @ svd.Vh).transpose()

    @staticmethod
    def pseudoinverse_of_diagonal(diag: np.ndarray) -> np.ndarray:
        return np.array([s ** -1 if not np.isclose(s, 0) else 0 for s in diag])
