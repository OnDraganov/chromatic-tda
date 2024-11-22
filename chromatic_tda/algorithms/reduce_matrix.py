
class MatrixReduction:
    @staticmethod
    def reduce(matrix, order_function, order_function_row = None, return_reduction_matrix = False):
        """
        Reduce given matrix. Using `order_function` to order columns.
        If `order_function_row` is given, it is used to order rows,
        otherwise `order_function` is also used for rows.
        The matrix is copied, i.e., given `matrix` remains unchanged.

        Returns a dictionary with the following keys:
            reduced_matrix ... the reduced boundary matrix
            pivots ... pivots of the reduced matrix as a dictionary: row indices as keys and column indices as values
            reduction_matrix (if return_reduction_matrix=True) ... the matrix V s.t. reduced_matrix = matrix * V
        """
        if order_function_row is None:
            order_function_row = order_function
        R = {k : v for k, v in matrix.items()}
        if return_reduction_matrix:
            V = {k : {k} for k in matrix}
        low_inv = {}  # low_inv[i]=index of column with the lowest 1 at i
        for s in sorted(R, key=order_function):
            t = low_inv.get(max(R[s], key=order_function_row), -1) if len(R[s]) != 0 else -1
            while t != -1:
                R[s] = R[t] ^ R[s]  # symmetric difference of t-th and s-th columns
                if return_reduction_matrix:
                    V[s] = V[t] ^ V[s]
                t = low_inv.get(max(R[s], key=order_function_row), -1) if len(R[s]) != 0 else -1
            if len(R[s]) != 0:
                low_inv[max(R[s], key=order_function_row)] = s

        return_dictionary = {'reduced_matrix': R, 'pivots': low_inv}
        if return_reduction_matrix:
            return_dictionary['reduction_matrix'] = V
        return return_dictionary
