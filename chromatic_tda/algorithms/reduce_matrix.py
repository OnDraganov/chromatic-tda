from chromatic_tda.utils.singleton import singleton

@singleton
class ReduceMatrixAlgorithm():

    ######################
    def reduce(self, matrix, order_function, order_function_row = None, return_V = False):
        '''
        Reduce given matrix. Using `order_function` to order columns.
        If `order_function_row` is given, it is used to order rows,
        otherwise `order_function` is also used for rows.
        The matrix is copied, i.e., given `matrix` remains unchanged.
        '''
        if order_function_row == None: order_function_row = order_function
        R = { k : v for k,v in matrix.items()}
        if return_V:
            V = { k : {k} for k in matrix}
        low_inv = {} # low_inv[i]=index of column with the lowest 1 at i
        for s in sorted(R, key=order_function):
            t = low_inv.get(max(R[s], key=order_function_row),-1) if len(R[s])!=0 else -1
            while t!=-1:
                R[s] = R[t]^R[s] # symmetric difference of t-th and s-th columns
                if return_V:
                    V[s] = V[t]^V[s]
                t = low_inv.get(max(R[s], key=order_function_row),-1) if len(R[s])!=0 else -1
            if len(R[s])!=0:
                low_inv[max(R[s], key=order_function_row)] = s

        return_dictionary = {}
        return_dictionary['R'] = R
        if return_V:
            return_dictionary['V'] = V
        return_dictionary['low_inv'] = low_inv
        return return_dictionary
