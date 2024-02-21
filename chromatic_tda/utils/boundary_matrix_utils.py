from chromatic_tda.utils.timing import TimingUtils


class BoundaryMatrixUtils:
    @staticmethod
    def print_matrix(matrix, order_function):
        order = sorted(matrix, key=order_function)
        for row in order:
            for column in order:
                print('1' if row in matrix[column] else '0', end='')
            print()

    @staticmethod
    def make_co_boundary(boundary):  # add_co_bnd function
        TimingUtils().start("Make Co-Boundary")

        co_boundary = {simplex: set() for simplex in boundary}
        for simplex in boundary:
            for face in boundary[simplex]:
                co_boundary[face].add(simplex)

        TimingUtils().stop("Make Co-Boundary")
        return co_boundary
