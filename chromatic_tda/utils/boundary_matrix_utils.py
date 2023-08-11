from chromatic_tda.utils.singleton import singleton
from chromatic_tda.utils.timing import TimingUtils

@singleton
class BoundaryMatrixUtils():
    def __len__(self, boundary):
        return len(boundary)

    def __contains__(self, boundary, item):
        return item in boundary
    
    def __iter__(self, boundary):
        return (simplex for simplex in boundary)

    def make_co_boundary(self, boundary):     # add_co_bnd function
        TimingUtils().start("Make Co-Boundary")

        co_boundary = { simplex : set() for simplex in boundary}
        for simplex in boundary:
            for face in boundary[simplex]:
                co_boundary[face].add(simplex)

        TimingUtils().stop("Make Co-Boundary")
        return co_boundary

    def get_restricted_boundary(self, complex, simplices_set: set):
        return { simplex : complex.boundary[simplex] & simplices_set for simplex in set(complex.boundary) & simplices_set }

    def print_matrix(self, matrix, order_function):
        order = sorted(matrix, key=order_function)
        for row in order:
            for column in order:
                print('1' if row in matrix[column] else '0', end='')
            print()
