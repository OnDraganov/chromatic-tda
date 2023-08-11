import numpy as np

from chromatic_tda.utils.singleton import singleton
from chromatic_tda.utils.timing import TimingUtils


@singleton
class GeometricalUtils:

    def intersect_lines(self, p, p_dir, q, q_dir):
        """Return intersection of two lines in 2D given by a point and
        a direction.
        Return None if the directions are parallel.
        """
        TimingUtils().start("GeometricalUtils - intersect lines")
        t,s = np.linalg.solve(np.transpose((p_dir, -q_dir)), (q-p)) # p + t*p_dir = q + s*q_dir
        TimingUtils().stop("GeometricalUtils - intersect lines")
        return p + t*p_dir

    def project_to_line(self, p, p_dir, q):
        """Return orthogonal projection of point q into 2D line given
        by a point p and a direction p_dir."""
        TimingUtils().start("GeometricalUtils - project to line")
        res = self.intersect_lines(p, p_dir, q, np.array([-p_dir[1],p_dir[0]]))
        TimingUtils().stop("GeometricalUtils - project to line")
        return res

    def bisector(self, p, q):
        """Return the bisector of points p,q as a point and a direction.
        The point is the midpoint between p and q."""
        TimingUtils().start("GeometricalUtils - bisector")
        dif = q-p
        res = (q+p)/2, np.array([-dif[1], dif[0]])
        TimingUtils().stop("GeometricalUtils - bisector")
        return res

    def circum_center(self, p, q, r):
        """Return circum center of the triangle pqr."""
        TimingUtils().start("GeometricalUtils - circum center")
        res = self.intersect_lines(*self.bisector(p,q), *self.bisector(q,r))
        TimingUtils().stop("GeometricalUtils - circum center")
        return res

    def sq_dist(self, p, q):
        """Return squared distance of two points."""
        TimingUtils().start("GeometricalUtils - sq dist")
        if len(p)!=len(q):
            raise ValueError("Both points need to be of the same dimension.")
        res = sum(i**2 for i in p-q)
        TimingUtils().stop("GeometricalUtils - sq dist")
        return res

    def sq_dist_max(self, p, *args):
        """Return the max squared distance of p and points in args"""
        TimingUtils().start("GeometricalUtils - sq dist max")
        res = max( (self.sq_dist(p, q) for q in args), default=0 )
        TimingUtils().stop("GeometricalUtils - sq dist max")
        return res
