import unittest
import numpy as np

from chromatic_tda.utils.geometry_utils import GeometryUtils, AffineSpace


class GeometryTestReflections(unittest.TestCase):
    def test_reflection_3d_through_3d(self):  # identity
        # Reflections also test orthogonal projection
        vector_space = np.array([
            [1, 1, 0],
            [2, 1, 0],
            [.4, -.123, .03]
        ])
        shift = np.array([1, 1, 1])
        points = [
            np.array([2, 3, 4]),
            np.array([0, 0, 0]),
            np.array([1, 1, 1]),
            np.array([5, 6, 1]),
            np.array([2, 1, -3]),
            np.array([1.2, 3.1, 2.789])
        ]
        reflected_points = GeometryUtils.reflect_points_through_affine_space(AffineSpace(shift, vector_space), *points)
        for pt, pt_ref in zip(points, reflected_points):
            assert np.isclose(pt, pt_ref).all()

    def test_reflection_3d_through_2d(self):
        vector_space = np.array([
            [1, 1, 0],
            [2, 1, 0]
        ])
        shift = np.array([1, 1, 1])
        points = [
            np.array([2, 3, 4]),
            np.array([0, 0, 0]),
            np.array([1, 1, 1]),
            np.array([5, 6, 1]),
            np.array([2, 1, -3]),
            np.array([1.2, 3.1, 2.789])
        ]
        reflected_points = GeometryUtils.reflect_points_through_affine_space(AffineSpace(shift, vector_space), *points)
        for pt, pt_ref in zip(points, reflected_points):
            assert np.isclose((pt - (0, 0, 1)) * (1, 1, -1) + (0, 0, 1), pt_ref).all()

    def test_reflection_3d_through_2d_rotated(self):
        rot_angle = .7
        rotation = np.array([
            [1, 0, 0],
            [0, np.cos(rot_angle), - np.sin(rot_angle)],
            [0, np.sin(rot_angle), np.cos(rot_angle)]
        ])
        vector_space = np.array([
            rotation @ [1, 1, 0],
            rotation @ [2, 1, 0]
        ])
        shift = np.array([1, 1, 1])
        points = [
            np.array([2, 3, 4]),
            np.array([0, 0, 0]),
            np.array([1, 1, 1]),
            np.array([5, 6, 1]),
            np.array([2, 1, -3]),
            np.array([1.2, 3.1, 2.789])
        ]
        reflected_points = GeometryUtils.reflect_points_through_affine_space(AffineSpace(shift, vector_space), *points)
        for pt, pt_ref in zip(points, reflected_points):
            assert np.isclose(rotation @ ((rotation.transpose() @ (pt - (1, 1, 1))) * (1, 1, -1)) + (1, 1, 1),
                              pt_ref).all()

    def test_reflection_3d_through_2d_single_point(self):
        vector_space = np.array([
            [1, 1, 1],
            [0, 0, 1]
        ])
        shift = np.array([1, 1, 1])
        point = np.array([1, 0, 0])
        reflected_points = GeometryUtils.reflect_points_through_affine_space(AffineSpace(shift, vector_space), point)
        assert len(reflected_points) == 1
        assert np.isclose(reflected_points[0], (0, 1, 0)).all()

    def test_reflection_3d_through_1d(self):
        vector_space = np.array([
            [0, 0, 102.3],
        ])
        shift = np.array([5, 6, 1])
        points = [
            np.array([2, 3, 4]),
            np.array([0, 0, 0]),
            np.array([1, 1, 1]),
            np.array([5, 6, 1]),
            np.array([2, 1, -3]),
            np.array([1.2, 3.1, 2.789])
        ]
        reflected_points = GeometryUtils.reflect_points_through_affine_space(AffineSpace(shift, vector_space), *points)
        for pt, pt_ref in zip(points, reflected_points):
            assert np.isclose((pt - (5, 6, 0)) * (-1, -1, 1) + (5, 6, 0), pt_ref).all()

    def test_reflection_3d_through_0d(self):  # reflection through the point [1, 1, 1]
        vector_space = np.zeros(shape=(0, 3))
        shift = np.array([1, 1, 1])
        points = [
            np.array([2, 3, 4]),
            np.array([0, 0, 0]),
            np.array([1, 1, 1]),
            np.array([5, 6, 1]),
            np.array([2, 1, -3]),
            np.array([1.2, 3.1, 2.789])
        ]
        reflected_points = GeometryUtils.reflect_points_through_affine_space(AffineSpace(shift, vector_space), *points)
        for pt, pt_ref in zip(points, reflected_points):
            assert np.isclose(- (pt - (1, 1, 1)) + (1, 1, 1), pt_ref).all()

    def test_circumcircle_of_weighted_basic_triangle(self):
        points = np.array([
            [1, 2, 3],
            [1, 1, 1],
            [4, 3, 2]
        ])
        weights = np.array([0, 0, 0])
        x, rad2 = GeometryUtils.circumsphere_of_weighted_points(points, weights)
        assert np.allclose(np.square(points - x).sum(axis=1), rad2)

    def test_circumcircle_of_weighted_basic_weighted_triangle(self):
        points = np.array([
            [1, 2, 3],
            [1, 1, 1],
            [4, 3, 2]
        ])
        weights = np.array([.3, -1.1, 0])
        x, rad2 = GeometryUtils.circumsphere_of_weighted_points(points, weights)
        print(np.square(points[0] - x).sum() + weights[0])
        print(np.square(points - x).sum(axis=1) + weights)
        print(rad2)
        assert np.allclose(np.square(points - x).sum(axis=1) + weights, rad2)

    def test_circumcircle_of_weighted_4d_line(self):
        points = np.array([
            [1, 2, 3, 0],
            [1, 0, 1, 1]
        ])
        weights = np.array([0, 0])
        x, rad2 = GeometryUtils.circumsphere_of_weighted_points(points, weights)
        assert np.allclose(np.square(points - x).sum(axis=1), rad2)
        assert np.isclose(9 / 4, rad2)

    def test_circumcircle_of_weighted_4d_weighted_line(self):
        points = np.array([
            [1, 2, 3, 0],
            [1, 0, 1, 1]
        ])
        weights = np.array([2, 5])
        x, rad2 = GeometryUtils.circumsphere_of_weighted_points(points, weights)
        assert np.allclose(np.square(points - x).sum(axis=1) + weights, rad2)
