import unittest
import numpy as np

from chromatic_tda.utils.geometry_utils import GeometryUtils


class GeometryTestReflections(unittest.TestCase):
    def test_reflection_3d_through_3d(self):  # identity
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
        reflected_points = GeometryUtils.reflect_points_through_affine_space(shift, vector_space, *points)
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
        reflected_points = GeometryUtils.reflect_points_through_affine_space(shift, vector_space, *points)
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
        reflected_points = GeometryUtils.reflect_points_through_affine_space(shift, vector_space, *points)
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
        reflected_points = GeometryUtils.reflect_points_through_affine_space(shift, vector_space, point)
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
        reflected_points = GeometryUtils.reflect_points_through_affine_space(shift, vector_space, *points)
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
        reflected_points = GeometryUtils.reflect_points_through_affine_space(shift, vector_space, *points)
        for pt, pt_ref in zip(points, reflected_points):
            assert np.isclose(- (pt - (1, 1, 1)) + (1, 1, 1), pt_ref).all()
