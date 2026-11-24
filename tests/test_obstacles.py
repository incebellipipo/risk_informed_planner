"""Analytical vessel-motion and closest-approach scenarios."""
import unittest

import numpy as np

from tests.helpers import Node
from risk_informed_planner.obstacles import LinearDynamicObstacle, StaticObstacle, compute_cpa
from risk_informed_planner.utils import smooth_path


class ObstacleTests(unittest.TestCase):
    def test_static_obstacle_does_not_move(self):
        obstacle = StaticObstacle((1, 2), (3, 4))
        for t in [0, 100]:
            np.testing.assert_array_equal(obstacle.position(t), [[1, 2], [3, 4]])

    def test_linear_motion_follows_heading_and_speed(self):
        obstacle = LinearDynamicObstacle(10, 20, np.pi/2, 3)
        np.testing.assert_allclose(obstacle.position(4), [10, 32, np.pi/2])
        np.testing.assert_allclose(obstacle.position(0), [10, 20, np.pi/2])

    def test_factory_reaches_course_endpoint_at_travel_time(self):
        obstacle = LinearDynamicObstacle.create((1, 2), (4, 6), 2.5)
        np.testing.assert_allclose(obstacle.position(2)[:2], [4, 6])
        self.assertAlmostEqual(obstacle.yaw, np.arctan2(4, 3))

    def test_head_on_collision(self):
        tcpa, dcpa, *_ = compute_cpa(np.array([0, 0, 0]), np.array([100, 0, np.pi]), 5, 5)
        self.assertAlmostEqual(tcpa, 10)
        self.assertAlmostEqual(dcpa, 0)

    def test_crossing_vessels_meet_at_the_intersection(self):
        tcpa, dcpa, *_ = compute_cpa(np.array([0, 0, 0]), np.array([10, -10, np.pi/2]), 1, 1)
        self.assertAlmostEqual(tcpa, 10)
        self.assertAlmostEqual(dcpa, 0)

    def test_equal_velocities_keep_current_separation(self):
        tcpa, dcpa, *_ = compute_cpa(np.array([0, 0, 0]), np.array([3, 4, 0]), 2, 2)
        self.assertEqual(tcpa, np.inf)
        self.assertEqual(dcpa, 5)

    def test_diverging_vessels_have_past_cpa(self):
        tcpa, dcpa, *_ = compute_cpa(np.array([0, 0, np.pi]), np.array([10, 0, 0]), 1, 1)
        self.assertLess(tcpa, 0)
        self.assertAlmostEqual(dcpa, 0)

    def test_cpa_is_invariant_under_translation_and_vessel_swap(self):
        a, b = np.array([5., 10., .3]), np.array([70., -20., 2.])
        expected = compute_cpa(a, b, 3, 4)[:2]
        shift = np.array([1000, -3000, 0])
        np.testing.assert_allclose(compute_cpa(a+shift, b+shift, 3, 4)[:2], expected)
        np.testing.assert_allclose(compute_cpa(b, a, 4, 3)[:2], expected)

    def test_straight_path_smoothing_preserves_the_line_and_endpoints(self):
        path = [Node(x, 2*x+1, x) for x in range(6)]
        x, y = smooth_path(path, degree=0)
        self.assertEqual(x.shape, (100,))
        np.testing.assert_allclose(y, 2*x+1, atol=1e-10)
        np.testing.assert_allclose([x[0], x[-1]], [0, 5], atol=1e-10)
        self.assertTrue(np.all(np.diff(x) > 0))
