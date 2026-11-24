"""Evidence discretization, grounding lookahead, and moving-vessel encounters."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tests.helpers import Node, open_graph
from risk_informed_planner.bbn import bbn_autonomous_ranked_generated as bn
from risk_informed_planner.obstacles import LinearDynamicObstacle, StaticObstacle
from risk_informed_planner.risk.dynamic_risk import DynamicRisk


class DynamicRiskTests(unittest.TestCase):
    def make_model(self, **options):
        defaults = dict(graph=open_graph(), the_network=Mock(), max_distance_from_shore=100,
                        max_dcpa_for_collision_risk=100, max_tcpa_for_collision_risk=100,
                        desired_speed=5)
        defaults.update(options)
        return DynamicRisk(**defaults)

    def node(self, distance=0):
        node = Node(0, 0, 1)
        node.parent = Node(-1, 0, 0)
        node.distance_cost = distance
        return node

    def test_equal_bins_include_boundaries_and_clamp_extremes(self):
        for value, expected in [(-1, 0), (0, 0), (19.9, 0), (20, 1), (80, 4), (100, 4), (1000, 4)]:
            with self.subTest(value=value):
                self.assertEqual(DynamicRisk._discretize_value(value, 100, 5), expected)
        self.assertIsNone(DynamicRisk._discretize_value(None, 100, 5))

    def test_reversed_and_offset_bins(self):
        self.assertEqual(DynamicRisk._discretize_value(10, 20, 5, min_value=10), 0)
        self.assertEqual(DynamicRisk._discretize_value(12, 20, 5, min_value=10), 1)
        self.assertEqual(DynamicRisk._discretize_value(0, 100, 5, reverse=True), 4)
        self.assertEqual(DynamicRisk._discretize_value(100, 100, 5, reverse=True), 0)

    def test_irregular_intervals(self):
        for value, expected in [(0, 0), (1, 1), (3, 2), (10, 3)]:
            with self.subTest(value=value):
                self.assertEqual(DynamicRisk._discretize_value_with_intervals(value, [1, 3, 10]), expected)
        self.assertIsNone(DynamicRisk._discretize_value_with_intervals(None, [1, 3]))

    def test_obstacle_lists_and_default_cost_models_are_independent(self):
        obstacles = [StaticObstacle((0, 0), (1, 1))]
        first = self.make_model(obstacles=obstacles)
        second = self.make_model()
        obstacles.clear()
        self.assertEqual(len(first.obstacles), 1)
        self.assertEqual(second.obstacles, [])
        self.assertIsNot(first.cost_model, second.cost_model)

    def test_grounding_lookahead_preserves_original_depth_predicate(self):
        for depth, expected in [(20, bn.Vessel_on_grounding_course.YES),
                                (6, bn.Vessel_on_grounding_course.NO),
                                (3, bn.Vessel_on_grounding_course.NO)]:
            with self.subTest(depth=depth):
                model = self.make_model(graph=open_graph(depth), grounding_depth_hard_limit=6,
                                        lookahead_distance=30)
                self.assertEqual(model._evidence_grounding_course(self.node()), expected)
                args = model.graph.is_free_path.call_args.args
                np.testing.assert_allclose(args[1], [30, 0])

    def test_grounding_lookahead_uses_travel_heading(self):
        model = self.make_model(lookahead_distance=10)
        node = Node(0, 0, 1)
        node.parent = Node(0, -1, 0)
        model._evidence_grounding_course(node)
        np.testing.assert_allclose(model.graph.is_free_path.call_args.args[1], [0, 10])

    def test_no_encounter_without_a_future_collision_course(self):
        scenarios = [[], [StaticObstacle((1, 2), (3, 4))],
                     [LinearDynamicObstacle(100, 200, np.pi, 5)],
                     [LinearDynamicObstacle(-100, 0, np.pi, 5)]]
        for obstacles in scenarios:
            with self.subTest(obstacles=obstacles):
                evidence = self.make_model(obstacles=obstacles)._evidence_encounter(self.node())
                self.assertEqual(list(evidence), [None, None, None, bn.Encounter.UNLIKELY])

    def test_head_on_encounter_is_detected(self):
        obstacle = LinearDynamicObstacle(100, 0, np.pi, 5)
        evidence = self.make_model(obstacles=[obstacle])._evidence_encounter(self.node())
        self.assertEqual(evidence, (bn.DCPA.STATE0, bn.TCPA.STATE0,
                                    bn.Relative_velocity.STATE4, bn.Encounter.LIKELY))

    def test_encounter_uses_obstacle_position_at_node_arrival(self):
        obstacle = LinearDynamicObstacle(100, 0, np.pi, 5)
        model = self.make_model(obstacles=[obstacle])
        with patch.object(obstacle, "position", wraps=obstacle.position) as position:
            model._evidence_encounter(self.node(distance=100))
        position.assert_called_once_with(20)

    def test_worst_encounter_prioritizes_closest_approach_over_earliest_time(self):
        # CPA at 10 seconds / 30 m, versus 20 seconds / 10 m.
        obstacles = [LinearDynamicObstacle(100, 30, np.pi, 5),
                     LinearDynamicObstacle(200, 10, np.pi, 5)]
        model = self.make_model(obstacles=obstacles)
        evidence = model._evidence_encounter(self.node())
        self.assertEqual(evidence[0], bn.DCPA.STATE0)
        self.assertEqual(evidence[1], bn.TCPA.STATE1)
        model.obstacles.reverse()
        self.assertEqual(model._evidence_encounter(self.node()), evidence)

    def test_evidence_is_cleared_before_each_query_and_shallow_water_forces_failure(self):
        network = Mock()
        model = self.make_model(graph=open_graph(3), the_network=network,
                                grounding_depth_hard_limit=6)
        network.reset_mock()
        model.set_evidence(self.node())
        self.assertEqual(network.method_calls[0][0], "clear_all_evidence")
        network.set_failure_of_recovery_from_grounding_course.assert_called_once_with(
            bn.Failure_of_recovery_from_grounding_course.YES)
        model.graph = open_graph(20)
        model.set_evidence(self.node())
        self.assertEqual(network.clear_all_evidence.call_count, 2)
        self.assertEqual(network.set_failure_of_recovery_from_grounding_course.call_count, 1)

    def test_inference_precedes_cost_evaluation(self):
        order = []
        network, cost_model = Mock(), Mock()
        model = self.make_model(the_network=network, cost_model=cost_model)
        network.update_beliefs.side_effect = lambda: order.append("inference")
        cost_model.compute_cost.side_effect = lambda net: order.append("cost") or .42
        self.assertEqual(model.compute_risk(self.node()), .42)
        self.assertEqual(order, ["inference", "cost"])
        cost_model.compute_cost.assert_called_once_with(network)
