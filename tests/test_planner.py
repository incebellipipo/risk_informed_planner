"""Deterministic planner decisions, edge costs, and risk cutoff regressions."""
import math
import unittest
from unittest.mock import patch

import numpy as np

from tests.helpers import Node, make_planner, open_graph
from risk_informed_planner.planner import RiskAwareAngleCostRRTStarSearch, turn_angle


class PlannerTests(unittest.TestCase):
    def test_turn_angles_include_wrapping_and_degenerate_segments(self):
        cases = [((2, 0), 0), ((1, 1), np.pi/2), ((1, -1), np.pi/2),
                 ((0, 0), np.pi), ((1, 0), 0)]
        for end, angle in cases:
            with self.subTest(end=end):
                self.assertAlmostEqual(turn_angle(Node(0, 0, 0), Node(1, 0, 1), Node(*end, 2)), angle)
        self.assertEqual(turn_angle(Node(1, 0, 0), Node(1, 0, 1), Node(2, 1, 2)), 0)
        self.assertAlmostEqual(turn_angle(Node(1, .001, 0), Node(0, 0, 1), Node(-1, .001, 2)),
                               2*np.arctan(.001))

    def test_steering_caps_length_without_overshooting(self):
        planner = make_planner(STEP_SIZE=5)
        self.assertEqual(planner.steer(planner.start, (3, 4)), (3, 4))
        self.assertEqual(planner.steer(planner.start, (6, 8)), (3, 4))
        self.assertEqual(planner.steer(planner.start, (1, 0)), (1, 0))
        self.assertIsNone(planner.steer(planner.start, (0, 0)))

    def test_neighbors_use_radius_and_exclude_the_query_itself(self):
        planner = make_planner(NEIGHBOR_RADIUS=5)
        query = Node(0, 0, 3)
        inside, boundary, outside = Node(1, 0, 1), Node(3, 4, 2), Node(6, 0, 4)
        planner.nodes = [outside, boundary, planner.start, inside]
        self.assertEqual(planner.get_neighbors(query), [boundary, inside])
        self.assertEqual(planner.get_closest_neighbors(query, 1), [inside])

    def test_sampling_empty_free_space_fails_explicitly(self):
        graph = open_graph()
        graph.free_space_mask.return_value = np.zeros((2, 2), dtype=bool)
        with self.assertRaisesRegex(RuntimeError, "No free space"):
            make_planner(graph=graph).sample_point()

    def test_depth_collision_check_preserves_original_ten_metre_threshold(self):
        method = RiskAwareAngleCostRRTStarSearch.FreeCheckMethod.DEPTH
        for depth, expected_collision in [(3, True), (6, True), (7, True),
                                          (9.9, True), (10, False), (15, False)]:
            with self.subTest(depth=depth):
                planner = make_planner(graph=open_graph(depth), method=method, min_depth=6)
                self.assertEqual(planner.check_collision(0, 0, 10, 0), expected_collision)

    def test_original_study_policy_samples_depth_but_checks_occupancy(self):
        method = RiskAwareAngleCostRRTStarSearch.FreeCheckMethod.DEPTH_SAMPLING_OCCUPANCY_CHECKS
        graph = open_graph(depth=3)
        planner = make_planner(graph=graph, method=method, min_depth=6)
        self.assertEqual(graph.free_space_mask.call_args.kwargs['layer_name'], 'depth')
        self.assertEqual(graph.free_space_mask.call_args.kwargs['threshold'], 6)
        self.assertFalse(planner.check_collision(0, 0, 10, 0))
        self.assertNotIn('layer_name', graph.is_free_path.call_args.kwargs)

    def test_occupancy_collision_blocks_an_edge(self):
        graph = open_graph()
        graph.is_free_path.side_effect = None
        graph.is_free_path.return_value = False
        self.assertTrue(make_planner(graph=graph).check_collision(0, 0, 10, 0))

    def test_sample_count_respects_spacing_degenerate_edges_and_cap(self):
        planner = make_planner(RISK_SAMPLE_SPACING=5, MAX_RISK_SAMPLES_PER_EDGE=3)
        for length, expected in [(0, 1), (4, 1), (5, 1), (6, 2), (100, 3)]:
            with self.subTest(length=length):
                self.assertEqual(planner.num_risk_samples(length), expected)
        self.assertEqual(make_planner(RISK_SAMPLE_SPACING=None).num_risk_samples(100), 1)

    def test_edge_sampling_sees_interior_risk_in_every_aggregation_mode(self):
        for aggregation, expected in [("mean", .35), ("max", .8), ("endpoint", .3)]:
            with self.subTest(aggregation=aggregation):
                values = iter([.1, .8, .2, .3])
                planner = make_planner(lambda node: next(values), RISK_SAMPLE_SPACING=5,
                                       RISK_AGGREGATION=aggregation)
                risk, peak = planner.compute_edge_risk(planner.start, Node(20, 0, 1))
                self.assertAlmostEqual(risk, expected)
                self.assertAlmostEqual(peak, .8)
                self.assertEqual(planner.risk_evaluations, 4)
                self.assertEqual(len(planner.risk_compute_times), 4)

    def test_samples_preserve_heading_arrival_distance_and_endpoint_snapshot(self):
        planner = make_planner(RISK_SAMPLE_SPACING=5, record=True)
        planner.start.distance_cost = 50
        end = Node(20, 0, 1)
        planner.compute_edge_risk(planner.start, end)
        samples = [c.kwargs["node"] for c in planner.risk_model.compute_risk.call_args_list]
        self.assertEqual([n.pos() for n in samples], [(5, 0), (10, 0), (15, 0), (20, 0)])
        self.assertEqual([n.distance_cost for n in samples], [55, 60, 65, 70])
        self.assertTrue(all(n.parent.y == n.y and n.parent.x < n.x for n in samples))
        self.assertEqual(end.bbn_snapshot["Collision"]["values"], [.1, .9])
        planner.risk_model.bbn_network.snapshot.assert_called_once()

    def test_recording_disabled_does_not_copy_bbn_snapshots(self):
        planner = make_planner(record=False)
        end = Node(10, 0, 1)
        planner.compute_edge_risk(planner.start, end)
        self.assertEqual(end.bbn_snapshot, {})
        planner.risk_model.bbn_network.snapshot.assert_not_called()

    def test_turn_penalty_is_additive_to_distance_and_normalized_risk(self):
        planner = make_planner(lambda node: .5, ALPHA=.2, BETA=4, GAMMA=.6,
                               TURN_REFERENCE_LENGTH=100)
        # Reconstruct with a risk ceiling of 2 rather than relying on model units.
        planner.risk_model.cost_model.compute_worst_case_cost.return_value = 2.0
        planner = RiskAwareAngleCostRRTStarSearch(
            planner.start, planner.goal, planner.cfg, planner.risk_model, planner.graph,
            callback_functions=[])
        parent = Node(10, 0, 1)
        parent.parent = planner.start
        parent.cost, parent.distance_cost = 7, 12
        edge = planner.compute_cost(parent, Node(10, 10, 2))
        self.assertAlmostEqual(edge.heuristic, 40)  # distance 2 + risk exposure 8 + turn 30
        self.assertAlmostEqual(edge.cost, 47)
        self.assertAlmostEqual(edge.distance_cost, 22)
        self.assertAlmostEqual(edge.risk, .5)

    def test_root_has_no_turn_penalty(self):
        planner = make_planner(GAMMA=10)
        self.assertEqual(planner.compute_turn_cost(planner.start, Node(0, 10, 1)), (0, 0))

    def test_over_limit_turn_is_rejected_before_bbn_evaluation(self):
        planner = make_planner(MAX_TURN_ANGLE=np.pi/4)
        parent = Node(10, 0, 1)
        parent.parent = planner.start
        edge = planner.compute_cost(parent, Node(10, 10, 2))
        self.assertFalse(edge.is_feasible())
        self.assertTrue(math.isinf(edge.peak_risk))
        planner.risk_model.compute_risk.assert_not_called()

    def test_parent_selection_minimizes_cost_and_ignores_disconnected_nodes(self):
        planner = make_planner(ALPHA=1)
        connected, orphan, target = Node(9, 0, 1), Node(9.5, 0, 2), Node(10, 0, 3)
        connected.parent, connected.cost = planner.start, 2
        planner.nodes += [connected, orphan]
        parent, edge = planner.choose_parent(target)
        self.assertIs(parent, connected)
        self.assertEqual(edge.cost, 3)

    def test_parent_selection_returns_none_when_every_connection_is_blocked(self):
        planner = make_planner()
        planner.graph.is_free_path.side_effect = None
        planner.graph.is_free_path.return_value = False
        self.assertEqual(planner.choose_parent(Node(5, 0, 1)), (None, None))
        planner.risk_model.compute_risk.assert_not_called()

    def test_rewiring_uses_a_cheaper_feasible_parent(self):
        planner = make_planner(ALPHA=1)
        new, neighbor = Node(4, 0, 1), Node(5, 0, 2)
        new.parent, new.cost, new.distance_cost = planner.start, 4, 4
        neighbor.parent, neighbor.cost = planner.start, 20
        planner.nodes += [new, neighbor]
        planner.rewire(new)
        self.assertIs(neighbor.parent, new)
        self.assertEqual(neighbor.cost, 5)
        self.assertEqual(neighbor.distance_cost, 5)

    def test_rewiring_rejects_interior_risk_even_when_mean_is_below_cutoff(self):
        planner = make_planner(lambda node: .9 if node.x < 5 else 0,
                               ALPHA=1, RISK_SAMPLE_SPACING=.5, RISK_CUTOFF=.5)
        new, neighbor = Node(4, 0, 1), Node(5, 0, 2)
        new.parent, new.cost = planner.start, 4
        neighbor.parent, neighbor.cost = planner.start, 20
        planner.nodes += [new, neighbor]
        planner.rewire(new)
        self.assertIs(neighbor.parent, planner.start)
        self.assertEqual(neighbor.cost, 20)

    def test_run_connects_goal_and_returns_ordered_path(self):
        planner = make_planner(record=True)
        # Exhaustion fails promptly if a regression asks for additional iterations.
        with patch.object(planner, "sample_point", side_effect=[(9, 0)]):
            nodes, path = planner.run()
        self.assertEqual([n.pos() for n in path], [(0, 0), (9, 0), (10, 0)])
        self.assertEqual(len(nodes), 3)
        self.assertTrue(planner.goal.bbn_snapshot)
        self.assertIs(path[-1].parent, path[-2])

    def test_run_does_not_connect_a_goal_above_the_risk_cutoff(self):
        planner = make_planner(lambda node: 1.0 if node.x == 10 else 0.0,
                               MAX_NODES=2, RISK_CUTOFF=.5)
        with patch.object(planner, "sample_point", side_effect=[(9, 0)]):
            nodes, path = planner.run()
        self.assertEqual(path, [])
        self.assertIsNone(planner.goal.parent)
        self.assertNotIn(planner.goal, nodes)

    def test_high_risk_samples_remain_data_points_without_becoming_path_parents(self):
        planner = make_planner(lambda node: 1.0 if node.y else 0.0, RISK_CUTOFF=.5)
        with patch.object(planner, "sample_point", side_effect=[(5, 1), (9, 0)]):
            nodes, path = planner.run()
        self.assertIsNone(nodes[1].parent)
        self.assertNotIn(nodes[1], path)
        self.assertIs(path[0], planner.start)
        self.assertIs(path[-1], planner.goal)
