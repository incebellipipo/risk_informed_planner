"""Persistence preserves graph references, node links, and legacy import paths."""
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests.helpers import Node
from risk_informed_planner.config import Config
from risk_informed_planner.experiment import PlannerRun
from risk_informed_planner.obstacles import LinearDynamicObstacle
from risk_informed_planner.planner import EdgeCost, RiskAwareAngleCostRRTStarSearch


class PlannerRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        start, goal = Node(1, 2, 0), Node(3, 4, 1)
        goal.parent, goal.cost = start, 7
        goal.bbn_snapshot = {"Collision": {"values": [.1, .9]}}
        self.run = PlannerRun([start, goal], [start, goal], Config(), None,
                              obstacles=[LinearDynamicObstacle(1, 2, 0, 3)],
                              t_computation=1.5, speed=3, metadata={"scenario": "test"})

    def test_save_creates_parent_directories_and_preserves_node_identity(self):
        path = self.run.save(self.folder / "nested/run.pkl")
        loaded = PlannerRun.load(path)
        self.assertIs(loaded.path[-1], loaded.nodes[-1])
        self.assertIs(loaded.path[-1].parent, loaded.path[0])
        self.assertEqual(loaded.path[-1].cost, 7)
        self.assertEqual(loaded.path[-1].bbn_snapshot, self.run.path[-1].bbn_snapshot)
        self.assertEqual(loaded.metadata, {"scenario": "test"})
        self.assertEqual(loaded.obstacles[0].velocity, 3)

    def test_campaign_round_trip_preserves_order_and_shared_objects(self):
        second = PlannerRun(self.run.nodes, self.run.path, Config(), None, metadata={"step": 2})
        path = PlannerRun.save_many([self.run, second], self.folder / "campaign.pkl")
        runs = PlannerRun.load_many(path)
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[1].metadata["step"], 2)
        self.assertIs(runs[0].nodes[0], runs[1].nodes[0])

    def test_empty_campaign_round_trip(self):
        self.assertEqual(PlannerRun.load_many(PlannerRun.save_many([], self.folder / "empty.pkl")), [])

    def test_single_run_loader_rejects_wrong_payloads(self):
        for payload in [{}, [], None, 42]:
            with self.subTest(payload=payload):
                path = self.folder / "invalid.pkl"
                path.write_bytes(pickle.dumps(payload))
                with self.assertRaisesRegex(TypeError, "not a PlannerRun"):
                    PlannerRun.load(path)

    def test_campaign_loader_rejects_wrong_payloads_and_mixed_lists(self):
        for payload in [self.run, {}, [self.run, 42]]:
            with self.subTest(payload=type(payload).__name__):
                path = self.folder / "invalid.pkl"
                path.write_bytes(pickle.dumps(payload))
                with self.assertRaisesRegex(TypeError, "list of PlannerRun"):
                    PlannerRun.load_many(path)

    def test_legacy_single_run_and_campaign_load_without_old_package_alias(self):
        for payload, loader in [(self.run, PlannerRun.load), ([self.run], PlannerRun.load_many)]:
            with self.subTest(loader=loader.__name__):
                # Protocol 0 stores textual module names; preserve all other data.
                legacy = pickle.dumps(payload, protocol=0).replace(
                    b"risk_informed_planner", b"risk_based_planner")
                path = self.folder / "legacy.pkl"
                path.write_bytes(legacy)
                restored = loader(path)
                restored = restored[0] if isinstance(restored, list) else restored
                self.assertEqual(restored.path[-1].pos(), (3, 4))
                self.assertIs(restored.path[-1].parent, restored.nodes[0])
                self.assertEqual(restored.obstacles[0].velocity, 3)

    def test_old_planner_module_references_load_after_module_rename(self):
        self.run.metadata = {
            "edge": EdgeCost(1, 1, .2, .3, 5, .1),
            "collision_mode": RiskAwareAngleCostRRTStarSearch.FreeCheckMethod.DEPTH,
        }
        for package in ["risk_informed_planner", "risk_based_planner"]:
            with self.subTest(package=package):
                payload = pickle.dumps(self.run, protocol=0).replace(
                    b"risk_informed_planner.planner",
                    (package + ".risk_exploration_angle_cost").encode())
                path = self.folder / "legacy_planner.pkl"
                path.write_bytes(payload)
                restored = PlannerRun.load(path)
                self.assertIsInstance(restored.metadata["edge"], EdgeCost)
                self.assertEqual(restored.metadata["edge"].peak_risk, .3)
                self.assertIs(restored.metadata["collision_mode"],
                              RiskAwareAngleCostRRTStarSearch.FreeCheckMethod.DEPTH)

    def test_graph_is_not_serialized_when_map_reference_is_available(self):
        graph = SimpleNamespace(start=(1, 2), goal=(3, 4), unpicklable=lambda: None)
        self.run.graph = graph
        self.run.map_file = str(self.folder / "map.yaml")
        # A graph containing a lambda cannot be pickled; saving succeeds by omitting it.
        self.run.save(self.folder / "run.pkl")
        self.assertIs(self.run.graph, graph)
        state = self.run.__getstate__()
        self.assertIsNone(state["graph"])
        self.assertEqual(state["_graph_start_goal"], ((1, 2), (3, 4)))

    def test_corrupt_pickle_is_rejected(self):
        path = self.folder / "corrupt.pkl"
        path.write_bytes(b"not a pickle")
        with self.assertRaises(pickle.UnpicklingError):
            PlannerRun.load(path)
