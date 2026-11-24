"""Notebook syntax, real maps, and native BBN integration; no figure tests."""
import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from tests.helpers import ROOT, Node, make_planner
from cartoblobpy.graph import Graph
from risk_informed_planner.config import Config
from risk_informed_planner.experiment import PlannerRun


class StudyWorkflowTests(unittest.TestCase):
    def test_retained_notebook_syntax(self):
        notebooks = [p for folder in ("experiments", "statistics")
                     for p in (ROOT / "notebooks" / folder).glob("*.ipynb")]
        self.assertTrue(notebooks)
        for notebook in notebooks:
            with self.subTest(notebook=notebook.name):
                for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]:
                    if cell["cell_type"] == "code":
                        ast.parse("".join(cell["source"]), filename=str(notebook))

    def test_all_preserved_maps_load(self):
        maps = list((ROOT / "notebooks/maps").glob("*/map.yaml"))
        self.assertEqual(len(maps), 6)
        for map_file in maps:
            with self.subTest(map=map_file.parent.name):
                graph = Graph()
                graph.load_from_yaml(str(map_file))
                self.assertGreater(graph.width, 0)
                self.assertGreater(graph.height, 0)
                self.assertTrue(graph.free_space_mask(threshold=0.5).any())

    def test_saved_map_reference_reloads_from_another_working_directory(self):
        map_file = ROOT / "notebooks/maps/empty/map.yaml"
        graph = Graph()
        graph.load_from_yaml(str(map_file))
        graph.start, graph.goal = (100, 100), (300, 300)
        run = PlannerRun([Node(100, 100, 0)], [], Config(), graph, map_file=str(map_file))
        with tempfile.TemporaryDirectory() as folder:
            saved = run.save(Path(folder) / "run.pkl")
            previous = Path.cwd()
            try:
                os.chdir(folder)
                restored = PlannerRun.load(saved)
            finally:
                os.chdir(previous)
        self.assertEqual(restored.graph.start, graph.start)
        self.assertEqual(restored.graph.goal, graph.goal)
        self.assertEqual(restored.graph.width, graph.width)

    def test_depth_policies_with_a_real_graph_and_a_shallow_patch(self):
        from unittest.mock import Mock
        from risk_informed_planner.bbn import bbn_autonomous_ranked_generated as bn
        from risk_informed_planner.risk.dynamic_risk import DynamicRisk
        from risk_informed_planner.planner import RiskAwareAngleCostRRTStarSearch

        graph = Graph()
        graph.load_from_yaml(str(ROOT / "notebooks/maps/empty/map.yaml"))
        # Change the in-memory layer only; the map files remain untouched.
        graph.layers["depth"][:] = 7
        planner = make_planner(graph=graph, min_depth=6,
                              method=RiskAwareAngleCostRRTStarSearch.FreeCheckMethod.DEPTH)
        risk = DynamicRisk(graph=graph, the_network=Mock(), lookahead_distance=30,
                           grounding_depth_hard_limit=6)
        node = Node(100, 100, 1)
        node.parent = Node(90, 100, 0)
        # Historical depth edge checks reject 7 m even though sampling allows it.
        self.assertTrue(planner.check_collision(100, 100, 130, 100))
        self.assertEqual(risk._evidence_grounding_course(node), bn.Vessel_on_grounding_course.YES)
        row, col = graph.world_to_grid([110, 100]).astype(int)
        graph.layers["depth"][row, col] = 3
        self.assertTrue(planner.check_collision(100, 100, 130, 100))
        self.assertEqual(risk._evidence_grounding_course(node), bn.Vessel_on_grounding_course.YES)
        graph.layers["depth"][:] = 3
        self.assertEqual(risk._evidence_grounding_course(node), bn.Vessel_on_grounding_course.NO)

    def test_native_bbn_inference_and_evidence_reset(self):
        # pySMILE may exit the interpreter on license errors. Keep that failure
        # inside a child process so the rest of the suite can still finish.
        code = textwrap.dedent("""
            import math
            from cartoblobpy.graph import Graph
            from risk_informed_planner.risk.dynamic_risk import DynamicRisk
            from risk_informed_planner.utils import Node
            g = Graph()
            g.load_from_yaml('notebooks/maps/kristiansund_bridge/map.yaml')
            model = DynamicRisk(graph=g, desired_speed=3, max_distance_from_shore=150)
            def node(x, y):
                n = Node(x, y, 1)
                n.parent = Node(x-50, y, 0)
                n.distance_cost = 50
                return n
            first = model.compute_risk(node(200, 800))
            other = model.compute_risk(node(400, 490))
            repeated = model.compute_risk(node(200, 800))
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in [first, other, repeated])
            assert math.isclose(first, repeated, rel_tol=1e-12, abs_tol=1e-12)
            snapshot = model.bbn_network.snapshot()
            for name in ['Collision', 'Grounding']:
                assert math.isclose(sum(snapshot[name]['values']), 1.0, abs_tol=1e-10)
            # Reference values verified against the original package, using its
            # normalized expected-loss model and historical grounding predicate.
            model = DynamicRisk(graph=g, desired_speed=3, max_distance_from_shore=150,
                                lookahead_distance=150, grounding_depth_hard_limit=6)
            references = [
                (200, 300, 0.031665623742350046, 0.1543401024496984),
                (400, 490, 0.05018773855474079, 0.21756459863761407),
                (1000, 600, 0.010170428133242426, 0.0011785300000000001),
            ]
            for x, y, expected_loss, expected_grounding in references:
                loss = model.compute_risk(node(x, y))
                assert math.isclose(loss, expected_loss, rel_tol=1e-12, abs_tol=1e-14), (x, y, loss)
                grounding = model.bbn_network.get_grounding()[0]
                assert math.isclose(grounding, expected_grounding, rel_tol=1e-12, abs_tol=1e-14)
            print('native inference passed')
        """)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0,
                         "Native BBN check failed; check the PySMILE installation and license.\n"
                         + result.stdout + result.stderr)
        self.assertIn("native inference passed", result.stdout)
