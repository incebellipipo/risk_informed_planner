"""Small deterministic fixtures; no native network is constructed here."""
import sys
import operator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from risk_informed_planner.config import Config
from risk_informed_planner.planner import RiskAwareAngleCostRRTStarSearch
from risk_informed_planner.utils import Node


def severity_network(grounding, collision):
    return SimpleNamespace(get_grounding_severity=lambda: grounding,
                           get_collision_severity=lambda: collision)


def open_graph(depth=20.0):
    graph = Mock(width=100.0, height=100.0, resolution=1.0)
    graph.free_space_mask.return_value = np.ones((10, 10), dtype=bool)
    def is_free_path(a, b, **kwargs):
        if kwargs.get("layer_name") == "depth":
            return not kwargs.get("op", operator.gt)(depth, kwargs["threshold"])
        return True
    graph.is_free_path.side_effect = is_free_path
    graph.grid_to_world.side_effect = lambda rc: np.asarray(rc, dtype=float)
    graph.distance_to_closest_obstacle.return_value = 50.0
    graph.value_at.side_effect = lambda xy, layer_name: depth if layer_name == "depth" else 0.0
    return graph


def make_planner(evaluator=None, graph=None, record=False, method=None, min_depth=6, **options):
    settings = dict(STEP_SIZE=10, NEIGHBOR_RADIUS=20, GOAL_RADIUS=2, MAX_NODES=3)
    settings.update(options)
    risk = Mock()
    risk.cost_model.compute_worst_case_cost.return_value = 1.0
    risk.compute_risk.side_effect = evaluator or (lambda node: 0.0)
    risk.bbn_network.snapshot.return_value = {"Collision": {"values": [0.1, 0.9]}}
    return RiskAwareAngleCostRRTStarSearch(
        start=Node(0, 0, 0), goal=Node(10, 0, -1), cfg=Config(**settings),
        risk_model=risk, graph=graph if graph is not None else open_graph(),
        callback_functions=[], record_bbn_snapshots=record, min_depth=min_depth,
        free_check_method=method or RiskAwareAngleCostRRTStarSearch.FreeCheckMethod.OCCUPANCY)
