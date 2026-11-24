"""Run from the repository root with a valid PySMILE license."""
import time
from pathlib import Path

import numpy as np
from cartoblobpy.graph import Graph

from risk_informed_planner.config import Config
from risk_informed_planner.experiment import PlannerRun
from risk_informed_planner.obstacles import LinearDynamicObstacle
from risk_informed_planner.risk.cost_model import NormalizedExpectedLoss
from risk_informed_planner.risk.dynamic_risk import DynamicRisk
from risk_informed_planner.planner import RiskAwareAngleCostRRTStarSearch
from risk_informed_planner.utils import Node

map_file = Path("notebooks/maps/kristiansund_bridge/map.yaml").resolve()
graph = Graph()
graph.load_from_yaml(str(map_file))
graph.start, graph.goal = (200.0, 800.0), (1500.0, 600.0)
obstacles = [LinearDynamicObstacle.create((500.0, 200.0), (300.0, 800.0), 2.0)]
cfg = Config(
    X_LIMITS=(0, graph.width), Y_LIMITS=(0, graph.height),
    STEP_SIZE=50, GOAL_RADIUS=75, NEIGHBOR_RADIUS=150, MAX_NODES=1500,
    ALPHA=0.1, BETA=100.0, GAMMA=0.6, MAX_TURN_ANGLE=np.pi / 4,
    RISK_CUTOFF=0.10, RISK_SAMPLE_SPACING=15.0, RISK_AGGREGATION="max",
)
minimum_depth = 6.0
risk = DynamicRisk(
    graph=graph, obstacles=obstacles, desired_speed=3.0,
    max_distance_from_shore=150.0, lookahead_distance=150.0,
    max_dcpa_for_collision_risk=100.0, max_tcpa_for_collision_risk=120.0,
    max_relative_velocity_for_collision_risk=5.0,
    max_depth_for_grounding_risk=200.0, grounding_depth_hard_limit=minimum_depth,
    cost_model=NormalizedExpectedLoss(),
)
planner = RiskAwareAngleCostRRTStarSearch(
    start=Node(*graph.start, 0), goal=Node(*graph.goal, -1),
    cfg=cfg, graph=graph, risk_model=risk, min_depth=minimum_depth,
    record_bbn_snapshots=True,
    free_check_method=RiskAwareAngleCostRRTStarSearch.FreeCheckMethod.DEPTH_SAMPLING_OCCUPANCY_CHECKS,
)
started = time.perf_counter()
nodes, path = planner()
elapsed = time.perf_counter() - started
run = PlannerRun(nodes, path, cfg, graph, obstacles=obstacles,
                 t_computation=elapsed, speed=risk.desired_speed,
                 map_file=str(map_file), metadata={"scenario": "kristiansund_bridge"})
run.save("notebooks/experiments/results/study_03.pkl")
print(f"{len(nodes)} nodes, {len(path)} route nodes, {elapsed:.1f} seconds")
