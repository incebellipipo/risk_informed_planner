"""Angle-aware RRT* exploration with sampled edge risk and tree telemetry."""
from __future__ import annotations

import dataclasses
import enum
import math
import operator
import time

import cartoblobpy.graph
import numpy as np

from tqdm.auto import tqdm

from .config import Config

from .risk.dynamic_risk import DynamicRisk
from .utils import (  # explicit import keeps namespace clear for type checking
    Node,
    Point,
)



def distance(node1: Node, node2: Node) -> float:
    """Return Euclidean distance between two nodes in world-coordinate units."""
    return np.hypot(node1.x - node2.x, node1.y - node2.y)

def squared_distance(node1: Node, node2: Node) -> float:
    """Return squared Euclidean distance between two nodes."""
    return (node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2

def turn_angle(prev: Node, curr: Node, nxt: Node) -> float:
    """Absolute heading change at ``curr`` when travelling prev -> curr -> nxt.

    Returns a value in ``[0, pi]``. Degenerate (zero-length) segments have no
    defined heading and are reported as a straight-through transit.
    """
    in_x, in_y = curr.x - prev.x, curr.y - prev.y
    out_x, out_y = nxt.x - curr.x, nxt.y - curr.y

    if np.hypot(in_x, in_y) < 1e-9 or np.hypot(out_x, out_y) < 1e-9:
        return 0.0

    delta = np.arctan2(out_y, out_x) - np.arctan2(in_y, in_x)
    return abs((delta + np.pi) % (2 * np.pi) - np.pi)


@dataclasses.dataclass
class EdgeCost:
    """Store the result of evaluating a candidate edge.
    
    Attributes
    ----------
    cost : float
        Accumulated cost from the root through this candidate connection.
    heuristic : float
        Incremental cost of this edge alone.
    risk : float
        Aggregate sampled edge risk according to the configured aggregation.
    peak_risk : float
        Largest sampled risk, used separately for cutoff checks.
    distance_cost : float
        Travelled distance from the root to the endpoint.
    turn_angle : float
        Heading change at the proposed parent, in radians.
    """

    cost: float           # from_node.cost + heuristic (cost-to-come of to_node)
    heuristic: float      # cost of this edge alone
    risk: float           # edge risk, aggregated over the samples along the edge
    peak_risk: float      # worst risk seen anywhere on the edge; gates RISK_CUTOFF
    distance_cost: float  # arc length from the root
    turn_angle: float     # heading change at from_node, radians

    def is_feasible(self) -> bool:
        """Return whether accumulated cost is finite.
        
        This does not apply a peak-risk cutoff; the planner checks that separately.
        """
        return math.isfinite(self.cost)


_pbar = None
_last_count = 0

def progress_bar(nodes, path, obj: RiskAwareAngleCostRRTStarSearch):
    """Display tree growth and path size with tqdm.
    
    Receives ``(nodes, path, planner)`` as a planner callback. Progress state is
    module-global; use separate processes for independent concurrent bars.
    """
    global _pbar, _last_count

    current_count = len(nodes)

    # 1. Lazy initialization on the first call
    if _pbar is None:
        # We set total, but tqdm allows going over it gracefully
        _pbar = tqdm(total=obj.cfg.MAX_NODES, desc="Progress", unit="node")
        _last_count = 0

    # 2. Calculate delta (handles normal growth and overshooting)
    delta = current_count - _last_count

    if delta > 0:
        _pbar.update(delta)
        _last_count = current_count

    # 3. Dynamic stats display on the right-hand side
    _pbar.set_postfix({
        'Path': len(path),
        # 'Nodes': f"{current_count}/{obj.cfg.MAX_NODES}"
    })

    # 4. Safe termination condition
    # Caps off if we hit/exceed max nodes AND a valid path is found
    if current_count >= obj.cfg.MAX_NODES and len(path) > 0:
        _pbar.close()
        _pbar = None  # Reset global reference so the next run starts fresh

class RiskAwareAngleCostRRTStarSearch:
    """Explore a graph using sampled edge risk and an additive turn penalty.
    
    Parameters
    ----------
    start, goal : Node
        Root and goal nodes in graph world coordinates.
    cfg : Config
        Growth, cost, sampling, turn, and risk-cutoff settings.
    risk_model : DynamicRisk, optional
        Risk model or compatible object. Supply one before running the planner.
    graph : object, optional
        Loaded cartoblobpy graph used for free-space sampling and collision checks.
    min_depth : float, optional
        Minimum sampling depth for depth-based sampling; defaults to 15 metres.
        Historical DEPTH edge checks use a fixed 10 metre threshold and reject
        depths strictly below it, independently of this sampling limit.
    callback_functions : list, optional
        Callbacks receiving ``(nodes, path, planner)`` after normal iterations.
        Some rejected attempts continue before callbacks run. Pass an empty list
        to suppress the default progress callback.
    record_bbn_snapshots : bool, optional
        Record endpoint posterior values and state names on evaluated nodes for
        subsequent hazard figures. Defaults to False.
    free_check_method : FreeCheckMethod, optional
        OCCUPANCY (default), DEPTH, or DEPTH_SAMPLING_OCCUPANCY_CHECKS.
        The mixed policy reproduces the original studies' foreign-enum behavior.
    
    Attributes
    ----------
    nodes : list
        Root, connected tree nodes, and disconnected retained risk observations.
    path : list or None
        Current route, empty if no acceptable goal connection is retained.
    risk_evaluations : int
        Number of risk-model calls while sampling edges.
    no_of_cost_evaluations : int
        Candidate edge-cost evaluations, including turn rejections.
    risk_compute_times : list
        Per-call durations in seconds, including evidence construction, native
        inference, and loss computation.
    
    Notes
    -----
    The root has no incoming heading and turns for free. Edge risk can be sampled
    throughout the segment, while snapshots describe only endpoint evidence.
    Finite spacing may miss hazards between samples.
    
    For length L, normalized aggregate risk r, turn angle delta, and reference
    length Lref, the edge cost is ``ALPHA*L + (1-ALPHA)*BETA*r*L`` plus
    ``GAMMA*(abs(delta)/pi)*Lref``. Lref defaults to STEP_SIZE. Accumulated cost adds
    the parent's cost. Turns over the configured limit are rejected before BBN
    inference; cutoff handling uses sampled peak risk rather than mean risk.
    """

    class FreeCheckMethod(enum.Enum):
        """Choose occupancy-based or minimum-depth-based geometric collision checks."""
        OCCUPANCY = 1
        DEPTH = 2
        DEPTH_SAMPLING_OCCUPANCY_CHECKS = 3
        """Preserve the original studies' depth sampling and occupancy edge checks."""

    class RiskAggregation(enum.Enum):
        """Aggregate edge samples by MEAN, MAX, or ENDPOINT; always retain peak risk."""
        MEAN = "mean"          # path integral of risk over the edge
        MAX = "max"            # the edge is as risky as its worst point
        ENDPOINT = "endpoint"  # legacy: sample the far end only

    def __init__(
        self,
        start: Node,
        goal: Node,
        cfg: Config,
        risk_model: DynamicRisk = None,
        graph: cartoblobpy.graph.Graph = None,
        min_depth: float = 15.0,
        callback_functions: list[callable] = [progress_bar],
        record_bbn_snapshots: bool = False,
        free_check_method: FreeCheckMethod = FreeCheckMethod.OCCUPANCY,
        # risk_max: float = 1.0
    ):
        self.start = start
        self.goal = goal
        self.cfg = cfg
        self.risk_model = risk_model
        self.nodes: list[Node] = [start]
        self.graph = graph
        self.path = None
        self.callback_functions = callback_functions or []
        self.record_bbn_snapshots = record_bbn_snapshots
        self.__risk_max = self.risk_model.cost_model.compute_worst_case_cost(self.risk_model.bbn_network)

        self.__free_check_method = free_check_method

        self.no_of_cost_evaluations = 0
        self.no_of_risk_evaluations = 0

        if self.__free_check_method == self.FreeCheckMethod.OCCUPANCY:
            self.__free_grid = self.graph.free_space_mask(threshold=0.5, op=operator.le)
        else: # if self.__free_check_method == self.FreeCheckMethod.DEPTH:
            self.__free_grid = self.graph.free_space_mask(layer_name="depth", threshold=min_depth, op=operator.gt)

        self.__free_grid_indices = np.argwhere(self.__free_grid)
        self.__squared_neighbor_radius = self.cfg.NEIGHBOR_RADIUS ** 2
        self.__squared_goal_radius = self.cfg.GOAL_RADIUS ** 2

        self.__gamma = getattr(self.cfg, "GAMMA", 0.0) or 0.0
        self.__max_turn = getattr(self.cfg, "MAX_TURN_ANGLE", None)

        # A full 180 deg turn is priced like this much extra path length, so the
        # penalty stays in the same units as the distance term and does not
        # depend on how long the individual edge happens to be.
        self.__turn_reference_length = (
            getattr(self.cfg, "TURN_REFERENCE_LENGTH", None) or self.cfg.STEP_SIZE
        )

        self.__risk_sample_spacing = getattr(self.cfg, "RISK_SAMPLE_SPACING", None)
        self.__max_risk_samples = getattr(self.cfg, "MAX_RISK_SAMPLES_PER_EDGE", 32) or 32
        self.__risk_aggregation = self.RiskAggregation(
            getattr(self.cfg, "RISK_AGGREGATION", "mean") or "mean"
        )

        # Cheap telemetry: lets you see what the sampling actually cost you.
        self.risk_evaluations = 0

        # Wall-clock time [s] spent inside risk_model.compute_risk() per call,
        # one entry per evaluation. Covers discretization (evidence-setting),
        # BBN inference and cost-model computation, since all three happen
        # inside that one call.
        self.risk_compute_times: list[float] = []

    # def compute_risk(self, node: Node) -> float:
    #     return self.risk_model.compute_risk(node)

    def compute_turn_cost(self, from_node: Node, to_node: Node) -> tuple[float, float]:
        """Cost of the heading change incurred at ``from_node``.

        The turn is measured between the incoming edge (``from_node.parent`` ->
        ``from_node``) and the outgoing edge (``from_node`` -> ``to_node``). A
        root node has no incoming heading and therefore turns for free.

        Returns ``(angle, cost)`` where ``cost`` is ``inf`` when the turn exceeds
        ``MAX_TURN_ANGLE``, which makes every caller reject the edge.
        """
        if from_node.parent is None:
            return 0.0, 0.0

        angle = turn_angle(from_node.parent, from_node, to_node)

        if self.__max_turn is not None and angle > self.__max_turn:
            return angle, float("inf")

        return angle, self.__gamma * (angle / np.pi) * self.__turn_reference_length

    def num_risk_samples(self, edge_len: float) -> int:
        """How many risk samples this edge gets, given the configured spacing."""
        spacing = self.__risk_sample_spacing

        if not spacing or spacing <= 0.0 or edge_len < 1e-6:
            return 1

        n = int(math.ceil(edge_len / spacing))
        return int(min(max(n, 1), self.__max_risk_samples))

    def compute_edge_risk(self, from_node: Node, to_node: Node) -> tuple[float, float]:
        """Risk of traversing ``from_node -> to_node``, sampled along the edge.

        Samples sit at the far end of each of ``n`` equal sub-segments, so the
        edge's own endpoint is always one of them and ``n == 1`` reduces exactly
        to the legacy endpoint-only behaviour.

        Each sample carries the heading of the edge (so grounding-course and CPA
        evidence see the real course) and the arc length from the root at that
        point (so the dynamic obstacles are placed at the time the vessel is
        actually there, rather than all at t=0).

        Returns ``(aggregated_risk, peak_risk)``.
        """
        edge_len = distance(from_node, to_node)
        n = self.num_risk_samples(edge_len)

        risks = []
        prev_x, prev_y = from_node.x, from_node.y

        for i in range(1, n + 1):
            t = i / n
            sample = Node(
                from_node.x + (to_node.x - from_node.x) * t,
                from_node.y + (to_node.y - from_node.y) * t,
                -1,
            )

            # The parent only ever supplies a heading. Taking it from the previous
            # sample keeps that heading equal to the edge's own direction.
            sample.parent = Node(prev_x, prev_y, -1)

            # Arc length travelled from the root to this sample. DynamicRisk turns
            # this into a time-of-arrival to propagate the dynamic obstacles.
            sample.distance_cost = from_node.distance_cost + edge_len * t

            t0 = time.perf_counter()
            sample_risk = self.risk_model.compute_risk(node=sample)
            self.risk_compute_times.append(time.perf_counter() - t0)

            risks.append(sample_risk)
            self.risk_evaluations += 1

            prev_x, prev_y = sample.x, sample.y

            if self.record_bbn_snapshots and i == n:
                to_node.bbn_snapshot = self.risk_model.bbn_network.snapshot()

        peak_risk = max(risks)

        if self.__risk_aggregation is self.RiskAggregation.MAX:
            return peak_risk, peak_risk
        if self.__risk_aggregation is self.RiskAggregation.ENDPOINT:
            return risks[-1], peak_risk

        # MEAN: the samples are uniformly spaced, so the mean is the rectangle-rule
        # estimate of (1/L) * integral of risk ds. Multiplying it by the edge length
        # downstream therefore recovers the true risk exposure integral.
        return sum(risks) / len(risks), peak_risk

    def compute_cost(self, from_node: Node, to_node: Node) -> EdgeCost:

        """Evaluate an incoming edge and return its EdgeCost.
        
        Include the parent's accumulated cost, sampled aggregate and peak risk,
        travelled distance, and turn angle. An excessive turn returns an infeasible
        edge without risk-model calls.
        
        Raises
        ------
        ValueError
            If either endpoint is missing.
        """
        if not from_node or not to_node:
            raise ValueError("Both from_node and to_node must be provided for cost computation.")

        self.no_of_cost_evaluations += 1

        edge_len = distance(from_node, to_node)

        angle, turn_cost = self.compute_turn_cost(from_node, to_node)

        # An over-limit turn is rejected outright; no point paying for the BBN
        # inferences on an edge nobody can use.
        if not math.isfinite(turn_cost):
            return EdgeCost(
                cost=float("inf"),
                heuristic=float("inf"),
                risk=float("inf"),
                peak_risk=float("inf"),
                distance_cost=from_node.distance_cost + edge_len,
                turn_angle=angle,
            )

        edge_risk, peak_risk = self.compute_edge_risk(from_node, to_node)

        r = edge_risk / self.__risk_max
        risk_exposure = r * edge_len

        # The turn penalty is added on top of the distance/risk combination rather
        # than blended into it, so ALPHA and BETA keep meaning exactly what they
        # mean in the base planner.
        heuristic = (
            + self.cfg.ALPHA * edge_len
            + (1 - self.cfg.ALPHA) * self.cfg.BETA * risk_exposure
            + turn_cost
        )

        return EdgeCost(
            cost=from_node.cost + heuristic,
            heuristic=heuristic,
            risk=edge_risk,
            peak_risk=peak_risk,
            distance_cost=from_node.distance_cost + edge_len,
            turn_angle=angle,
        )

    @staticmethod
    def _apply(node: Node, parent: Node, edge: EdgeCost) -> None:
        node.parent = parent
        node.cost = edge.cost
        node.heuristic = edge.heuristic
        node.risk = edge.risk
        node.peak_risk = edge.peak_risk
        node.distance_cost = edge.distance_cost
        node.turn_angle = edge.turn_angle

    def get_neighbors(self, new_node: Node) -> list[Node]:
        """Return nodes within NEIGHBOR_RADIUS, excluding coincident positions."""
        nodes = []
        for node in self.nodes:
            sqrted_dist = squared_distance(node, new_node)
            if sqrted_dist <= self.__squared_neighbor_radius and sqrted_dist > 1e-3:
                nodes.append(node)
        return nodes


    def get_closest_neighbors(self, new_node: Node, max_neighbors: int) -> list[Node]:
        """Return up to max_neighbors nearby nodes sorted by increasing distance."""
        neighbors = self.get_neighbors(new_node)
        neighbors.sort(key=lambda n: squared_distance(n, new_node))
        return neighbors[:max_neighbors]

    def extract_path(self) -> list[Node]:
        """Return the reversed chain of goal parents.
        
        The run loop treats this as a route only when the goal has a parent.
        """
        path = []
        node = self.goal
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1]

    def steer(self, from_node: Node, to_point: Point) -> Point | None:
        """Return a point at most STEP_SIZE toward the target without overshooting.
        
        A coincident target returns None.
        """
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        dist = np.hypot(dx, dy)
        if dist == 0:
            return None
        scale = min(self.cfg.STEP_SIZE / dist, 1.0)
        return from_node.x + dx * scale, from_node.y + dy * scale

    def sample_point(self) -> Point:

        # Sample free points only using the graph

        # Create a mask of free cells
        # free = self.graph.grid < 0.5
        # indices = np.argwhere(free)
        """Sample a free grid cell, perturb its world coordinates, and clip to the map.
        
        Raises
        ------
        RuntimeError
            If the selected collision policy leaves no free cells.
        """
        if len(self.__free_grid_indices) == 0:
            raise RuntimeError("No free space available in the graph for sampling.")

        # Choose one randomly
        chosen_idx = self.__free_grid_indices[np.random.choice(len(self.__free_grid_indices))]

        # Get the world coordinates of the chosen cell
        x, y = self.graph.grid_to_world(chosen_idx)

        # wiggle a bit to avoid grid artifacts
        x += np.random.uniform(-self.graph.resolution, self.graph.resolution)
        y += np.random.uniform(-self.graph.resolution, self.graph.resolution)

        x = np.clip(x, 0, self.graph.width)
        y = np.clip(y, 0, self.graph.height)

        return (x, y)

    def check_collision(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Return whether a segment is blocked under the selected occupancy or depth policy."""
        if self.__free_check_method == self.FreeCheckMethod.DEPTH:
            return not self.graph.is_free_path([x1, y1], [x2, y2], layer_name="depth", threshold=10, op=operator.lt)
        else: # if self.__free_check_method == self.FreeCheckMethod.OCCUPANCY:
            # anything else
            return not self.graph.is_free_path([x1, y1], [x2, y2], threshold=0.5, op=operator.gt)

    def rewire(self, new_node: Node):
        """Consider cheaper incoming edges from new_node to neighboring tree nodes.
        
        Rewiring respects geometric collisions, turn limits, and sampled peak-risk
        cutoffs. MAX_NEIGHBORS_RISK_RENEWAL can limit the considered neighborhood.
        """
        neighbors = []
        if self.cfg.MAX_NEIGHBORS_RISK_RENEWAL not in (None, float("inf")):
            neighbors = self.get_closest_neighbors(
                new_node=new_node,
                max_neighbors=(self.cfg.MAX_NEIGHBORS_RISK_RENEWAL or len(self.nodes))
            )
        else:
            neighbors = self.get_neighbors(new_node)

        for neighbor in neighbors:
            if neighbor is new_node.parent:
                continue
            if self.check_collision(new_node.x, new_node.y, neighbor.x, neighbor.y):
                continue

            if neighbor.peak_risk > self.cfg.RISK_CUTOFF:
                continue

            edge = self.compute_cost(from_node=new_node, to_node=neighbor)

            # Gate on the worst point of the edge, not its aggregate: an edge whose
            # mean risk is acceptable can still pass straight through a red zone.
            if edge.peak_risk > self.cfg.RISK_CUTOFF:
                continue

            if edge.cost < neighbor.cost:
                self._apply(neighbor, new_node, edge)

    def choose_parent(self, new_node: Node) -> tuple[Node | None, EdgeCost | None]:
        """Return the cheapest finite collision-free candidate and its EdgeCost.
        
        Return ``(None, None)`` when no candidate is feasible. Disconnected retained
        observations are not eligible parents. The run loop handles peak-risk cutoff
        rejection after choosing a candidate.
        """
        best_parent = None
        best_edge = None

        for neighbor in self.get_neighbors(new_node):

            if neighbor.parent is None and neighbor.idx != self.nodes[0].idx:
                # a neighbor without a parent is only a data point.
                continue

            if self.check_collision(neighbor.x, neighbor.y, new_node.x, new_node.y):
                continue

            edge = self.compute_cost(to_node=new_node, from_node=neighbor)

            if not edge.is_feasible():
                continue

            if best_edge is None or edge.cost < best_edge.cost:
                best_parent = neighbor
                best_edge = edge

        if best_parent is None:
            return None, None

        return best_parent, best_edge

    def run(self) -> tuple[list[Node], list[Node]]:
        """Grow the tree and return the explored nodes and ordered route.
        
        Calling the planner object invokes this method. The node list includes
        retained high-risk observations without parents. The route is empty if no
        acceptable goal connection is retained.
        
        MAX_NODES and MAX_ITER are stopping targets after a collision-free geometric
        approach to the goal is found, not hard limits. A geometrically reachable goal
        may have every connection rejected by turn or risk limits. A geometrically
        unreachable goal may keep the search running beyond those targets.
        """
        goal_reached = False
        it = 0

        while True:

            it += 1
            if getattr(self.cfg, "MAX_ITER", None):
                if it >= self.cfg.MAX_ITER:
                    if not goal_reached:
                        pass
                    else:
                        break

            if getattr(self.cfg, "MAX_NODES", None):
                if len(self.nodes) >= self.cfg.MAX_NODES:
                    if not goal_reached:
                        pass
                    else:
                        break

            point = self.sample_point()

            nearest = min(self.nodes, key=lambda n: ((n.x - point[0])**2 + (n.y - point[1])**2))

            new_pos = self.steer(nearest, point)

            if not new_pos:
                continue

            new_node = Node(new_pos[0], new_pos[1], len(self.nodes))

            if self.check_collision(nearest.x, nearest.y, new_node.x, new_node.y):
                continue

            new_parent, new_edge = self.choose_parent(new_node)

            if new_parent is None:
                continue

            self._apply(new_node, new_parent, new_edge)
            self.nodes.append(new_node)

            if new_edge.peak_risk > self.cfg.RISK_CUTOFF:
                # orphan the node if risk value is above cutoff. Keep it as a datapoint.
                new_node.parent = None
                continue

            self.rewire(new_node)

            if squared_distance(new_node, self.goal) < self.__squared_goal_radius:
                if self.check_collision(new_node.x, new_node.y, self.goal.x, self.goal.y):
                    continue

                # "Reachable at all" -- deliberately not gated on the risk checks
                # below, otherwise a goal that only admits risky connections would
                # never set the flag and run() would never hit its break.
                goal_reached = True

                goal_edge = self.compute_cost(to_node=self.goal, from_node=new_node)

                if goal_edge.is_feasible() and goal_edge.peak_risk <= self.cfg.RISK_CUTOFF:
                    if getattr(self.goal, "parent", None) is None:
                        self._apply(self.goal, new_node, goal_edge)
                        self.nodes.append(self.goal)
                    elif goal_edge.cost < self.goal.cost:
                        self._apply(self.goal, new_node, goal_edge)

            self.path = self.extract_path() if getattr(self.goal, "parent", None) else []

            for cb in self.callback_functions:
                cb(self.nodes, self.path, self)


        return self.nodes, self.path

    def __call__(self, *args, **kwds):
        return self.run()

    def rewire_all(self) -> tuple[list[Node], list[Node]]:

        # rewire all nodes randomly to avoid bias from the order of insertion
        """Reconsider tree connections in shuffled order and return the nodes and route."""
        np.random.shuffle(self.nodes)

        for node in self.nodes:
            self.rewire(node)

        self.path = self.extract_path() if getattr(self.goal, "parent", None) else []
        return self.nodes, self.path






