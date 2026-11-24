"""Environmental evidence and inference for the ranked autonomous BBN."""


import bisect

import cartoblobpy.graph
import numpy as np

from .. import obstacles
from ..bbn import (
    bbn_autonomous_ranked_generated as bn,
)
from ..utils import Node
from .cost_model import CostModel, NormalizedExpectedLoss


class DynamicRisk:

    """Build map and vessel evidence, infer BBN posteriors, and evaluate loss.
    
    Parameters
    ----------
    graph : object, optional
        Loaded cartoblobpy graph. Supply one before evaluating nodes.
    inflation_sharpness : float, optional
        Retained compatibility setting; defaults to 10.0.
    inflation_base_shore_distance : bool, optional
        Retained compatibility setting, default False. Current evidence uses
        pure distance bins rather than an alternative inflation computation.
    inflation_base_depth : bool, optional
        Retained compatibility setting, default False. Current evidence uses depth bins.
    max_distance_from_shore : float, optional
        Upper scale for shore-distance bins in metres. Defaults to zero; supply
        a positive scale explicitly for meaningful distance evidence.
    max_dcpa_for_collision_risk : float, optional
        DCPA cutoff and bin scale in metres, default 150.0.
    max_tcpa_for_collision_risk : float, optional
        Future encounter horizon and bin scale in seconds, default 120.0.
    max_relative_velocity_for_collision_risk : float, optional
        Upper scale for relative-speed evidence in metres per second, default 10.0.
    max_depth_for_grounding_risk : float, optional
        Upper scale for depth bins in metres, default 150.0.
    lookahead_distance : float, optional
        Forward grounding-course lookahead in metres, default 30.0.
        Preserves the original graph predicate: any depth greater than the hard
        limit blocks the segment and produces grounding-course YES evidence.
    grounding_depth_hard_limit : float, optional
        Depth below which recovery-failure evidence is forced, default 5.0 metres.
    the_network : object, optional
        Network wrapper. None creates the packaged ranked autonomous BBN and
        requires a valid PySMILE license.
    obstacles : list, optional
        Static and moving obstacles. The list is copied; None creates a new empty list.
    desired_speed : float, optional
        Constant own-vessel speed for arrival time and CPA, default 5.0 metres per second.
    cost_model : CostModel, optional
        None creates an independent NormalizedExpectedLoss instance.
    
    Notes
    -----
    Each query clears previous evidence. Node.parent supplies incoming heading,
    and node.distance_cost supplies arrival distance. Moving vessels are advanced
    to arrival time before closest-approach geometry is computed. Only
    LinearDynamicObstacle instances participate in encounter evidence. Among
    qualifying future encounters, the selected one has smallest DCPA, then smallest
    TCPA, then largest relative speed.
    """
    def __init__(
        self,
        graph: cartoblobpy.graph.Graph = None,
        inflation_sharpness=10.0,
        inflation_base_shore_distance: bool = False,
        inflation_base_depth: bool = False,
        max_distance_from_shore: float = 0.0,
        max_dcpa_for_collision_risk: float = 150.0,
        max_tcpa_for_collision_risk: float = 120.0,
        max_relative_velocity_for_collision_risk: float = 10.0,
        max_depth_for_grounding_risk: float = 150.0,
        lookahead_distance: float = 30.0,
        grounding_depth_hard_limit: float = 5.0,
        the_network: bn.BayesianBeliefNetwork | None = None,
        obstacles: list[obstacles.Obstacle] | None = None,
        desired_speed: float = 5.0,
        cost_model: CostModel | None = None,
    ):

        """Initialize independent obstacle storage and the selected network and loss model."""
        self.bbn_network = the_network if the_network is not None else bn.BayesianBeliefNetwork()
        self.bbn_network.update_beliefs()

        self.lookahead_distance = lookahead_distance
        self.grounding_depth_hard_limit = grounding_depth_hard_limit

        self.max_distance_from_shore = max_distance_from_shore
        self.max_dcpa_for_collision_risk = max_dcpa_for_collision_risk
        self.max_tcpa_for_collision_risk = max_tcpa_for_collision_risk
        self.max_relative_velocity_for_collision_risk = max_relative_velocity_for_collision_risk
        self.max_depth_for_grounding_risk = max_depth_for_grounding_risk
        self.inflation_base_shore_distance = inflation_base_shore_distance
        self.inflation_base_depth = inflation_base_depth
        self.inflation_sharpness = inflation_sharpness
        self.graph = graph

        self.obstacles = list(obstacles) if obstacles is not None else []
        self.desired_speed = desired_speed

        self.cost_model = cost_model if cost_model is not None else NormalizedExpectedLoss()

    def set_evidence(self, node) -> float:
        # Clear all the evidence before setting new evidence for the current node
        """Replace previous evidence with environmental and encounter observations.
        
        Set grounding course, shore distance, depth, traffic, and collision-course
        inputs. Clear the network first so evidence does not leak between queries.
        Nodes below the grounding hard limit force recovery-failure evidence.
        """
        self.bbn_network.clear_all_evidence()

        distance = self._distance_to_shore(node)

        ## Provide grounding course evidence
        self.bbn_network.set_vessel_on_grounding_course(
            self._evidence_grounding_course(node)
        )

        ## Provide distance to shore evidence with optional inflation
        # distance_to_shore_evidence = None
        # if self.inflation_base_shore_distance:
        #     distance_to_shore_evidence = self._evidence_inflation_based_distance(distance)
        # else:
        distance_to_shore_evidence = self._evidence_pure_distance(distance)
        self.bbn_network.set_distance_to_shore(distance_to_shore_evidence)

        # self.bbn_network.set_own_vessel_fails_to_recover(
        #     bn.Own_vessel_fails_to_recover.LIKELY
        # )
        # self.bbn_network.set_target_vessel_fails_to_recover(
        #     bn.Target_vessel_fails_to_recover.LIKELY
        # )

        ## Provide depth evidence based on distance to shore with optional inflation
        self.bbn_network.set_depth(
            self._evidence_depth(node)
        )

        depth = self.graph.value_at((node.x, node.y), layer_name='depth')
        if depth < self.grounding_depth_hard_limit:
            self.bbn_network.set_failure_of_recovery_from_grounding_course(
                bn.Failure_of_recovery_from_grounding_course.YES
            )

        ## Provide ship traffic evidence based on graph's traffic layer
        self.bbn_network.set_ship_traffic(
            self._evidence_ship_traffic(node)
        )

        ## Provide DCPA evidence based on dynamic obstacles
        dcpa, tcpa, rv, encounter = self._evidence_encounter(node)

        # if encounter == bn.Encounter.LIKELY:
        #     print(f"Node ({node.x:.1f}, {node.y:.1f}): DCPA={dcpa}, TCPA={tcpa}, RelVel={rv}, Encounter={encounter}")

        self.bbn_network.set_dcpa(dcpa)
        self.bbn_network.set_tcpa(tcpa)
        self.bbn_network.set_relative_velocity(rv)
        self.bbn_network.set_encounter(encounter)

    def inference(self):
        # run BBN inference here
        """Update beliefs on the current network."""
        self.bbn_network.update_beliefs()

    def compute_risk(self, node: Node) -> float:


        """Clear and set evidence, infer posteriors, and return the scalar loss.
        
        The supplied node must have a parent providing its incoming heading and a
        distance_cost providing arrival distance. An uninitialized root is not a
        standalone risk query. The returned loss summarizes severity, not simply
        accident probability.
        """
        self.set_evidence(node)

        self.inference()

        cost = self.cost_model.compute_cost(self.bbn_network)

        # print(f"Node ({node.x:.1f}, {node.y:.1f}): Total Severity={total_severity}, Cost={cost}")

        return cost

    def _evidence_grounding_course(self, node: Node) -> bn.Vessel_on_grounding_course | None:
        """
        Check if the vessel is on a grounding course by looking ahead in the direction of travel.

        Parameters:
        - node: The current node representing the vessel's position.
        Returns the original study's enum evidence, or None for a zero heading.
        For reproducibility, the graph's default greater-than depth predicate is
        retained: a blocked segment maps to YES and a free segment maps to NO.
        """

        # course = np.arctan2(node.parent.y - node.y, node.parent.x - node.x)

        dx = node.x - node.parent.x
        dy = node.y - node.parent.y

        norm = np.linalg.norm([dx, dy])
        if norm < 1e-6:
            # If the node is very close to its parent, we cannot determine a reliable course.
            # In this case, we can choose to return None or a default value.
            print(f"Node ({node.x:.1f}, {node.y:.1f}) is very close to its parent. Cannot determine grounding course.")
            print(f"Parent Node: ({node.parent.x:.1f}, {node.parent.y:.1f})")
            return None

        unit_vector = np.array([dx, dy]) / np.linalg.norm([dx, dy])
        lookahead_point = np.array([node.x, node.y]) + unit_vector * self.lookahead_distance


        # Preserve the original study's default predicate and state ordering.
        # Changing this to a shallow-water predicate changes the risk surface.
        value = float(
            self.graph.is_free_path(
                [node.x, node.y],
                lookahead_point,
                layer_name='depth',
                threshold=self.grounding_depth_hard_limit,
            )
        )

        index = self._discretize_value(
            value,
            max_value=1.0,
            num_bins=len(bn.Vessel_on_grounding_course),
        )

        return bn.Vessel_on_grounding_course(index)

    def _distance_to_shore(self, node: Node) -> float:
        """
        Compute the minimum distance from the node to the nearest static obstacle (shoreline).
        """
        min_dist = float('inf')
        dist = self.graph.distance_to_closest_obstacle([node.x, node.y])
        min_dist = min(min_dist, dist)
        return min_dist

    def _evidence_pure_distance(self, distance: float) -> bn.Distance_to_shore:
        """
        Discretize the distance to shore into bins for BBN input.
        """

        index = self._discretize_value(
            distance,
            max_value=self.max_distance_from_shore,
            num_bins=len(bn.Distance_to_shore),
        )

        return bn.Distance_to_shore(index)

    def _evidence_depth(self, node: Node) -> bn.Depth:
        """
        Provide depth evidence based on the graph's depth layer value at the node's position.
        """

        depth = self.graph.value_at((node.x, node.y), layer_name='depth')

        index = self._discretize_value(
            depth,
            max_value=self.max_depth_for_grounding_risk,
            # max_value=5.0,
            num_bins=len(bn.Depth),
            reverse=False,
        )
        return bn.Depth(index)


    def _evidence_ship_traffic(self, node: Node) -> bn.Ship_traffic:
        """
        Provide traffic evidence based on the graph's traffic layer value at the node's position.
        """
        value = self.graph.value_at((node.x, node.y), layer_name='traffic')

        index = self._discretize_value(
            value,
            max_value=1.0,
            num_bins=len(bn.Ship_traffic),
            reverse=True
        )
        return bn.Ship_traffic(index)


    def _evidence_encounter(
            self,
            node: Node
        ) -> tuple[
            bn.DCPA | None,
            bn.TCPA | None,
            bn.Relative_velocity | None,
            bn.Encounter | None
        ]:
        """
        Compute DCPA evidence for BBN based on dynamic obstacles.
        """

        speed = self.desired_speed
        time_to_node = node.distance_cost / speed if speed > 0 else 0

        # Get all the LinearDynamicObstacles from the obstacle list
        dynamic_obstacles = [obs for obs in self.obstacles if isinstance(obs, obstacles.LinearDynamicObstacle)]

        encounter_qualities = []

        course = np.arctan2(node.y - node.parent.y, node.x - node.parent.x)
        # course = np.arctan2(node.parent.y - node.y, node.parent.x - node.x)

        for obs in dynamic_obstacles:

            pose_at_t = obs.position(time_to_node)

            tcpa, dcpa, v_os, v_ts, dp, dv = obstacles.compute_cpa(
                eta_os=np.array([node.x, node.y, course]),
                eta_ts=pose_at_t,
                u_os=self.desired_speed,
                u_ts=obs.velocity
            )

            collision_risk = (
                (dcpa < self.max_dcpa_for_collision_risk)
                and
                (0 < tcpa < self.max_tcpa_for_collision_risk)
            )

            if collision_risk:
                 encounter_qualities.append({
                    'DCPA': dcpa,
                    'TCPA': tcpa,
                    'RelVel': np.linalg.norm(dv)
                })

        if not encounter_qualities:
            # No collision course detected
            return [None, None, None, bn.Encounter.UNLIKELY]

        # We have collision otherwise.

        # find the worst collision risk based on DCPA, then TCPA, then relative velocity
        encounter_qualities.sort(key=lambda x: (x['DCPA'], x['TCPA'], -x['RelVel']))

        worst_encounter = encounter_qualities[0]

        dcpa_index = self._discretize_value(
            worst_encounter['DCPA'],
            max_value=self.max_dcpa_for_collision_risk,
            num_bins=len(bn.DCPA),
        )

        tcpa_index = self._discretize_value(
            worst_encounter['TCPA'],
            max_value=self.max_tcpa_for_collision_risk,
            num_bins=len(bn.TCPA),
        )

        rel_vel_index = self._discretize_value(
            worst_encounter["RelVel"],
            max_value=self.max_relative_velocity_for_collision_risk,
            num_bins=len(bn.Relative_velocity),
        )

        return (
            bn.DCPA(dcpa_index),
            bn.TCPA(tcpa_index),
            bn.Relative_velocity(rel_vel_index),
            bn.Encounter.LIKELY
        )

    @staticmethod
    def _discretize_value(
        value: float,
        max_value: float,
        num_bins: int,
        min_value: float = 0.0,
        reverse: bool = False
    ) -> int:
        """
        Discretize a continuous value into bins for BBN input.

        Parameters:
        - value: The continuous value to discretize.
        - max_value: The maximum value corresponding to the last bin.
        - num_bins: The total number of bins (excluding the None state).
        - min_value: The minimum value corresponding to the first bin (default is 0.0).
        - reverse: Whether to reverse the binning (higher values get lower indexes).

        Example:
        >>> _discretize_value(0.3, max_value=1.0, num_bins=5)  #> 1

        Returns:
        - An integer index representing the discretized bin for BBN input.
        """
        if value is None:
            return None

        # Generate bin edges: for N bins, we need N-1 edges.
        bins = [min_value + i * (max_value - min_value) / num_bins for i in range(1, num_bins)]

        return DynamicRisk._discretize_value_with_intervals(value, bins, reverse)

    @staticmethod
    def _discretize_value_with_intervals(
        value: float,
        bins: list[float],
        reverse: bool = False
    ) -> int:
        """
        Discretize a continuous value into bins defined by specific intervals.

        Parameters:
        - value: The continuous value to discretize.
        - bins: A list of bin edges defining the intervals. Must be sorted in ascending order.
        - reverse: Whether to reverse the binning (higher values get lower indexes).

        Example:
        >>> bins = [0.0, 0.5, 1.0]
        >>> _discretize_value_with_intervals(0.3, bins)  #> 1
        >>> _discretize_value_with_intervals(0.6, bins)  #> 2

        Returns:
        - An integer index representing the discretized bin for BBN input.

        """
        if value is None:
            return None

        # bisect_right returns:
        # 0 if value < bins[0]
        # len(bins) if value >= bins[-1]
        index = bisect.bisect_right(bins, value)

        if reverse:
            # If there are N bins (len(bins) + 1),
            # reversing maps index 0 -> N-1, index N-1 -> 0.
            return len(bins) - index

        return index
