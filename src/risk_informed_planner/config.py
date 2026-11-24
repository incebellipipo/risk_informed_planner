
# config.py
import dataclasses
from typing import Tuple

@dataclasses.dataclass
class Config:
    """Configure tree growth, sampled edge risk, and heading-change penalties.
    
    All distances use the loaded map's world-coordinate units (metres in the study
    maps), and heading changes use radians. Construct a separate configuration for
    each experiment. Map geometry, vessel speed, obstacles, and evidence thresholds
    are supplied separately to the planner and risk model.
    
    ``MAX_NODES`` and ``MAX_ITER`` are stopping targets after a collision-free
    geometric approach to the goal is found, not hard time or iteration limits.
    At ``ALPHA = 1``, the risk-exposure cost disappears but risk cutoffs still apply.
    The turn penalty is additive and does not rescale the distance/risk tradeoff.
    """

    X_LIMITS: Tuple[int, int] = (0, 100)
    """X extent for the workspace and plots. Sampling uses the graph geometry."""
    Y_LIMITS: Tuple[int, int] = (0, 100)
    """Y extent for the workspace and plots. Sampling uses the graph geometry."""
    STEP_SIZE: int = 2
    """Maximum nominal extension length in metres."""
    MAX_ITER: int = None
    """Iteration stopping target after a geometric goal approach is found."""
    MAX_NODES: int = None
    """Tree-size stopping target after a geometric goal approach is found."""
    MAX_NEIGHBORS_RISK_RENEWAL: int = None
    """Cap on neighbors considered during rewiring; None considers all neighbors in radius."""
    GOAL_RADIUS: int = 7
    """Distance within which a new node can attempt a goal connection."""
    NEIGHBOR_RADIUS: int = 5
    """Radius for parent candidates and rewiring neighbors."""
    ALPHA: float = 1.0  # Distance weight
    """Distance weight in the edge cost; the risk term is multiplied by 1 - ALPHA."""
    BETA: float = 10.0  # Risk weight
    """Risk-exposure scale within the edge cost."""
    GAMMA: float = 0.0  # Turn weight (angle-aware planner only; 0 = no penalty)
    """Additional heading-change penalty weight."""
    TURN_REFERENCE_LENGTH: float = None
    """Distance scale for a half-turn; None uses STEP_SIZE."""
    MAX_TURN_ANGLE: float = None
    """Maximum allowed heading change in radians; None disables the limit."""
    RISK_SAMPLE_SPACING: float = None  # [m]; None = endpoint only (legacy)
    """Spacing in metres between edge-risk samples; None uses the endpoint only."""
    RISK_AGGREGATION: str = "mean"     # "mean" | "max" | "endpoint"
    """Aggregation of edge samples; choose mean, max, or endpoint."""
    MAX_RISK_SAMPLES_PER_EDGE: int = 32
    """Upper bound on inference samples per edge."""
    GRID_RESOLUTION: int = 1
    """Retained grid setting; tree sampling uses the map resolution."""
    RISK_CUTOFF: float = 1.0
    """Largest permitted sampled peak risk for connected tree edges and the goal."""
