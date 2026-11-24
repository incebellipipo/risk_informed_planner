"""Planner nodes and B-spline path smoothing for the study notebooks."""

from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np
from scipy.interpolate import splprep, splev

Point = Tuple[float, float]


class Node:
    """Represent a world-coordinate point and its tree metadata.
    
    Parameters
    ----------
    x, y : float
        World coordinates in metres for the study maps.
    idx : int
        Node identifier used for bookkeeping.
    
    Attributes
    ----------
    parent : Node or None
        Incoming tree link. The root and disconnected risk observations have no parent.
    cost : float
        Accumulated cost from the root.
    heuristic : float
        Cost of the incoming edge alone.
    risk : float
        Aggregate sampled risk on the incoming edge.
    peak_risk : float
        Largest sampled risk on that edge, used by cutoff checks.
    distance_cost : float
        Distance travelled from the root, used to infer arrival time.
    turn_angle : float
        Incoming heading change in radians.
    bbn_snapshot : dict
        Endpoint posterior values and state names. Each node has its own dictionary.
    """


    def __init__(self, x: float, y: float, idx: int) -> None:
        """Initialize a node.

        Args:
            x (float): X coordinate.
            y (float): Y coordinate.
            idx (int): Index/identifier for bookkeeping.
        """
        self.x = x
        self.y = y
        self.idx = idx
        self.parent = None
        self.cost = 0.0
        self.risk = 0.0
        self.peak_risk = 0.0
        self.heuristic = 0.0
        self.distance_cost = 0.0
        self.turn_angle = 0.0
        self.bbn_snapshot = {}

    def pos(self) -> Point:
        """Return the node position as a tuple.

        Returns:
            Tuple[float, float]: ``(x, y)`` coordinates.
        """
        return (self.x, self.y)

def smooth_path(path: Sequence[Node], degree=5) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a B-spline to an ordered node path and sample 100 points.
    
    Parameters
    ----------
    path : sequence
        Nodes with x/y coordinates. Supply enough distinct points for SciPy's
        default cubic spline.
    degree : float, optional
        Smoothing parameter passed to ``scipy.interpolate.splprep`` as ``s``.
        Despite its historical name, this does not select the spline degree.
    
    Returns
    -------
    tuple
        NumPy arrays ``(x_smooth, y_smooth)``, each of length 100.
    
    Notes
    -----
    The resulting curve is not checked again for collisions or risk. Increasing
    ``degree`` allows more smoothing; zero requests interpolation through the points.
    """
    x = [node.x for node in path]
    y = [node.y for node in path]
    tck, _ = splprep([x, y], s=degree)
    x_smooth, y_smooth = splev(np.linspace(0, 1, 100), tck)
    return x_smooth, y_smooth
