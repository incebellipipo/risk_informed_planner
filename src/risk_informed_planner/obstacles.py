"""Obstacle geometry and constant-heading closest-approach calculations."""
import abc
import numpy as np


class Obstacle(abc.ABC):
    """Abstract obstacle interface requiring position at a supplied time."""
    @abc.abstractmethod
    def position(self, t):
        """Return the obstacle geometry at time t, in seconds."""
        pass

class StaticObstacle(Obstacle):
    """Represent an axis-aligned rectangle using two world-coordinate corners.
    
    Parameters
    ----------
    bottom_left, top_right : tuple
        Rectangle corner coordinate pairs.
    
    Notes
    -----
    ``position(t)`` returns the same two corners at every time. Rectangles can be
    drawn but do not participate in moving-vessel encounter evidence. Static
    collision geometry is supplied by the graph.
    """
    def __init__(self, bottom_left, top_right):
        self.bottom_left = np.array(bottom_left)
        self.top_right = np.array(top_right)

    def position(self, t):
        """Return the two corner arrays; time does not change a static rectangle."""
        return (self.bottom_left, self.top_right)

class LinearDynamicObstacle(Obstacle):
    """Represent a vessel moving at constant heading and speed.
    
    Parameters
    ----------
    x, y : float
        Initial world coordinates in metres.
    yaw : float
        Constant heading in radians.
    velocity : float
        Constant surge speed in metres per second.
    
    Notes
    -----
    Motion is extrapolated for every supplied time. The obstacle has no destination
    or automatic stopping behavior.
    """
    def __init__(self, x, y, yaw, velocity):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.velocity = velocity

    def position(self, t) -> np.ndarray:
        """Return a NumPy array containing x(t), y(t), and yaw.
        
        Time is in seconds relative to the initial pose. Heading remains constant.
        """
        new_x = self.x + self.velocity * np.cos(self.yaw) * t
        new_y = self.y + self.velocity * np.sin(self.yaw) * t

        return np.array((new_x, new_y, self.yaw), dtype=float)

    @staticmethod
    def create(start_pose, end_pose, velocity):
        """Construct a vessel with the heading from start_pose to end_pose.
        
        The poses are coordinate pairs. The endpoint defines the course direction,
        not a destination at which the vessel stops. Velocity is in metres per second.
        """
        start_x, start_y = start_pose
        end_x, end_y = end_pose

        # Calculate the direction from start to end
        direction_x = end_x - start_x
        direction_y = end_y - start_y
        yaw = np.arctan2(direction_y, direction_x)

        return LinearDynamicObstacle(start_x, start_y, yaw, velocity)


def compute_cpa(eta_os, eta_ts, u_os, u_ts):
    """
    Compute TCPA and DCPA between ownship and target vessel.

    Assumes zero sway velocity (v = 0) for both vessels, so each
    vessel moves along its heading with surge speed only. The
    computation uses linear extrapolation of current states.

    Parameters
    ----------
    eta_os : numpy.ndarray, shape (3,)
        Ownship pose in NED frame, ``[x, y, psi]`` where
        ``x`` is North position [m], ``y`` is East position [m],
        and ``psi`` is heading [rad].
    eta_ts : numpy.ndarray, shape (3,)
        Target ship pose in NED frame, ``[x, y, psi]`` where
        ``x`` is North position [m], ``y`` is East position [m],
        and ``psi`` is heading [rad].
    u_os : float
        Ownship surge speed [m/s]. Must be non-negative.
    u_ts : float
        Target ship surge speed [m/s]. Must be non-negative.

    Returns
    -------
    tcpa : float
        Time to closest point of approach [s]. Positive means CPA
        is in the future, negative means vessels are diverging.
        Returns ``numpy.inf`` if relative velocity is near zero.
    dcpa : float
        Distance at closest point of approach [m]. If
        ``tcpa = numpy.inf``, this equals the current range.
    v_os : numpy.ndarray, shape (2,)
        Ownship velocity in NED frame, ``[v_N, v_E]`` [m/s].
    v_ts : numpy.ndarray, shape (2,)
        Target ship velocity in NED frame, ``[v_N, v_E]`` [m/s].
    dp : numpy.ndarray, shape (2,)
        Relative position vector (target minus ownship) in NED
        frame, ``[dN, dE]`` [m].
    dv : numpy.ndarray, shape (2,)
        Relative velocity vector (target minus ownship) in NED
        frame, ``[dv_N, dv_E]`` [m/s].

    Examples
    --------
    >>> import numpy as np
    >>> eta_os = np.array([0.0, 0.0, np.radians(45)])
    >>> eta_ts = np.array([300.0, 100.0, np.radians(180)])
    >>> tcpa, dcpa, *_ = compute_cpa(eta_os, eta_ts, 5.0, 4.0)
    """
    # Unpack poses
    p_os, psi_os = eta_os[:2], eta_os[2]
    p_ts, psi_ts = eta_ts[:2], eta_ts[2]

    # Relative position
    dp = p_ts - p_os

    # NED velocities (v=0 => only surge along heading)
    v_os = u_os * np.array([np.cos(psi_os), np.sin(psi_os)])
    v_ts = u_ts * np.array([np.cos(psi_ts), np.sin(psi_ts)])
    dv = v_ts - v_os

    dv_sq = np.dot(dv, dv)
    if dv_sq < 1e-12:
        return np.inf, np.linalg.norm(dp), v_os, v_ts, dp, dv

    tcpa = -np.dot(dp, dv) / dv_sq
    dcpa = np.linalg.norm(dp + dv * tcpa)

    return tcpa, dcpa, v_os, v_ts, dp, dv

