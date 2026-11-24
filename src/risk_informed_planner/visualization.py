"""Plot tree samples, saved BBN posteriors, routes, and moving-vessel campaigns.

Heatmaps interpolate the sampled nodes rather than evaluating a new dense BBN
risk grid. Panel functions return Matplotlib axes and can be used with saved
PlannerRun keyword arguments. World coordinates are inherited from the map.
"""

# planner/visualization.py
from pathlib import Path

import cartoblobpy.graph
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import IdentityTransform
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .obstacles import Obstacle, LinearDynamicObstacle, StaticObstacle
from .utils import Node





def plot_nodes_with_values(nodes, cfg, key=None, ax=None, min_value=0.0, add_cbar=False, vmin=None, vmax=None, **plt_kwargs) -> plt.Axes:
    """Draw nodes colored by a scalar attribute.

    Args:
        nodes: Iterable of nodes with attributes ``x``, ``y``, ``heuristic`` and optional ``parent``.
        key: Attribute to plot, such as 'heuristic', 'risk', or 'cost'.
        cfg: Configuration object providing ``X_LIMITS`` and ``Y_LIMITS``.
        ax: Optional existing Matplotlib ``Axes``.
        vmin, vmax: Fixed color-scale bounds. Defaults to the min/max of the
            plotted values (as before), which makes the color scale -- and
            therefore the figure -- differ from run to run. Pass fixed
            bounds to make multiple figures comparable.
        **plt_kwargs: Forwarded to ``ax.scatter``.

    Returns:
        plt.Axes: The axes containing the plot.
    """
    import matplotlib as mpl

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_ax = True


    ax.set_xlim(*cfg.X_LIMITS)
    ax.set_ylim(*cfg.Y_LIMITS)
    ax.set_aspect('equal')

    selected_nodes = [node for node in nodes if getattr(node, key, 0.0) > min_value]

    x_coords = [node.x for node in selected_nodes]
    y_coords = [node.y for node in selected_nodes]
    value = [getattr(node, key) for node in selected_nodes]

    cmap_name = plt_kwargs.pop("cmap", "cool")

    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(
        vmin=vmin if vmin is not None else min(value),
        vmax=vmax if vmax is not None else max(value),
    )

    for node in selected_nodes:
        n_value = getattr(node, key)
        if getattr(node, "parent", None) is not None:
            color = cmap(norm(n_value))
            ax.plot(
                [node.x, node.parent.x],
                [node.y, node.parent.y],
                color=color,
                linewidth=0.2,
                alpha=0.3,
                zorder=0,
            )

    scatter = ax.scatter(
        x_coords,
        y_coords,
        c=value,
        cmap=cmap,
        norm=norm,
        s=10,
        **plt_kwargs
    )

    if add_cbar:
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
        # cbar = plt.colorbar(im, ax=ax, cax=cax)

        cbar = plt.colorbar(scatter, ax=ax, cax=cax)
        cbar.set_label(f'Node {key} Value')

    if created_ax:
        plt.tight_layout()
    return ax


def plot_nodes_with_bbn_state(nodes, bbn_node_id, state_name, cfg, ax=None, min_probability=0.0, add_cbar=False, vmin=None, vmax=None, **plt_kwargs) -> plt.Axes:
    """Plot RRT nodes colored by a BBN node's state probability.

    Args:
        nodes: Iterable of RRT nodes with ``x``, ``y`` attributes, optional ``parent``,
            and a ``bbn_snapshot`` attribute (output of BBN.snapshot()).
        bbn_node_id (str): The BBN node ID to plot (e.g., 'Collision', 'Grounding').
        state_name (str): The state name within that node (e.g., 'Yes', 'No').
        cfg: Configuration object providing ``X_LIMITS`` and ``Y_LIMITS``.
        ax: Optional existing Matplotlib ``Axes``.
        min_probability (float): Minimum probability threshold for filtering nodes.
        add_cbar (bool): Whether to add a colorbar.
        vmin, vmax: Fixed color-scale bounds. Defaults to the min/max of the
            plotted probabilities (as before); pass fixed bounds (e.g. 0/1)
            to make multiple figures comparable.
        **plt_kwargs: Forwarded to ``ax.scatter``.

    Returns:
        plt.Axes: The axes containing the plot.
    """
    import matplotlib as mpl

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_ax = True

    ax.set_xlim(*cfg.X_LIMITS)
    ax.set_ylim(*cfg.Y_LIMITS)
    ax.set_aspect('equal')

    # Extract probabilities from node snapshots
    selected_nodes = []
    probabilities = []

    for node in nodes:
        if not hasattr(node, 'bbn_snapshot') or not node.bbn_snapshot:
            continue

        snapshot = node.bbn_snapshot
        if bbn_node_id not in snapshot:
            continue

        node_data = snapshot[bbn_node_id]
        states = node_data['states']
        values = node_data['values']

        # Find the index of the requested state
        try:
            state_idx = states.index(state_name)
            prob = values[state_idx]
            if prob > min_probability:
                selected_nodes.append(node)
                probabilities.append(prob)
        except ValueError:
            # State not found in this node
            continue

    if not selected_nodes:
        if created_ax:
            plt.tight_layout()
        return ax

    x_coords = [node.x for node in selected_nodes]
    y_coords = [node.y for node in selected_nodes]

    cmap_name = plt_kwargs.pop("cmap", "RdYlGn")

    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(
        vmin=vmin if vmin is not None else min(probabilities),
        vmax=vmax if vmax is not None else max(probabilities),
    )

    # Plot tree connections
    for node in selected_nodes:
        prob = probabilities[selected_nodes.index(node)]
        if getattr(node, "parent", None) is not None:
            color = cmap(norm(prob))
            ax.plot(
                [node.x, node.parent.x],
                [node.y, node.parent.y],
                color=color,
                linewidth=0.2,
                alpha=0.3,
                zorder=0,
            )

    scatter = ax.scatter(
        x_coords,
        y_coords,
        c=probabilities,
        cmap=cmap,
        norm=norm,
        **plt_kwargs
    )

    if add_cbar:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f'P({bbn_node_id}={state_name})')

    if created_ax:
        plt.tight_layout()
    return ax



def plot_heatmap(nodes, cfg, key='risk', ax=None, add_cbar=False, vmin=None, vmax=None, min_value=None, max_value=None, **plt_kwargs) -> plt.Axes:
    """Plot an interpolated risk heatmap based on nodes' risk values.

    Args:
        vmin, vmax: Fixed color-scale bounds forwarded to ``ax.imshow``.
            Defaults to Matplotlib's usual behavior of scaling to the data's
            own min/max, which makes the scale differ from run to run; pass
            fixed bounds to make multiple figures comparable.
        min_value, max_value: Values below/above these are masked out
            (rendered transparent), not merely a color-scale bound. ``None``
            leaves that side unmasked.
    """
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_ax = True
    ax.set_xlim(*cfg.X_LIMITS)
    ax.set_ylim(*cfg.Y_LIMITS)
    ax.set_aspect('equal')

    if not nodes:
        return ax

    if vmin is not None:
        plt_kwargs.setdefault("vmin", vmin)
    if vmax is not None:
        plt_kwargs.setdefault("vmax", vmax)

    x_coords = np.array([node.x for node in nodes])
    y_coords = np.array([node.y for node in nodes])
    values = np.array([getattr(node, key) for node in nodes], dtype=float)

    # Dense grid for interpolation
    nx, ny = 200, 200
    grid_x, grid_y = np.mgrid[
        cfg.X_LIMITS[0]:cfg.X_LIMITS[1]:complex(0, nx),
        cfg.Y_LIMITS[0]:cfg.Y_LIMITS[1]:complex(0, ny),
    ]

    # Linear interpolation inside the convex hull, nearest outside.
    # This avoids the hard fill_value=0 edge AND avoids cubic's overshoot.
    grid_values = griddata(
        (x_coords, y_coords), values, (grid_x, grid_y),
        method='linear',
    )
    # Fill NaN regions (outside convex hull) with nearest-neighbor values
    # so the Gaussian smoothing below doesn't create edge artifacts.
    nan_mask = np.isnan(grid_values)
    if nan_mask.any():
        nearest = griddata(
            (x_coords, y_coords), values, (grid_x, grid_y),
            method='nearest',
        )
        grid_values[nan_mask] = nearest[nan_mask]

    # Smooth to get a continuous surface. sigma is in grid cells.
    # sigma=2-4 gives visible smoothing without washing out structure.
    grid_values = gaussian_filter(grid_values, sigma=3.0)

    # mask out-of-range cells; masked entries render transparent
    grid_values = np.ma.masked_outside(
        grid_values,
        min_value if min_value is not None else -np.inf,
        max_value if max_value is not None else np.inf,
    )

    im = ax.imshow(
        grid_values.T,
        extent=[*cfg.X_LIMITS, *cfg.Y_LIMITS],
        origin='lower',
        **plt_kwargs,
    )

    if add_cbar:
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, ax=ax, cax=cax)

    if created_ax:
        plt.tight_layout()
    return ax


def plot_heatmap_bbn_state(nodes, bbn_node_id, state_name, cfg, ax=None, add_cbar=False, min_probability=0.0, max_probability=1.0, vmin=None, vmax=None, **plt_kwargs) -> plt.Axes:
    """Plot an interpolated heatmap from node BBN snapshot state probabilities.

    Args:
        nodes: Iterable of nodes with ``x``, ``y``, and ``bbn_snapshot``.
            ``bbn_snapshot`` is expected to come from ``BBN.snapshot()``.
        bbn_node_id (str): BBN node id to read from each snapshot.
        state_name (str): State name within the BBN node.
        cfg: Configuration object providing ``X_LIMITS`` and ``Y_LIMITS``.
        ax: Optional existing Matplotlib ``Axes``.
        add_cbar (bool): Whether to add a colorbar.
        min_probability (float): Values below this are masked out (rendered
            transparent), not merely a color-scale bound.
        max_probability (float): Values above this are masked out (rendered
            transparent), not merely a color-scale bound.
        vmin, vmax: Fixed color-scale bounds forwarded to ``ax.imshow``,
            independent of the ``min_probability``/``max_probability``
            masking above. Defaults to Matplotlib's own data-driven scaling;
            pass e.g. ``vmin=0, vmax=1`` to make multiple figures comparable.
        **plt_kwargs: Forwarded to ``ax.imshow``.

    Returns:
        plt.Axes: The axes containing the heatmap.
    """
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_ax = True

    ax.set_xlim(*cfg.X_LIMITS)
    ax.set_ylim(*cfg.Y_LIMITS)
    ax.set_aspect('equal')

    filtered_nodes = []
    probabilities = []

    for node in nodes:
        snapshot = getattr(node, "bbn_snapshot", None)
        if not snapshot or bbn_node_id not in snapshot:
            continue

        node_data = snapshot[bbn_node_id]
        states = node_data.get("states", [])
        values = node_data.get("values", [])

        try:
            state_idx = states.index(state_name)
            prob = float(values[state_idx])
        except (ValueError, IndexError, TypeError):
            continue

        # if prob > min_probability:
        filtered_nodes.append(node)
        probabilities.append(prob)

    if not filtered_nodes:
        if created_ax:
            plt.tight_layout()
        return ax

    x_coords = np.array([node.x for node in filtered_nodes], dtype=float)
    y_coords = np.array([node.y for node in filtered_nodes], dtype=float)
    values = np.array(probabilities, dtype=float)

    nx, ny = 200, 200
    grid_x, grid_y = np.mgrid[
        cfg.X_LIMITS[0]:cfg.X_LIMITS[1]:complex(0, nx),
        cfg.Y_LIMITS[0]:cfg.Y_LIMITS[1]:complex(0, ny),
    ]

    # For sparse samples, fall back to nearest to avoid QHull failures.
    method = 'linear' if len(filtered_nodes) >= 3 else 'nearest'
    grid_values = griddata(
        (x_coords, y_coords), values, (grid_x, grid_y),
        method=method,
    )

    if method == 'linear':
        nan_mask = np.isnan(grid_values)
        if nan_mask.any():
            nearest = griddata(
                (x_coords, y_coords), values, (grid_x, grid_y),
                method='nearest',
            )
            grid_values[nan_mask] = nearest[nan_mask]

    grid_values = gaussian_filter(grid_values, sigma=3.0)

    # mask out-of-range cells; masked entries render transparent
    masked = np.ma.masked_outside(grid_values, min_probability, max_probability)

    plt_kwargs.setdefault("cmap", "viridis")
    if vmin is not None:
        plt_kwargs.setdefault("vmin", vmin)
    if vmax is not None:
        plt_kwargs.setdefault("vmax", vmax)
    # plt_kwargs.setdefault("interpolation", "nearest")

    im = ax.imshow(
        masked.T,
        extent=[*cfg.X_LIMITS, *cfg.Y_LIMITS],
        origin="lower",
        **plt_kwargs,
    )
    # grid_values = gaussian_filter(grid_values, sigma=3.0)

    # # replace probabilities less than min_probability with NaN to make them transparent in the plot
    # grid_values[(grid_values < min_probability) | (grid_values > max_probability)] = np.nan

    # plt_kwargs.setdefault("cmap", "viridis")
    # im = ax.imshow(
    #     grid_values.T,
    #     extent=[*cfg.X_LIMITS, *cfg.Y_LIMITS],
    #     origin='lower',
    #     **plt_kwargs,
    # )

    if add_cbar:
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, ax=ax, cax=cax)
        cbar.set_label(f'P({bbn_node_id}={state_name})')

    if created_ax:
        plt.tight_layout()
    return ax


def plot_nodes_bbn_state(nodes, bbn_node_id, state_name, cfg, ax=None, min_probability=0.0, max_probability=1.0, **plt_kwargs) -> plt.Axes:
    """Plot RRT nodes colored by a BBN node's state probability.

    Args:
        nodes: Iterable of RRT nodes with ``x``, ``y`` attributes, optional ``parent``,
            and a ``bbn_snapshot`` attribute (output of BBN.snapshot()).
        bbn_node_id (str): The BBN node ID to plot (e.g., 'Collision', 'Grounding').
        state_name (str): The state name within that node (e.g., 'Yes', 'No').
        cfg: Configuration object providing ``X_LIMITS`` and ``Y_LIMITS``.
        ax: Optional existing Matplotlib ``Axes``.
        min_probability (float): Minimum probability threshold for filtering nodes.
        **plt_kwargs: Forwarded to ``ax.scatter``.

    Returns:
        plt.Axes: The axes containing the plot.
    """
    import matplotlib as mpl

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_ax = True

    ax.set_xlim(*cfg.X_LIMITS)
    ax.set_ylim(*cfg.Y_LIMITS)
    ax.set_aspect('equal')

    selected_nodes = []
    probabilities = []

    for node in nodes:
        if not hasattr(node, 'bbn_snapshot') or not node.bbn_snapshot:
            continue

        snapshot = node.bbn_snapshot
        if bbn_node_id not in snapshot:
            continue

        node_data = snapshot[bbn_node_id]
        states = node_data['states']
        values = node_data['values']

        try:
            state_idx = states.index(state_name)
            prob = values[state_idx]
            if min_probability <= prob <= max_probability:
                selected_nodes.append(node)
                probabilities.append(prob)
        except ValueError:
            continue

    if not selected_nodes:
        if created_ax:
            plt.tight_layout()
        return ax

    x_coords = [node.x for node in selected_nodes]
    y_coords = [node.y for node in selected_nodes]

    cmap_name = plt_kwargs.pop("cmap", "RdYlGn")

    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=min(probabilities), vmax=max(probabilities))

    for node in selected_nodes:
        prob = probabilities[selected_nodes.index(node)]
        if getattr(node, "parent", None) is not None:
            color = cmap(norm(prob))

            ax.plot(
                [node.x, node.parent.x],
                [node.y, node.parent.y],
                color=color,
                linewidth=0.2,
                alpha=0.3,
                zorder=0,
            )

    scatter = ax.scatter(
        x_coords,
        y_coords,
        c=probabilities,
        cmap=cmap,
        norm=norm,
        **plt_kwargs
    )
    if created_ax:
        plt.tight_layout()
    return ax

def plot_static_obstacles(static_obstacles, ax=None, **plt_kwargs) -> plt.Axes:
    """Helper to plot static obstacles on given axes.

    Args:
        ax: Matplotlib Axes to draw on.
        static_obstacles: Iterable of static rectangular obstacles defined
            as ((x_min, y_min), (x_max, y_max)).

    Returns:
        None
    """
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_ax = True

    first_label = plt_kwargs.pop("label", "Static Obstacles")
    for (x_min, y_min), (x_max, y_max) in static_obstacles:
        label = first_label
        first_label = None  # Only label the first obstacle for legend
        ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, **plt_kwargs, label=label))

    if created_ax:
        plt.tight_layout()

    return ax

def plot_start_and_goal(start, goal, ax=None, **plt_kwargs) -> plt.Axes:
    """Helper to plot start and goal nodes on given axes.

    Args:
        ax: Matplotlib Axes to draw on.
        start: Start node with ``x`` and ``y`` attributes.
        goal: Goal node with ``x`` and ``y`` attributes.
    Returns:
        None
    """
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_ax = True

    start_color = plt_kwargs.pop("start_color", "green")
    goal_color = plt_kwargs.pop("goal_color", "red")

    ax.plot(start.x, start.y, marker='o', color=start_color, markersize=10, label='Start', **plt_kwargs)
    ax.plot(goal.x, goal.y, marker='*', color=goal_color, markersize=15, label='Goal', **plt_kwargs)
    ax.legend()

    if created_ax:
        plt.tight_layout()

    return ax

def plot_obstacles(
        obstacles: list[Obstacle],
        max_time: float = 60,
        t_computation: float = None,
        ax=None,
        course_label: str = "Target Course",
        start_label: str = "Target Start",
        position_label: str = None,
        marker_size: float = 9,
        outline_color: str = None,
        markers: bool = True,
        **plt_kwargs,
    ) -> plt.Axes:
    """Helper to plot a list of Obstacle objects on given axes.

    Dynamic obstacles are drawn as their course -- a dashed line from t=0 to
    ``max_time`` with a short directional arrowhead at the midpoint -- plus
    a circle at the obstacle's starting position and, if ``t_computation``
    is given, a cross marking where the obstacle will be once the plan has
    finished computing. The circle/cross pair mirrors how the ownship's own
    start and current-time position are marked (:func:`plot_start_and_goal`,
    :func:`plot_vessel_on_path_at_t`).

    Args:
        obstacles: List of Obstacle objects, each with attributes defining its shape and position.
        max_time: Duration (s) to draw the dynamic obstacle course for.
        t_computation: If given, mark each dynamic obstacle's position at this time.
        ax: Matplotlib Axes to draw on.
        course_label: Legend label for the obstacle course line.
        start_label: Legend label for the obstacle's starting position (t=0).
        position_label: Legend label for the ``t_computation`` marker.
            Defaults to a label that includes ``t_computation``.
        marker_size: Marker size used for the start/``t_computation`` markers.
        outline_color: If given, the dynamic obstacles' course line, arrowhead,
            and markers get a thin outline in this color, which keeps them
            visible on top of a heatmap with similar colors.
        markers: ``False`` draws only the dynamic obstacles' course (line
            and arrowhead), without the start and ``t_computation`` markers.
        **plt_kwargs: Forwarded to the course line and the static obstacle
            patches (e.g. ``color``, ``alpha``, ``linewidth``); ``linewidth``
            also sets the arrowhead's line width.

    Returns:
        plt.Axes: The axes containing the plot.
    """
    from matplotlib import patheffects

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_ax = True

    color = plt_kwargs.pop("color", "dimgray")
    linewidth = plt_kwargs.get("linewidth", plt_kwargs.get("lw", 1.5))
    outline = {}
    marker_outline = {}
    if outline_color is not None:
        outline = dict(path_effects=[
            patheffects.Stroke(linewidth=linewidth + 2.0, foreground=outline_color),
            patheffects.Normal(),
        ])
        marker_outline = dict(markeredgecolor=outline_color, markeredgewidth=1.2)

    if position_label is None and t_computation is not None:
        position_label = f"Target Position at t={t_computation:.1f}s"

    course_labeled = False
    start_labeled = False
    position_labeled = False
    static_labeled = False

    for obs in obstacles:
        if isinstance(obs, LinearDynamicObstacle):
            x0 = obs.position(t=0)
            x1 = obs.position(t=max_time)

            # Obstacle course as a dashed line.
            ax.plot(
                [x0[0], x1[0]], [x0[1], x1[1]],
                color=color, linestyle='--',
                label=None if course_labeled else course_label,
                **outline,
                **plt_kwargs,
            )
            course_labeled = True

            # Short directional arrowhead at the course midpoint. Anchored
            # between two points close together (not from x0 to the
            # midpoint) so only the arrowhead is visible -- annotate draws a
            # full solid shaft between its two points, which otherwise
            # overlays half the dashed line and reads as a second, solid path.
            mid_time = max_time / 2.0
            arrow_span = max_time * 0.03
            p_before = obs.position(t=max(mid_time - arrow_span, 0.0))
            p_after = obs.position(t=min(mid_time + arrow_span, max_time))
            ax.annotate(
                '',
                xy=(p_after[0], p_after[1]),
                xytext=(p_before[0], p_before[1]),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=linewidth, shrinkA=0, shrinkB=0, **outline),
            )

            if not markers:
                continue

            # Starting position, mirroring the ownship start marker.
            ax.plot(
                x0[0], x0[1],
                marker='o', color=color, linestyle='None',
                markersize=marker_size, zorder=5,
                label=None if start_labeled else start_label,
                **marker_outline,
            )
            start_labeled = True

            # Where the obstacle will be once the plan has finished
            # computing, mirroring the ownship's own 'X' position marker.
            if t_computation is not None:
                pos_t = obs.position(t=t_computation)
                ax.plot(
                    pos_t[0], pos_t[1],
                    marker='X', color=color, linestyle='None',
                    markersize=marker_size, zorder=5,
                    label=None if position_labeled else position_label,
                    **marker_outline,
                )
                position_labeled = True

        if isinstance(obs, StaticObstacle):
            x_min, y_min = obs.bottom_left
            x_max, y_max = obs.top_right
            ax.add_patch(plt.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                color=color,
                label=None if static_labeled else "Static Obstacles",
                **plt_kwargs,
            ))
            static_labeled = True

    if created_ax:
        plt.tight_layout()

    return ax


def plot_path(path, ax=None, **plt_kwargs) -> plt.Axes:
    """Helper to plot a path on given axes.

    Args:
        ax: Matplotlib Axes to draw on.
        path (Iterable): The final path as an ordered sequence of
            nodes with ``x``/``y`` attributes.

    Returns:
        None
    """
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_ax = True

    # label = plt_kwargs.pop("label", "Final Path")
    ax.plot([n.x for n in path], [n.y for n in path], **plt_kwargs)

    # Add arrow to the middle of the line to indicate direction
    if len(path) >= 2:
        mid_index = len(path) // 2
        ax.annotate(
            '',
            xy=(path[mid_index].x, path[mid_index].y),
            xytext=(path[mid_index - 1].x, path[mid_index - 1].y),
            arrowprops=dict(arrowstyle='->', color=plt_kwargs.get('color', 'blue'))
        )

    if created_ax:
        plt.tight_layout()

    return ax

def plot_path_nodes_with_condition(path, key, value, ax=None, **plt_kwargs) -> plt.Axes:
    # If variable held by key is greater than value, plot that node with scatter
    """Draw path nodes satisfying an attribute condition."""
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_ax = True
    x_coords = [node.x for node in path if getattr(node, key, 0.0) > value]
    y_coords = [node.y for node in path if getattr(node, key, 0.0) > value]
    ax.scatter(x_coords, y_coords, **plt_kwargs)
    if created_ax:
        plt.tight_layout()
    return ax

def enable_latex(enable: bool = True):
    """Configure Matplotlib to render text with LaTeX, or switch it back off.

    LaTeX rendering looks nicer for paper figures but every draw call has to
    shell out to a LaTeX toolchain, which is slow while iterating. Call
    ``enable_latex(False)`` to go back to Matplotlib's normal (mathtext)
    text rendering without restarting the kernel.

    Args:
        enable: If True, render all text with LaTeX and a serif font.
            If False, restore Matplotlib's default text rendering.

    Returns:
        None
    """
    if enable:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 12,
            "font.size": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        })
    else:
        defaults = plt.rcParamsDefault
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": defaults["font.family"],
            "font.serif": defaults["font.serif"],
            "axes.labelsize": defaults["axes.labelsize"],
            "font.size": defaults["font.size"],
            "legend.fontsize": defaults["legend.fontsize"],
            "xtick.labelsize": defaults["xtick.labelsize"],
            "ytick.labelsize": defaults["ytick.labelsize"],
        })

def plot_vessel_on_path_at_t(path, time, speed, ax=None, **plt_kwargs) -> plt.Axes:

    """Mark vessel position at a supplied travel time and constant speed."""
    def find_position_at_time(path, time, speed):
        distance_traveled = 0.0
        for i in range(1, len(path)):
            segment_length = np.linalg.norm(
                np.array((path[i].x, path[i].y)) - np.array((path[i-1].x, path[i-1].y))
            )
            time_to_travel_segment = segment_length / speed
            if distance_traveled + time_to_travel_segment > time:
                # The vessel is on this segment at the given time
                remaining_time = time - distance_traveled
                ratio = remaining_time / time_to_travel_segment
                x = path[i-1].x + ratio * (path[i].x - path[i-1].x)
                y = path[i-1].y + ratio * (path[i].y - path[i-1].y)
                return (x, y)
            distance_traveled += time_to_travel_segment
        # If time exceeds the total travel time, return the last position
        return (path[-1].x, path[-1].y)

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_ax = True

    vessel_position = find_position_at_time(path, time, speed)
    ax.plot(vessel_position[0], vessel_position[1], **plt_kwargs)
    if created_ax:
        plt.tight_layout()
    return ax

# How the other vessels are drawn on the result panels: bright orange with a
# thick line, outlined in black so it stays visible over the orange/red part
# of the heatmap colormaps.
OBSTACLE_STYLE = dict(color="#ff8c00", alpha=1.0, linewidth=2.5, marker_size=10, outline_color="black")
"""Default appearance of other vessels on result panels."""


def _path_travel_time(path, speed) -> float:
    """Total travel time (s) along ``path`` at constant ``speed``, or 0 if unavailable."""
    if not path or not speed:
        return 0
    total_distance = sum(
        np.linalg.norm(np.array((path[i].x, path[i].y)) - np.array((path[i-1].x, path[i-1].y)))
        for i in range(1, len(path))
    )
    return total_distance / speed if total_distance > 0 else 0


def plot_panel(
        ax: plt.Axes,
        metric_fn,
        metric_kwargs: dict,
        nodes,
        path,
        cfg,
        graph: cartoblobpy.graph.Graph,
        obstacles: list[Obstacle] = (),
        t_computation: float = None,
        speed: float = None,
        title: str = None,
        legend: bool = True,
        legend_rows: int = None,
        legend_kwargs: dict = None,
        fontsize: float = None,
        axis_labels: bool = True,
        obstacle_style: dict = None,
    ) -> plt.Axes:
    """Render one metric (heatmap or node scatter) onto ``ax``, with the map,
    final path, vessel-at-t marker, and obstacle course overlaid.

    This is the shared building block behind :func:`plot_matrix` (a 3x3
    grid of these) and :func:`plot_paper_panels` (one figure per metric).

    Args:
        ax: Axes to draw on.
        metric_fn: One of ``plot_heatmap``, ``plot_nodes_with_values``,
            ``plot_heatmap_bbn_state``, ``plot_nodes_with_bbn_state``, etc.
            Called as ``metric_fn(nodes, cfg=cfg, ax=ax, **metric_kwargs)``.
        metric_kwargs: Keyword arguments for ``metric_fn``, e.g.
            ``dict(key='risk', add_cbar=True, cmap='viridis')`` or
            ``dict(bbn_node_id='Collision', state_name='Likely', add_cbar=True, cmap='Reds')``.
        nodes: RRT nodes to visualize.
        path: Final path, or falsy to omit the path/vessel overlay.
        cfg: Configuration object providing ``X_LIMITS``/``Y_LIMITS``.
        graph: Map graph, plotted as a grayscale background.
        obstacles: Static/dynamic obstacles to overlay.
        t_computation: Time at which planning finished; used to mark the
            vessel's and the obstacles' positions at that instant.
        speed: Vessel speed, needed to place the vessel-at-t marker.
        title: Axes title.
        legend: Whether to draw a legend. ``False`` also strips the legend
            that ``graph.plot()`` adds for its start/goal markers.
        legend_rows: If given, the legend is placed centered below the axes
            with its entries spread over this many rows. ``None`` keeps the
            in-axes ``loc='best'`` legend.
        legend_kwargs: Extra keyword arguments for ``ax.legend``, overriding
            the defaults above -- e.g. ``dict(bbox_to_anchor=(0.5, -0.15))``
            to push a below-axes legend further down.
        fontsize: Font size for the title, tick labels, colorbar, and legend.
            ``None`` keeps Matplotlib's defaults (legend at 9pt).
        axis_labels: ``False`` removes the x and y axis labels that
            ``graph.plot()`` sets (the tick labels stay).
        obstacle_style: Overrides for :data:`~risk_informed_planner.visualization.OBSTACLE_STYLE`, passed to
            :func:`plot_obstacles` -- e.g. ``dict(color="magenta")`` or
            ``dict(outline_color=None)`` to drop the outline.

    Returns:
        plt.Axes: ``ax``.
    """
    metric_fn(nodes, cfg=cfg, ax=ax, **metric_kwargs)

    if path:
        plot_path(path, ax=ax, color='black', linewidth=2, label='Final Path')
        if t_computation is not None and speed:
            plot_vessel_on_path_at_t(
                path, t_computation, speed=speed, ax=ax,
                marker="X", color='green', markersize=10, zorder=5,
                label=f'Vessel Position at t={t_computation:.1f}s',
            )

    plot_obstacles(
        obstacles, ax=ax,
        max_time=_path_travel_time(path, speed), t_computation=t_computation,
        **{**OBSTACLE_STYLE, **(obstacle_style or {})},
    )

    graph.plot(cmap='Greys', ax=ax)

    if title:
        ax.set_title(title)
    if not axis_labels:
        ax.set_xlabel("")
        ax.set_ylabel("")

    # graph.plot() draws its own start/goal legend, which picks up every
    # labeled artist already on the axes. Suppressing the legend therefore
    # means removing what it left behind, not just skipping our own call.
    existing_legend = ax.get_legend()
    if existing_legend is not None:
        existing_legend.remove()

    # before the legend, so a below-axes legend is placed under the resized labels
    if fontsize is not None:
        _set_axes_fontsize(ax, fontsize)

    if legend:
        _draw_legend(ax, legend_rows, legend_kwargs, fontsize)

    return ax


def _draw_legend(ax: plt.Axes, legend_rows: int = None, legend_kwargs: dict = None, fontsize: float = None,
                 first_labels=()):
    """Legend of every labeled artist on ``ax``: below the axes over
    ``legend_rows`` rows, or inside at ``loc='best'`` if ``legend_rows`` is
    ``None``. See :func:`plot_panel` for the arguments. Entries whose label
    starts with one of ``first_labels`` come first, in that order."""
    handles, labels = ax.get_legend_handles_labels()

    def rank(label):
        return next((i for i, prefix in enumerate(first_labels) if label.startswith(prefix)), len(first_labels))

    order = sorted(range(len(labels)), key=lambda i: rank(labels[i]))  # stable: keeps draw order within a rank
    handles, labels = [handles[i] for i in order], [labels[i] for i in order]

    kwargs = dict(fontsize=fontsize if fontsize is not None else 9)
    if legend_rows:
        kwargs.update(
            loc='upper center', ncol=max(1, -(-len(labels) // legend_rows)), frameon=False,
        )
    else:
        kwargs.update(loc='best')
    kwargs.update(legend_kwargs or {})
    ax.legend(handles, labels, **kwargs)
    if legend_rows and 'bbox_to_anchor' not in kwargs:
        place_legend_below(ax)


def _per_panel(option, name, metric, default=None):
    """Resolve a :func:`plot_paper_panels` option that is either one value for
    every panel or a dict keyed by panel name (``"risk_heatmap"``) or metric
    (``"risk"``); the panel name wins, and a missing key gives ``default``."""
    if isinstance(option, dict):
        return option.get(name, option.get(metric, default))
    return option


def place_legend_below(ax: plt.Axes, pad: float = 4.0):
    """Anchor the legend of ``ax`` just below its x tick labels and x label.

    The gap depends on the rendered label sizes, so call this again after
    anything that resizes the axes (``tight_layout``, a new ``figsize``).

    Args:
        pad: Gap between the x label and the legend, in points.
    """
    legend = ax.get_legend()
    if legend is None:
        return
    fig = ax.figure
    # the x label is only moved below the tick labels during a draw
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    xaxis_bottom = ax.xaxis.get_tightbbox(renderer).y0 - pad * fig.dpi / 72.0
    _, y = ax.transAxes.inverted().transform((0.0, xaxis_bottom))
    legend.set_bbox_to_anchor((0.5, y), transform=ax.transAxes)


def _set_axes_fontsize(ax: plt.Axes, fontsize: float):
    """Apply ``fontsize`` to the title, axis labels, and tick labels of ``ax``
    and of any colorbar attached to a mappable drawn on it."""
    axes = [ax]
    for mappable in [*ax.images, *ax.collections]:
        colorbar = getattr(mappable, 'colorbar', None)
        if colorbar is not None:
            axes.append(colorbar.ax)

    ax.title.set_fontsize(fontsize)
    for a in axes:
        a.tick_params(labelsize=fontsize)
        a.xaxis.label.set_fontsize(fontsize)
        a.yaxis.label.set_fontsize(fontsize)
        a.xaxis.get_offset_text().set_fontsize(fontsize)
        a.yaxis.get_offset_text().set_fontsize(fontsize)


def plot_matrix(
        nodes,
        path,
        cfg,
        graph: cartoblobpy.graph.Graph,
        t_computation: float = None,
        ax = None,
        color_map: str ='rainbow',
        speed: float = None,
        obstacles: list[Obstacle] = [],
    ) -> plt.Axes:
    """Combined 3x3 overview: heuristic, risk, and cost (heatmap + node
    values), BBN collision/grounding probability heatmaps, and a combined
    collision-vs-grounding panel. Handy while iterating; for paper-ready,
    one-figure-per-plot output (risk/collision/grounding, heatmap and node
    values) use :func:`plot_paper_panels` instead.
    """
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(3, 3, figsize=(22, 16))
        created_ax = True

    nds = nodes[1:]  # skip the root node for visualization since it may have uninitialized values

    panel = lambda a, fn, kwargs, title: plot_panel(
        a, fn, kwargs, nds, path, cfg, graph, obstacles, t_computation, speed, title=title,
    )

    # heuristic
    panel(ax[0, 0], plot_heatmap, dict(key='heuristic', add_cbar=True, cmap=color_map), "Heuristic map")
    panel(ax[1, 0], plot_nodes_with_values, dict(key='heuristic', add_cbar=True, cmap=color_map), "Heuristic Values at Nodes")

    # risk
    panel(ax[0, 1], plot_heatmap, dict(key='risk', add_cbar=True, cmap=color_map), "Risk map")
    panel(ax[1, 1], plot_nodes_with_values, dict(key='risk', add_cbar=True, cmap=color_map), "Risk Values at Nodes")

    # cost
    panel(ax[0, 2], plot_heatmap, dict(key='cost', add_cbar=True, cmap=color_map), "Cost map")
    panel(ax[1, 2], plot_nodes_with_values, dict(key='cost', add_cbar=True, cmap=color_map), "Cost Values at Nodes")

    # encounter properties from bbn
    panel(ax[2, 0], plot_heatmap_bbn_state, dict(bbn_node_id="Collision", state_name="Likely", add_cbar=True, cmap="Reds"), "BBN Collision Probability")
    panel(ax[2, 1], plot_heatmap_bbn_state, dict(bbn_node_id="Grounding", state_name="Likely", add_cbar=True, cmap="Blues"), "BBN Grounding Probability")

    # combined collision + grounding overlay
    plot_heatmap_bbn_state(nds, cfg=cfg, bbn_node_id="Collision", state_name="Likely", ax=ax[2, 2], cmap='Reds', min_probability=0.01, max_probability=1.0, alpha=0.7)
    plot_heatmap_bbn_state(nds, cfg=cfg, bbn_node_id="Grounding", state_name="Likely", ax=ax[2, 2], cmap='Blues', min_probability=0.01, max_probability=1.0, alpha=0.7)
    plot_obstacles(obstacles, ax=ax[2, 2], max_time=_path_travel_time(path, speed), t_computation=t_computation, **OBSTACLE_STYLE)
    if path:
        plot_path(path, ax=ax[2, 2], color='black', linewidth=2, label='Final Path')
        if t_computation is not None and speed:
            plot_vessel_on_path_at_t(path, t_computation, speed=speed, ax=ax[2, 2], marker="X", color='green', markersize=10, zorder=5, label=f'Vessel Position at t={t_computation:.1f}s')
    ax[2, 2].set_title("BBN Collision vs Grounding")
    ax[2, 2].legend(loc='best', fontsize=9)

    plt.gcf().tight_layout()

    return ax


def plot_paper_panels(
        nodes,
        path,
        cfg,
        graph: cartoblobpy.graph.Graph,
        obstacles: list[Obstacle] = (),
        t_computation: float = None,
        speed: float = None,
        metrics=("risk", "collision", "grounding"),
        modes=("heatmap", "nodes"),
        panels: list[tuple[str, str]] = None,
        vmin=None,
        vmax=None,
        color_map: str = "rainbow",
        figsize=(7, 6),
        latex: bool = None,
        save_dir: str = None,
        save_prefix: str = "",
        save_ext: str = "pdf",
        dpi: int = 300,
        legend=True,
        legend_rows: int = 3,
        legend_kwargs: dict = None,
        fontsize=None,
        title=True,
        axis_labels=True,
        obstacle_style: dict = None,
        show_target_vessel=True,
        heatmap_kwargs: dict = None,
    ) -> dict[str, plt.Axes]:
    """Render risk, collision probability, and grounding probability as
    separate, paper-ready figures (a heatmap and a node-values variant for
    each), instead of the combined grid produced by :func:`plot_matrix`.

    ``figsize``, ``legend``, ``fontsize``, ``title``, ``axis_labels``, and
    ``show_target_vessel`` accept either a single value applied to every panel, or a dict keyed by
    panel name (``"risk_heatmap"``) or by metric (``"risk"``) for per-panel
    values; the panel name wins over the metric, and a panel missing from
    the dict gets the default. E.g. one large risk map and two small hazard
    maps::

        figsize={"risk": (8, 7), "collision": (4, 3.5), "grounding": (4, 3.5)},
        fontsize={"risk": 14, "collision": 11, "grounding": 11},
        legend={"collision": False, "grounding": False},
        axis_labels={"collision": False, "grounding": False},

    Args:
        nodes, path, cfg, graph, obstacles, t_computation, speed: Same as
            :func:`plot_matrix`.
        metrics: Subset of ``("risk", "collision", "grounding")`` to plot.
        modes: Subset of ``("heatmap", "nodes")`` to plot for each metric.
            Together with ``metrics`` this draws every ``(metric, mode)``
            combination -- e.g. ``metrics=("risk",), modes=("heatmap",
            "nodes")`` draws both risk panels. Ignored if ``panels`` is given.
        panels: Explicit list of ``(metric, mode)`` pairs to draw, in order,
            instead of the ``metrics`` x ``modes`` cross-product. Use this
            when you don't want every combination -- e.g.
            ``panels=[("risk", "heatmap"), ("collision", "heatmap")]`` draws
            exactly those two and skips the rest.
        vmin, vmax: Fixed color-scale bounds, forwarded to the underlying
            ``plot_heatmap``/``plot_heatmap_bbn_state``/``plot_nodes_with_*``
            call for every panel. Without these, each panel's color scale is
            fit to its own data range, which makes panels from different
            runs hard to compare. Each can be either a single number applied
            to every metric, or a ``{metric: value}`` dict for per-metric
            bounds (e.g. ``vmin={"risk": 0.0}, vmax={"risk": 1.0,
            "collision": 1.0, "grounding": 1.0}``); a metric missing from
            the dict falls back to the default (auto-scaled) behavior.
        color_map: Colormap for the risk panels. Collision/grounding panels
            use their own fixed "rainbow" colormap.
        figsize: Figure size for each individual figure (per panel, see above).
        latex: If given, calls ``enable_latex(latex)`` before plotting, so
            LaTeX rendering can be toggled from a single call site.
        save_dir: If given, each figure is saved to
            ``{save_dir}/{save_prefix}{metric}_{mode}.{save_ext}`` right
            after it is drawn. If you add annotations afterwards (e.g.
            :func:`annotate_hazard_drivers`), leave this ``None`` and call
            :func:`save_paper_panels` once you're done instead.
        save_prefix: Filename prefix used when ``save_dir`` is given.
        save_ext: File extension used when ``save_dir`` is given.
        dpi: DPI used when saving.
        legend: Whether to draw a legend (per panel, see above). Pass
            ``False`` for animation frames.
        legend_rows: Number of rows of the legend, which is placed centered
            below the plot. ``None`` puts it inside the axes at
            ``loc='best'`` instead -- that re-picks a corner per frame, so
            avoid it for animations.
        legend_kwargs: Extra ``ax.legend`` keyword arguments applied to every
            panel, e.g. ``dict(bbox_to_anchor=(0.5, -0.12))`` to move the
            below-plot legend down if it overlaps the tick labels.
        fontsize: Font size of the title, ticks, colorbar, and legend (per
            panel, see above). ``None`` keeps Matplotlib's defaults.
        title: ``True`` for the default title, ``False`` for none, or a
            string to use instead (per panel, see above).
        axis_labels: ``False`` removes the x/y axis labels, keeping the
            tick labels (per panel, see above).
        obstacle_style: Overrides for how other vessels are drawn
            (:data:`~risk_informed_planner.visualization.OBSTACLE_STYLE`: orange, thick, black outline), e.g.
            ``dict(color="magenta", linewidth=3)``.
        show_target_vessel: ``False`` hides the dynamic obstacles (the
            target vessel's course and position markers) while keeping the
            static obstacles (per panel, see above) -- e.g.
            ``show_target_vessel={"grounding": False}``.
        heatmap_kwargs: ``{metric: dict}`` of extra keyword arguments for
            that metric's heatmap function (``plot_heatmap`` /
            ``plot_heatmap_bbn_state``), e.g. ``{"risk": dict(max_value=0.3)}``.
            Not applied to the ``"nodes"`` panels.

    Returns:
        dict[str, plt.Axes]: Maps ``"{metric}_{mode}"`` (e.g.
        ``"risk_heatmap"``, ``"collision_nodes"``) to the axes of the
        corresponding figure.
    """
    if latex is not None:
        enable_latex(latex)

    nds = nodes[1:] if nodes else nodes

    metric_specs = {
        "risk": dict(
            heatmap_fn=plot_heatmap, nodes_fn=plot_nodes_with_values,
            fn_kwargs=dict(key="risk"), cmap=color_map,
            heatmap_title="Risk map", nodes_title="Risk Values at Nodes",
        ),
        "collision": dict(
            heatmap_fn=plot_heatmap_bbn_state, nodes_fn=plot_nodes_with_bbn_state,
            fn_kwargs=dict(bbn_node_id="Collision", state_name="Likely"), cmap="rainbow",
            heatmap_title="Collision Probability map",
            nodes_title="Collision Probability at Nodes",
        ),
        "grounding": dict(
            heatmap_fn=plot_heatmap_bbn_state, nodes_fn=plot_nodes_with_bbn_state,
            fn_kwargs=dict(bbn_node_id="Grounding", state_name="Likely"), cmap="rainbow",
            heatmap_title="Grounding Probability map",
            nodes_title="Grounding Probability at Nodes",
        ),
    }

    def resolve_bound(bound, metric):
        if isinstance(bound, dict):
            return bound.get(metric)
        return bound

    per_panel = _per_panel

    # a bare string, e.g. modes=("heatmap"), would otherwise be iterated per character
    if isinstance(metrics, str):
        metrics = (metrics,)
    if isinstance(modes, str):
        modes = (modes,)

    panel_list = list(panels) if panels is not None else [
        (metric, mode) for metric in metrics for mode in modes
    ]

    results = {}
    for metric, mode in panel_list:
        spec = metric_specs[metric]
        name = f"{metric}_{mode}"

        fig, panel_ax = plt.subplots(figsize=per_panel(figsize, name, metric, (7, 6)))
        metric_fn = spec["heatmap_fn"] if mode == "heatmap" else spec["nodes_fn"]

        panel_title = per_panel(title, name, metric, True)
        if panel_title is True:
            panel_title = spec["heatmap_title"] if mode == "heatmap" else spec["nodes_title"]

        fn_kwargs = dict(**spec["fn_kwargs"], add_cbar=True, cmap=spec["cmap"])
        metric_vmin = resolve_bound(vmin, metric)
        metric_vmax = resolve_bound(vmax, metric)
        if metric_vmin is not None:
            fn_kwargs["vmin"] = metric_vmin
        if metric_vmax is not None:
            fn_kwargs["vmax"] = metric_vmax
        if mode == "heatmap":
            fn_kwargs.update((heatmap_kwargs or {}).get(metric, {}))

        panel_obstacles = obstacles
        if not per_panel(show_target_vessel, name, metric, True):
            panel_obstacles = [obs for obs in obstacles if not isinstance(obs, LinearDynamicObstacle)]

        plot_panel(
            panel_ax, metric_fn,
            fn_kwargs,
            nds, path, cfg, graph, panel_obstacles, t_computation, speed,
            title=panel_title or None,
            legend=per_panel(legend, name, metric, True),
            legend_rows=legend_rows,
            legend_kwargs=legend_kwargs,
            fontsize=per_panel(fontsize, name, metric),
            axis_labels=per_panel(axis_labels, name, metric, True),
            obstacle_style=obstacle_style,
        )
        fig.tight_layout()
        if legend_rows and 'bbox_to_anchor' not in (legend_kwargs or {}):
            place_legend_below(panel_ax)

        results[name] = panel_ax

    if save_dir is not None:
        save_paper_panels(results, save_dir, save_prefix=save_prefix, save_ext=save_ext, dpi=dpi)

    return results


def plot_paper_panels_thresholded(
        nodes,
        path,
        cfg,
        graph: cartoblobpy.graph.Graph,
        threshold,
        keep: str = "below",
        metrics=("risk", "collision", "grounding"),
        vmin=None,
        vmax=None,
        **kwargs,
    ) -> dict[str, plt.Axes]:
    """Like :func:`plot_paper_panels` with ``modes=("heatmap",)``, but each
    heatmap is drawn only up to ``threshold``: cells past it are transparent,
    so the chart (and the path/obstacles) shows through.

    Args:
        nodes, path, cfg, graph: Same as :func:`plot_paper_panels`.
        threshold: Cut-off value, either one number for every metric or a
            ``{metric: value}`` dict (e.g. ``{"risk": 0.3, "collision": 0.1}``);
            a metric missing from the dict is drawn unmasked.
        keep: ``"below"`` keeps values ``<= threshold`` (the heatmap runs
            from the minimum up to the threshold); ``"above"`` keeps values
            ``>= threshold`` instead.
        metrics: Subset of ``("risk", "collision", "grounding")`` to plot.
        vmin, vmax: Color-scale bounds, as in :func:`plot_paper_panels`. By
            default the scale ends at the threshold (``vmax`` for
            ``keep="below"``, ``vmin`` for ``keep="above"``), so the full
            colormap spans the visible range.
        **kwargs: Forwarded to :func:`plot_paper_panels` (``figsize``,
            ``fontsize``, ``legend``, ``save_dir``, ...). Files are saved as
            ``{save_prefix}{metric}_heatmap.{save_ext}``, so pass a distinct
            ``save_prefix`` to avoid overwriting the unthresholded figures.

    Returns:
        dict[str, plt.Axes]: Same as :func:`plot_paper_panels`.
    """
    if keep not in ("below", "above"):
        raise ValueError(f"keep must be 'below' or 'above', got {keep!r}")
    if isinstance(metrics, str):
        metrics = (metrics,)

    def per_metric(option):
        return {m: option.get(m) for m in metrics} if isinstance(option, dict) else dict.fromkeys(metrics, option)

    thresholds = per_metric(threshold)
    vmins, vmaxs = per_metric(vmin), per_metric(vmax)

    heatmap_kwargs = {}
    for metric, value in thresholds.items():
        if value is None:
            continue
        # plot_heatmap masks by value, plot_heatmap_bbn_state by probability
        lo, hi = ("min_value", "max_value") if metric == "risk" else ("min_probability", "max_probability")
        if keep == "below":
            heatmap_kwargs[metric] = {hi: value}
            if vmaxs[metric] is None:
                vmaxs[metric] = value
        else:
            heatmap_kwargs[metric] = {lo: value}
            if vmins[metric] is None:
                vmins[metric] = value

    return plot_paper_panels(
        nodes, path, cfg, graph,
        metrics=metrics, modes=("heatmap",),
        vmin=vmins, vmax=vmaxs,
        heatmap_kwargs=heatmap_kwargs,
        **kwargs,
    )


def save_paper_panels(
        panels: dict[str, plt.Axes],
        save_dir: str,
        save_prefix: str = "",
        save_ext: str = "pdf",
        dpi: int = 300,
    ):
    """Save each figure returned by :func:`plot_paper_panels` to
    ``{save_dir}/{save_prefix}{name}.{save_ext}``. Call this after adding
    annotations so they end up in the saved files."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for name, panel_ax in panels.items():
        panel_ax.figure.savefig(
            Path(save_dir) / f"{save_prefix}{name}.{save_ext}",
            dpi=dpi, bbox_inches="tight", pad_inches=0.01,
        )


# Line styles of the successive paths in :func:`plot_campaign_overlay`, in
# step order (cycled if more steps are shown). All in one color, so they stay
# readable on top of a colored heatmap and in grayscale print.
CAMPAIGN_PATH_STYLES = (
    dict(linestyle=":", linewidth=2.2),
    dict(linestyle="--", linewidth=2.2),
    dict(linestyle="-", linewidth=2.2),
    dict(linestyle="-.", linewidth=2.2),
)
"""Line styles cycled across paths in a replanning campaign."""


def _draw_step_labels(ax, marks, distance, spacing, side, fontsize):
    """Tag one vessel's position markers ``t_k`` in a row beside its track.

    ``marks`` is ``[(k, xy, heading)]`` in step order. The direction of travel
    is taken from the first to the last marker, or from the headings when the
    markers (nearly) coincide. Tag ``i`` is offset ``distance`` points
    sideways and ``(i - (n-1)/2) * spacing`` points along that direction from
    its own marker; since later markers lie further along the track, tags
    are always at least ``spacing`` apart. Assumes an equal-aspect axes, so
    directions in data and screen space agree.
    """
    xy = np.array([m[1] for m in marks], dtype=float)
    d = xy[-1] - xy[0]
    if np.linalg.norm(d) < 1e-6:
        headings = [np.asarray(m[2], dtype=float) for m in marks if m[2] is not None]
        d = np.mean(headings, axis=0) if headings else np.array([1.0, 0.0])
    norm = np.linalg.norm(d)
    d = d / norm if norm > 1e-12 else np.array([1.0, 0.0])
    normal = np.array([-d[1], d[0]]) if side == "left" else np.array([d[1], -d[0]])

    n = len(marks)
    for i, (k, p, _) in enumerate(marks):
        offset = normal * distance + d * (i - (n - 1) / 2) * spacing
        ax.annotate(
            rf"$t_{{{k}}}$", tuple(p), xytext=tuple(offset), textcoords="offset points",
            ha="center", va="center", fontsize=fontsize, zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
            arrowprops=dict(arrowstyle="-", color="0.25", linewidth=0.7, shrinkA=0, shrinkB=4),
        )


def plot_campaign_overlay(
        runs,
        steps=(0, 1, 2),
        background_step: int = None,
        metrics=("risk",),
        legend=True,
        legend_rows: int = 3,
        legend_kwargs: dict = None,
        path_color: str = "black",
        path_styles=None,
        vessel_color: str = "tab:green",
        step_labels: bool = True,
        label_distance: float = 14.0,
        label_spacing: float = 12.0,
        label_side: str = "left",
        **panel_kwargs,
    ) -> dict[str, plt.Axes]:
    """Show how a receding-horizon campaign advances: several time steps
    drawn on top of each other over one step's heatmap.

    For every step in ``steps`` it draws the path planned at that step, the
    own vessel's position when that plan started, and the other vessels'
    positions at that moment; markers are tagged ``t_k`` with the step index.
    The other vessels' course is drawn once, from the first shown step.

    Args:
        runs: The campaign as a list of :class:`~risk_informed_planner.experiment.PlannerRun`,
            one per step in order, e.g. from ``PlannerRun.load_many``. Step
            ``k`` starts at the sum of the ``t_computation`` of steps
            ``0..k-1``; that time is shown in the path's legend entry.
        steps: Indices into ``runs`` to overlay.
        background_step: Step whose heatmap is drawn; defaults to the first
            of ``steps``.
        metrics: Heatmaps to draw, one figure each (``"risk"``,
            ``"collision"``, ``"grounding"``); every one gets the overlay.
        legend, legend_rows, legend_kwargs: As in :func:`plot_paper_panels`
            (``legend`` can be per panel).
        path_color: Color of all paths; steps are told apart by line style.
        path_styles: Per-step ``ax.plot`` kwargs for the paths, in step
            order; defaults to :data:`~risk_informed_planner.visualization.CAMPAIGN_PATH_STYLES`.
        vessel_color: Fill color of the own vessel's position markers.
        step_labels: Tag each position marker with ``t_k``. The tags of one
            vessel are laid out in a row beside its track, ``label_spacing``
            apart along the direction of travel, each tied to its marker by
            a thin leader line -- so they never stack, even when the
            markers coincide.
        label_distance: Distance of that row from the track, in points.
        label_spacing: Spacing between consecutive tags, in points.
        label_side: ``"left"`` or ``"right"`` of the direction of travel.
        **panel_kwargs: Forwarded to :func:`plot_paper_panels` for the
            background -- ``vmin``, ``vmax``, ``figsize``, ``fontsize``,
            ``title``, ``axis_labels``, ``color_map``, ``latex``,
            ``obstacle_style``. Save with :func:`save_paper_panels`
            afterwards; ``save_dir`` here would save before the overlay is
            drawn.

    Returns:
        dict[str, plt.Axes]: Same as :func:`plot_paper_panels`, so
        :func:`annotate_hazard_drivers` and :func:`save_paper_panels` work on
        it too.
    """
    if "save_dir" in panel_kwargs:
        raise TypeError("pass save_dir to save_paper_panels() after plotting, not here")

    steps = list(steps)
    background_step = steps[0] if background_step is None else background_step
    path_styles = list(path_styles or CAMPAIGN_PATH_STYLES)
    obstacle_style = {**OBSTACLE_STYLE, **(panel_kwargs.get("obstacle_style") or {})}
    marker_size = obstacle_style.get("marker_size", 10)
    # campaign time at which each step's plan starts
    t_starts = np.concatenate([[0.0], np.cumsum([run.t_computation for run in runs])])

    background = runs[background_step]
    panels = plot_paper_panels(
        nodes=background.nodes, path=None, cfg=background.cfg, graph=background.graph,
        obstacles=[], t_computation=None, speed=background.speed,
        metrics=metrics, modes=("heatmap",), legend=False,
        **panel_kwargs,
    )

    if label_side not in ("left", "right"):
        raise ValueError(f"label_side must be 'left' or 'right', not {label_side!r}")

    first = runs[steps[0]]
    for name, ax in panels.items():
        metric = name.rsplit("_", 1)[0]
        fontsize = _per_panel(panel_kwargs.get("fontsize"), name, metric)

        plot_obstacles(
            first.obstacles, ax=ax, markers=False,
            max_time=_path_travel_time(first.path, first.speed),
            **obstacle_style,
        )

        # marker positions per vessel, for the step labels: key -> [(k, xy, heading)]
        tracks = {}
        for i, k in enumerate(steps):
            run = runs[k]
            own_dir = None
            if run.path:
                plot_path(
                    run.path, ax=ax, color=path_color,
                    label=rf"Path planned at $t_{{{k}}}$ = {t_starts[k]:.0f} s",
                    **path_styles[i % len(path_styles)],
                )
                own_xy = (run.path[0].x, run.path[0].y)
                if len(run.path) > 1:
                    own_dir = (run.path[1].x - own_xy[0], run.path[1].y - own_xy[1])
            else:
                own_xy = run.metadata.get("start")

            if own_xy is not None:
                ax.plot(
                    *own_xy, marker="X", markersize=marker_size, color=vessel_color,
                    markeredgecolor="black", markeredgewidth=1.2, linestyle="none", zorder=6,
                    label="Vessel Position" if i == 0 else None,
                )
                tracks.setdefault("own", []).append((k, own_xy, own_dir))

            for j, obs in enumerate(run.obstacles):
                if not isinstance(obs, LinearDynamicObstacle):
                    continue
                obs_xy = tuple(obs.position(t=0)[:2])
                ax.plot(
                    *obs_xy, marker="X", markersize=marker_size, color=obstacle_style["color"],
                    markeredgecolor="black", markeredgewidth=1.2, linestyle="none", zorder=6,
                    label="Target Position" if i == 0 and j == 0 else None,
                )
                obs_dir = (np.cos(obs.yaw), np.sin(obs.yaw))
                tracks.setdefault(("obstacle", j), []).append((k, obs_xy, obs_dir))

        if step_labels:
            for marks in tracks.values():
                _draw_step_labels(ax, marks, label_distance, label_spacing, label_side, fontsize)

        if _per_panel(legend, name, metric, True):
            _draw_legend(
                ax, legend_rows, legend_kwargs, fontsize,
                first_labels=("Path planned", "Vessel Position", "Obstacle Position", "Obstacle Course"),
            )
            ax.figure.tight_layout()
            if legend_rows and "bbox_to_anchor" not in (legend_kwargs or {}):
                place_legend_below(ax)

    return panels


def sample_heatmap(ax: plt.Axes, x: float, y: float) -> float:
    """Value of the heatmap drawn on ``ax`` at data coordinates ``(x, y)``,
    read from the displayed (interpolated and smoothed) pixel grid.

    Expects the heatmap to be the first image on ``ax``, as drawn by
    :func:`plot_heatmap` / :func:`plot_heatmap_bbn_state` through
    :func:`plot_panel`. Returns ``nan`` for masked pixels.
    """
    if not ax.images:
        raise ValueError("axes has no heatmap image to sample from")

    image = ax.images[0]
    grid = image.get_array()
    x0, x1, y0, y1 = image.get_extent()
    n_rows, n_cols = grid.shape

    # The grid is built with np.mgrid endpoints included, so pixel i sits at
    # x0 + i * (x1 - x0) / (n - 1); rows run along y because of origin='lower'.
    col = round((x - x0) / (x1 - x0) * (n_cols - 1))
    row = round((y - y0) / (y1 - y0) * (n_rows - 1))
    if not (0 <= col < n_cols and 0 <= row < n_rows):
        raise ValueError(f"point ({x}, {y}) lies outside the heatmap extent {image.get_extent()}")

    value = grid[row, col]
    return float("nan") if np.ma.is_masked(value) else float(value)


HAZARD_DRIVER_SOURCES = {
    "P(Collision)": ("collision_heatmap", "k"),
    "P(Grounding)": ("grounding_heatmap", "k"),
}
"""Default BBN heatmap sources used by hazard-driver annotations."""


# Drawing order of the hazard-driver annotations, above everything the map
# panels draw (path, vessel, obstacle markers): arrow < backdrop < inset <
# point markers.
_ZORDER_ARROW = 20
_ZORDER_BACKDROP = 21
_ZORDER_INSET = 22
_ZORDER_MARKER = 23


class _InsetBackdrop(Rectangle):
    """Opaque box behind an inset axes that covers everything the inset
    draws -- title, tick labels, and bar labels, not just the axes area.

    The size is taken from the inset's rendered extent at every draw, so it
    stays fitted when fonts or the figure size change.
    """

    def __init__(self, inset: plt.Axes, pad: float = 3.0, **kwargs):
        super().__init__((0, 0), 1, 1, transform=IdentityTransform(), **kwargs)
        self._inset = inset
        self._pad = pad  # points

    def draw(self, renderer):
        bbox = self._inset.get_tightbbox(renderer)
        pad = renderer.points_to_pixels(self._pad)
        self.set_bounds(bbox.x0 - pad, bbox.y0 - pad, bbox.width + 2 * pad, bbox.height + 2 * pad)
        super().draw(renderer)


def annotate_hazard_drivers(
        panels: dict[str, plt.Axes],
        points: list[dict],
        target: str = "risk_heatmap",
        sources: dict[str, tuple[str, str]] = None,
        inset_size: tuple[float, float] = (250.0, 110.0),
        value_lim: tuple[float, float] = (0.0, 1.35),
        show_value_axis: bool = False,
        fontsize: float = 10,
        mark_sources: bool = True,
        marker_kwargs: dict = None,
        arrow_kwargs: dict = None,
        box_kwargs: dict = None,
    ) -> list[dict]:
    """Explain points of the risk heatmap with a small horizontal bar chart
    of the hazard probabilities behind them.

    For each point, the values at that location are read from the pixels of
    the source heatmaps (collision "C" and grounding "G" by default), and
    drawn as a bar-chart inset on the ``target`` panel, with an arrow from
    the inset to the point.

    Call it in the same notebook cell as :func:`plot_paper_panels`: inline
    figures are displayed at the end of the cell that created them, so insets
    added in a later cell don't show, and re-running a separate cell would
    stack a second set of insets onto the same figures.

    Args:
        panels: Output of :func:`plot_paper_panels`; must contain ``target``
            and every source panel.
        points: One dict per point of interest, all coordinates in the map's
            data units (the same numbers as on the axes ticks):

            - ``xy``: ``(x, y)`` of the point of interest.
            - ``inset_xy``: ``(x, y)`` of the center of the bar chart.
            - ``label`` (optional): short tag such as ``"A"``, written next
              to the marker(s) and as the inset title.
            - ``inset_size`` (optional): per-point ``(width, height)``
              override.
        target: Name of the panel that receives the insets.
        sources: ``{bar_label: (panel_name, color)}``; defaults to
            :data:`~risk_informed_planner.visualization.HAZARD_DRIVER_SOURCES`. Add e.g.
            ``"R": ("risk_heatmap", "tab:gray")`` to show the risk too.
        inset_size: Default ``(width, height)`` of each inset, in data units.
        value_lim: Limits of the bars' value axis. The default leaves room
            past 1.0 for the value labels printed at the bar ends.
        show_value_axis: Draw the value axis (ticks under the bars). Off by
            default -- the values are printed at the bar ends.
        fontsize: Font size inside the insets and for point labels.
        mark_sources: Also mark (and label) the point on the source panels,
            so the reader can find it on the hazard maps.
        marker_kwargs: Overrides for the point marker (``ax.plot`` kwargs).
        arrow_kwargs: Overrides for the arrow (``arrowprops`` of
            ``ax.annotate``).
        box_kwargs: Overrides for the opaque box behind each inset, which
            covers its title and labels too -- ``Rectangle`` kwargs plus
            ``pad`` (margin around the contents, in points). E.g.
            ``dict(edgecolor="none")`` to drop the border.

    Returns:
        list[dict]: Per point, ``{"label", "xy", "values", "inset_ax"}``,
        where ``values`` maps each bar label to the sampled value.
    """
    sources = sources or HAZARD_DRIVER_SOURCES
    missing = [name for name in (target, *(p for p, _ in sources.values())) if name not in panels]
    if missing:
        raise KeyError(f"panels {missing} not found; plot them with plot_paper_panels(panels=[...]) first")

    ax = panels[target]
    marker_style = dict(marker="o", markersize=8, markerfacecolor="white",
                        markeredgecolor="black", markeredgewidth=1.5, linestyle="none", zorder=_ZORDER_MARKER)
    marker_style.update(marker_kwargs or {})
    arrow_style = dict(arrowstyle="-|>", color="gray", lw=1.2, shrinkA=0, shrinkB=6)
    arrow_style.update(arrow_kwargs or {})
    box_style = dict(facecolor="white", edgecolor="0.5", linewidth=0.8, zorder=_ZORDER_BACKDROP, pad=3.0)
    box_style.update(box_kwargs or {})

    def mark(on_ax, xy, label):
        on_ax.plot(*xy, **marker_style)
        if label:
            on_ax.annotate(
                label, xy, xytext=(6, 6), textcoords="offset points",
                fontsize=fontsize, fontweight="bold", zorder=_ZORDER_MARKER,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
            )

    results = []
    for point in points:
        xy = tuple(point["xy"])
        cx, cy = point["inset_xy"]
        width, height = point.get("inset_size", inset_size)
        label = point.get("label")

        values = {bar: sample_heatmap(panels[panel_name], *xy) for bar, (panel_name, _) in sources.items()}

        # The arrow starts at the inset center; the backdrop and inset are drawn
        # on top of it, so only the part outside the inset shows.
        ax.annotate("", xy=xy, xytext=(cx, cy), arrowprops=arrow_style, zorder=_ZORDER_ARROW)
        mark(ax, xy, label)
        if mark_sources:
            for panel_name, _ in sources.values():
                if panel_name != target:
                    mark(panels[panel_name], xy, label)

        inset = ax.inset_axes([cx - width / 2, cy - height / 2, width, height], transform=ax.transData,
                              zorder=_ZORDER_INSET)
        ax.add_artist(_InsetBackdrop(inset, **box_style))
        # numeric positions + tick labels: the inset shares the map's numeric
        # units, so categorical (string) values are rejected
        positions = np.arange(len(values))
        bars = inset.barh(
            positions, [0.0 if np.isnan(v) else v for v in values.values()],
            color=[color for _, color in sources.values()], height=0.7,
        )
        inset.set_yticks(positions, list(values))
        inset.invert_yaxis()  # first source on top
        inset.bar_label(bars, labels=[f"{v:.2f}" for v in values.values()], fontsize=fontsize, padding=2)
        inset.set_xlim(*value_lim)
        inset.tick_params(labelsize=fontsize, length=0, pad=2)
        inset.spines[["top", "right"]].set_visible(False)
        if not show_value_axis:
            inset.spines["bottom"].set_visible(False)
            inset.set_xticks([])
        if label:
            inset.set_title(label, fontsize=fontsize, fontweight="bold", pad=2)

        results.append(dict(label=label, xy=xy, values=values, inset_ax=inset))

    return results