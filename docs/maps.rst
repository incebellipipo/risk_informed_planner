Maps and coordinates
====================

Available datasets
------------------

All map inputs live under ``notebooks/maps/``. Each dataset contains
``map.yaml``, ``base.png``, ``depth.png``, and ``traffic.png``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Directory
     - Scenario
   * - ``empty``
     - Open-water baseline.
   * - ``kristiansund``
     - Kristiansund overview.
   * - ``kristiansund_bridge``
     - Main bridge-planning study.
   * - ``kristiansund_port_entry``
     - Port-entry map.
   * - ``kristiansund_small_port_entry``
     - Smaller port-entry study.
   * - ``trondheim``
     - Trondheim fjord studies and parameter sweeps.

Loading a map
-------------

.. code-block:: python

   from cartoblobpy.graph import Graph

   graph = Graph()
   graph.load_from_yaml("notebooks/maps/kristiansund_bridge/map.yaml")
   graph.start = (200.0, 800.0)
   graph.goal = (1500.0, 600.0)

The YAML references image files relative to its own location. Keep the map files
together. The notebook paths assume that the kernel starts in the notebook's
directory. The Python example above assumes the repository root.

Layers and units
----------------

The base image supplies occupancy. The depth layer supplies depth values mapped
from image colors, and the traffic layer supplies normalized intensity values.
YAML configuration also specifies resolution, origin, image interpretation,
and occupancy thresholds.

Use graph world coordinates for nodes and vessel poses, not image row/column
indices. The studies use metres for positions, depths, radii, and sample spacing;
metres per second for vessel speeds; seconds for time; and radians for heading.
Use ``graph.world_to_grid`` and ``graph.grid_to_world`` when converting between
those representations.

Collision-check modes
---------------------

``FreeCheckMethod.OCCUPANCY`` uses occupancy for tree sampling and path checks.
``FreeCheckMethod.DEPTH`` samples cells deeper than ``min_depth``. Historical
path checks use a separate, fixed 10 m threshold and reject depths below 10 m.
``FreeCheckMethod.DEPTH_SAMPLING_OCCUPANCY_CHECKS`` samples cells deeper than
``min_depth`` and checks edges against occupancy. The original studies obtained
this mixed behavior by passing an enum from another planner class; the retained
notebooks select it explicitly to preserve their results.

``DynamicRisk.grounding_depth_hard_limit`` is a separate risk-evidence setting.
For reproducibility, its lookahead retains the original graph predicate:
any sampled depth greater than the hard limit produces grounding-course YES
evidence. It separately forces grounding-recovery failure evidence when the
node itself is shallower than the hard limit. This historical behavior is
preserved even though the lookahead predicate is counterintuitive. Occupancy
mode still uses depth when computing risk.

Changing map inputs
-------------------

Changing layer values, resolution, origin, or start/goal positions changes the
experiment. Regenerate planning results and dependent figures after changing
those inputs. Saved runs with a map reference reload the current YAML and image
files, so preserve the associated map version when comparing experiments.
