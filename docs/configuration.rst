Configuration
=============

``Config`` contains tree-growth and edge-cost settings. Map geometry, vessel
speed, obstacle definitions, and evidence thresholds are supplied separately
to the planner and risk model.

Planner settings
----------------

The package defaults and field descriptions are generated from
:class:`~risk_informed_planner.config.Config` in :doc:`api/risk_informed_planner/config/index`. Edit the
attribute docstrings beside the dataclass fields to change their reference text.
The main notebook overrides these defaults; see :doc:`quickstart`.

Choose ALPHA in the usual range [0, 1] for an interpretable distance/risk
tradeoff. At ALPHA = 1 the risk-exposure term disappears, while risk cutoffs
still apply. GAMMA controls a separate additive turn penalty.

Stopping targets are not hard iteration or time bounds. See
:doc:`quickstart` before running searches with difficult goal locations.

Risk-model settings
-------------------

Set these on ``DynamicRisk``. Values below are the bridge notebook's choices,
not the constructor defaults:

.. list-table::
   :header-rows: 1
   :widths: 45 15 40

   * - Parameter
     - Study value
     - Meaning
   * - ``desired_speed``
     - ``3.0``
     - Constant vessel speed for arrival time and CPA.
   * - ``max_distance_from_shore``
     - ``150.0``
     - Upper scale for distance-to-shore bins.
   * - ``lookahead_distance``
     - ``150.0``
     - Forward segment for grounding-course evidence.
   * - ``max_dcpa_for_collision_risk``
     - ``100.0``
     - DCPA cutoff and evidence scale.
   * - ``max_tcpa_for_collision_risk``
     - ``120.0``
     - Future encounter horizon and evidence scale.
   * - ``max_relative_velocity_for_collision_risk``
     - ``5.0``
     - Upper scale for relative-speed evidence.
   * - ``max_depth_for_grounding_risk``
     - ``200.0``
     - Upper scale for depth evidence.
   * - ``grounding_depth_hard_limit``
     - ``6.0``
     - Depth below which recovery-failure evidence is forced.

Supply a positive ``max_distance_from_shore`` explicitly; its constructor
default is zero. The current evidence path uses pure distance bins. The
``inflation_*`` arguments are retained but do not activate another distance
computation in the current implementation.

Depth collision policy
----------------------

The planner's ``min_depth`` constructor argument is separate from
``grounding_depth_hard_limit``. It controls depth-based sampling. Historical
``FreeCheckMethod.DEPTH`` edge checks retain a fixed 10 m threshold. The main
notebook uses ``DEPTH_SAMPLING_OCCUPANCY_CHECKS`` to preserve the original study's
mixed policy. See :doc:`maps` for the distinction.

Recording for figures
---------------------

Pass ``record_bbn_snapshots=True`` to the planner when hazard figures are
needed later. Without snapshots, scalar risk and route information can still
be saved, but the missing BBN posterior fields cannot be reconstructed by the
plotting-only workflow.
