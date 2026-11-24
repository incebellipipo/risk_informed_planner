Notebook guide
==============

Run notebooks in order, using each notebook's directory as its kernel working
directory. Use the kernel configured in :doc:`installation`. Map paths are
relative to ``notebooks/maps/`` through ``../maps/``.

Primary workflow
----------------

1. Run ``experiments/kristiansund_bridge_planning.ipynb``.
2. Confirm that ``experiments/results/study_03.pkl`` was created.
3. Run ``kristiansund_bridge_figures.ipynb`` or
   ``kristiansund_bridge_threshold_figures.ipynb`` from that directory.

For campaign figures, first run ``kristiansund_bridge_replanning.ipynb`` to
create ``results/campaign.pkl``, then run ``replanning_campaign_figures.ipynb``.
A campaign may finish early; the figure notebook uses the available steps.

Planning and risk-map notebooks
-------------------------------

These files live in ``notebooks/experiments/``.

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Notebook
     - Purpose
   * - ``kristiansund_bridge_planning.ipynb``
     - Main bridge study; saves the run and its figures.
   * - ``kristiansund_bridge_figures.ipynb``
     - Redraw the saved bridge run without replanning.
   * - ``kristiansund_bridge_threshold_figures.ipynb``
     - Threshold-based views of the saved bridge run.
   * - ``kristiansund_bridge_replanning.ipynb``
     - Replan as the own vessel and obstacles advance between steps.
   * - ``replanning_campaign_figures.ipynb``
     - Overlay available steps from the saved bridge campaign.
   * - ``open_water_planning.ipynb``
     - Open-water baseline and node-posterior inspection.
   * - ``open_water_replanning.ipynb``
     - Replanning campaign on the open-water map.
   * - ``kristiansund_port_entry_planning.ipynb``
     - Planning on the small port-entry dataset.
   * - ``trondheim_fjord_planning.ipynb``
     - Planning on the Trondheim dataset.
   * - ``trondheim_parameter_sweep.ipynb``
     - Sweep extension length and distance/risk weights on Trondheim.
   * - ``tree_growth_figures.ipynb``
     - Capture one tree at several sizes and compare risk maps and timing.
   * - ``interpolate_risk_surface.ipynb``
     - Interpolate scattered values into a surface while preserving large holes.

Statistics notebooks
--------------------

These files live in ``notebooks/statistics/``. Full sweeps can require many
planner runs; inspect the node-size lists and repetition counts before starting.
The stored timing includes evidence construction, native inference, and loss
computation, rather than BBN inference alone.

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Notebook
     - Purpose
   * - ``risk_evaluation_counts.ipynb``
     - Inspect risk-query counts in a bridge planning run.
   * - ``risk_inference_timing.ipynb``
     - Measure per-call risk computation time over tree sizes.
   * - ``neighborhood_crowding.ipynb``
     - Trace neighborhood size, edge-sample counts, and evaluation workload.
   * - ``cost_evaluation_counts.ipynb``
     - Measure candidate cost-function calls as a tree grows.
   * - ``cost_evaluation_counts_neighbor_limit.ipynb``
     - Compare cost-call growth with limited neighbor renewal.
   * - ``risk_inference_and_crowding.ipynb``
     - Generate timing and crowding datasets and their figures.
   * - ``risk_inference_and_crowding_figures.ipynb``
     - Redraw timing-budget and crowding figures from saved statistics.

Data dependencies
-----------------

The figure notebook needs both timing and crowding datasets. Generate them by
running ``risk_inference_and_crowding.ipynb``, or run
``risk_inference_timing.ipynb`` followed by ``neighborhood_crowding.ipynb``.
Their saved filenames remain:

* ``experiment_compute_average_risk_compute_time_stats.npz``
* ``experiment_crowding_stats.npz``

The cost-count notebooks save ``experiment_cost_count_stats.npz`` and
``experiment_cost_count_stats_max_nodes.npz`` respectively. Several notebooks
share output filenames; copy datasets elsewhere when comparing independent
experiment campaigns.

Filename migration
------------------

The former main notebook was ``study_03.ipynb``. Notebook names now describe the
scenario or measurement. The repository's ``notebooks/README.md`` contains the
complete previous-to-current filename table. Historical result filenames remain
unchanged, and the run loader supports the former package namespace.
