Plotting saved results
======================

Use the plotting-only notebooks after producing their input files:

* ``kristiansund_bridge_figures.ipynb`` for the main study's overview and panels.
* ``kristiansund_bridge_threshold_figures.ipynb`` for threshold-based views.
* ``replanning_campaign_figures.ipynb`` for saved campaign overlays.

Run these examples from ``notebooks/experiments``. The notebooks load the historical
``results/study_03.pkl`` or ``results/campaign.pkl`` filenames.

Overview matrix
---------------

.. code-block:: python

   import matplotlib.pyplot as plt
   from risk_informed_planner.experiment import PlannerRun
   from risk_informed_planner.visualization import plot_matrix

   run = PlannerRun.load("results/study_03.pkl")
   fig, axes = plt.subplots(3, 3, figsize=(22, 16))
   plot_matrix(**run.plot_kwargs(), ax=axes)
   fig.savefig("plots/overview.pdf", bbox_inches="tight")

The matrix compares edge heuristics, risk, accumulated cost, and BBN accident
probabilities. Interpolated heatmaps describe the sampled tree, not an
independently evaluated dense risk grid.

Separate paper panels
---------------------

.. code-block:: python

   from risk_informed_planner.visualization import plot_paper_panels, save_paper_panels

   panels = plot_paper_panels(
       **run.plot_kwargs(),
       metrics=("risk", "collision", "grounding"),
       modes=("heatmap",), latex=False,
       vmin={"risk": 0.0}, vmax={"risk": 0.1},
       figsize={"risk": (8, 7), "collision": (4.5, 4), "grounding": (4.5, 4)},
   )
   save_paper_panels(panels, save_dir="plots", save_prefix="bridge_", save_ext="pdf")

The result is a mapping of panel names to Matplotlib axes. Many styling options
accept either a scalar or a dictionary keyed by panel name or metric. Setting
explicit color limits makes runs easier to compare.

Use ``latex=False`` for Matplotlib's built-in text rendering. ``latex=True``
enables external LaTeX rendering and requires a working LaTeX installation.

Thresholds and annotations
--------------------------

``plot_paper_panels_thresholded`` accepts ``threshold`` and ``keep="below"``
or ``keep="above"`` to mask the heatmap view. Thresholding a figure does not
rerun planning or change the original node risks.

``annotate_hazard_drivers`` adds bar insets and arrows for points of interest.
It samples the displayed hazard heatmaps; these interpolated values need not
match a fresh BBN evaluation at that exact coordinate.

Campaign overlays
-----------------

.. code-block:: python

   from risk_informed_planner.visualization import plot_campaign_overlay

   runs = PlannerRun.load_many("results/campaign.pkl")
   steps = tuple(range(min(3, len(runs))))
   panels = plot_campaign_overlay(runs, steps=steps, metrics=("risk",), latex=False)

A campaign can stop before the requested number of replanning steps. Select
only indices present in ``runs``. The companion notebook handles up to three
available steps.

The complete plotting entry points are listed in :doc:`api/risk_informed_planner/visualization/index`.
