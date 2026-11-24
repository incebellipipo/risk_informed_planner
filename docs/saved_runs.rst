Saving and loading runs
=======================

Use :class:`~risk_informed_planner.experiment.PlannerRun` to change figures
without rerunning the planner. It stores nodes, the route, configuration,
obstacles, timing, speed, metadata, and either the graph or a reference to its
map YAML.

Examples on this page assume ``notebooks/experiments`` as the working
directory, matching the notebook workflow.

Save a completed run
--------------------

.. code-block:: python

   from risk_informed_planner.experiment import PlannerRun

   run = PlannerRun(
       nodes=nodes, path=path, cfg=cfg, graph=graph,
       obstacles=obstacles, t_computation=elapsed,
       speed=risk.desired_speed, map_file=str(map_file),
       metadata={"scenario": "kristiansund_bridge"},
   )
   run.save("results/study_03.pkl")

Parent directories are created automatically. Node links and shared object
identity are preserved by pickle.

When ``map_file`` is supplied, the constructor resolves it to an absolute path.
Saving omits the graph object and retains its start/goal coordinates. Loading
reconstructs the graph from that YAML and restores start/goal. This keeps files
smaller, but the same absolute map path must remain available.

Without ``map_file``, an attached graph is serialized in full. That can make
result files substantially larger.

Reload and redraw
-----------------

.. code-block:: python

   from risk_informed_planner.experiment import PlannerRun
   from risk_informed_planner.visualization import plot_paper_panels

   run = PlannerRun.load("results/study_03.pkl")
   panels = plot_paper_panels(**run.plot_kwargs(), modes=("heatmap",), latex=False)

No BBN is reconstructed from a saved run. Hazard figures use the node snapshots
captured during planning, so enable snapshot recording before generating the
original run.

Campaigns
---------

.. code-block:: python

   PlannerRun.save_many(campaign_runs, "results/campaign.pkl")
   runs = PlannerRun.load_many("results/campaign.pkl")

Campaign order is preserved. ``load`` expects one ``PlannerRun``;
``load_many`` expects a list containing only ``PlannerRun`` objects. A mismatched
payload raises ``TypeError``.

Compatibility and portability
-----------------------------

The loader remaps the former ``risk_based_planner`` package name to
``risk_informed_planner``. Existing run files therefore remain usable after the
package rename. Other dependencies, object attributes, and map paths must still
be compatible.

Treat pickles as local iteration artifacts rather than an archival format.
Load only files you trust: Python pickle can execute code during deserialization.
For reproducible experiments, retain configuration, random seed, source revision,
map inputs, and the environment alongside the result.
