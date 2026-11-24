Run the main study
==================

Notebook workflow
-----------------

Open the primary notebook from its directory:

.. code-block:: powershell

   Set-Location notebooks/experiments
   python -m jupyterlab kristiansund_bridge_planning.ipynb

Run the cells in order with the environment described in :doc:`installation`.
The kernel working directory must be ``notebooks/experiments`` because
map and output paths are relative to it.

The notebook loads ``../maps/kristiansund_bridge/map.yaml``, defines the vessel
and obstacle scenario, grows the tree, and writes:

* ``results/study_03.pkl``: the tree, route, configuration, map reference,
  obstacle definitions, timing, and node-level BBN snapshots.
* ``plots/``: overview and paper figures.

The historical result filename is preserved. It does not need to match the
renamed notebook filename.

Generate ``results/study_03.pkl`` before running
``kristiansund_bridge_figures.ipynb`` or
``kristiansund_bridge_threshold_figures.ipynb``. Those notebooks redraw the
saved run without planning again.

Python workflow
---------------

The following example uses the same scenario and parameters as the main
notebook. Run it from the repository root with a valid PySMILE license.

.. literalinclude:: examples/plan_bridge.py
   :language: python
   :caption: docs/examples/plan_bridge.py

``nodes`` contains both connected tree nodes and samples retained as risk
observations. ``path`` contains the final root-to-goal route, or an empty list
if no acceptable goal connection was retained. Check ``path`` before using it.

Planning is stochastic. To compare parameter settings, set a NumPy random seed
and record the configuration alongside the result. A fixed seed alone does not
make wall-clock measurements reproducible.

Termination and unavailable paths
---------------------------------

``MAX_NODES`` and ``MAX_ITER`` are stopping targets after a collision-free
geometric approach to the goal has been found. They are not hard runtime
budgets. A goal that is geometrically reachable can still have all its
connections rejected by the risk cutoff or turn limit, producing an empty
route. A geometrically unreachable goal can keep the search running beyond
those targets. Choose reachable endpoints and monitor long experiments.

Next steps
----------

* Tune :doc:`configuration` and understand :doc:`risk_model`.
* Explore other scenarios in :doc:`notebooks`.
* Save and reload experiments with :doc:`saved_runs`.
* Customize figures using :doc:`plotting`.
