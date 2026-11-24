risk_informed_planner
=====================

Risk-informed motion planning for marine-vessel studies. The package combines
an angle-aware RRT* search, a ranked autonomous Bayesian belief network (BBN),
and normalized expected severity loss. Its main workflow plans a route through
Kristiansund and saves the explored tree and BBN posteriors for figure generation.

Start with :doc:`installation`, then follow :doc:`quickstart`. Use the
:doc:`notebooks` to find scenario studies and statistical experiments.

What the package provides
-------------------------

* Tree exploration with sampled edge risk, turn penalties, and risk cutoffs.
* Grounding and collision evidence from map layers and moving obstacles.
* Saved runs that support plotting without repeating BBN inference.
* Matrix, paper-panel, threshold, and replanning-campaign figures.

The retained implementation uses one BBN and one loss model. All six datasets
under ``notebooks/maps/`` are available to the study notebooks.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   notebooks

.. toctree::
   :maxdepth: 2
   :caption: User guide

   configuration
   risk_model
   maps
   saved_runs
   plotting

.. toctree::
   :maxdepth: 2
   :caption: Reference and development

   api/index
   testing
   publishing

.. note::

   The docs describe the implemented research workflow. Planner risk values are
   normalized model outputs, not calibrated probabilities of total damage.
