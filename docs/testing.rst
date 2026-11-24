Testing
=======

Run the suite from the repository root in the planning environment:

.. code-block:: powershell

   conda activate risk_aware_planner3.12
   python -m unittest discover -s tests -v

No separate test framework is required. The suite uses Python's ``unittest``
and the package's scientific dependencies. It does not test plotting functions.

Coverage
--------

* Normalized loss against analytical distributions, endpoint cases, scaling,
  monotonicity, and boundedness.
* Vessel motion and CPA geometry for head-on, crossing, parallel, and diverging
  situations, including translation and vessel-swap invariants.
* Evidence bins, moving-obstacle arrival times, worst-encounter selection,
  grounding lookahead, and evidence reset.
* Sampled edge risks, turn limits, parent selection, rewiring, cutoff rejection,
  and deterministic successful and unsuccessful goal connections.
* Current and legacy run loading, shared node links, campaign order, malformed
  files, and map-reference reconstruction from another working directory.
* All six map datasets and the syntax of the retained notebook code.

Integration checks
------------------

Most decision tests use deterministic risk substitutes. Additional checks use
real ``cartoblobpy`` maps, including an in-memory shallow patch to check both
depth policies. The map files are not edited.

The native BBN integration check requires a valid PySMILE license. It runs in a
subprocess because native initialization failures can otherwise exit Python
before the rest of the suite finishes. The subprocess checks finite normalized
outputs, repeated-query consistency, and posterior normalization.

Individual modules
------------------

.. code-block:: console

   python -m unittest tests.test_planner -v
   python -m unittest tests.test_dynamic_risk -v
   python -m unittest tests.test_experiment -v

Documentation checks
--------------------

Build documentation with warnings treated as errors:

.. code-block:: console

   python -m pip install -r docs/requirements.txt
   python -m sphinx -b html -W --keep-going docs docs/_build/html

The GitHub Actions documentation workflow uses the same command. It does not
run the native test suite or execute notebooks.

Maintaining source-driven API documentation
-------------------------------------------

Write class and function documentation as Python docstrings beside the
implementation. NumPy-style and Google-style sections are both supported by
Napoleon. Class docstrings should include constructor parameters, units,
important attributes, side effects, and error conditions. Place attribute docstrings immediately after dataclass fields and public
constants to document them. AutoAPI reads these without importing the module.

The entire ``docs/api/`` tree, including its index and module navigation, is
generated automatically from ``src/risk_informed_planner/`` during each build.
Do not create or edit files there. New public modules, classes, functions, and
members are discovered automatically; no per-object directives or registry
are needed. Generated source pages are removed after the build and ignored by
Git. The built HTML remains under ``docs/_build/html/api/``.

Keep tutorials and workflow examples in the user guides.

AutoAPI reads the Python sources using static analysis. Documentation builds
do not import the planner or require a PySMILE license. The license directory
is explicitly excluded from analysis. Run the strict build after changing
source docstrings to catch malformed markup and unresolved references.
