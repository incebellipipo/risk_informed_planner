Installation
============

Requirements
------------

Use Python 3.12 or newer. Planning requires NumPy, SciPy, Matplotlib,
``cartoblobpy``, ``tqdm``, and PySMILE. The notebook extra adds JupyterLab,
IPykernel, and scikit-learn for the interpolation notebook.

From a source checkout, run these commands at the repository root:

.. code-block:: console

   python -m pip install -r requirements.txt
   python -m pip install -e ".[notebooks]"

``requirements.txt`` installs PySMILE using BayesFusion's academic package
index. It also installs the shared scientific dependencies from
``requirements.in``. Installing the project alone does not install PySMILE.

With the existing Miniconda environment:

.. code-block:: powershell

   conda activate risk_aware_planner3.12
   python -m pip install -r requirements.txt
   python -m pip install -e ".[notebooks]"
   python -m ipykernel install --user --name risk_aware_planner3.12 --display-name "Risk-informed planner (Python 3.12)"

Select that kernel when opening the notebooks.

PySMILE license
---------------

Obtain a valid license from `BayesFusion <https://www.bayesfusion.com/>`_ and
place its Python license file at:

.. code-block:: text

   src/risk_informed_planner/licenses/pysmile_license.py

The BBN wrapper loads this module before constructing the network. Renew the
file when the license expires. License material belongs in the local execution
environment; it is not needed to build or publish this documentation.

Check the installation
----------------------

.. code-block:: console

   python -c "from risk_informed_planner.risk.dynamic_risk import DynamicRisk; model = DynamicRisk(); print(type(model.cost_model).__name__)"

A successful check prints ``NormalizedExpectedLoss``. A native message such as
``SMILE INITERR`` indicates that network initialization failed; check the
PySMILE installation and license first.

Building just the documentation
-------------------------------

The documentation has its own small dependency set and does not import the
planner, execute notebooks, or initialize PySMILE:

.. code-block:: console

   python -m pip install -r docs/requirements.txt
   python -m sphinx -b html -W --keep-going docs docs/_build/html

Open ``docs/_build/html/index.html`` to browse the site locally. See
:doc:`publishing` for GitHub Pages deployment.
