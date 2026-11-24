# risk_informed_planner

Risk-informed marine motion planning with an angle-aware RRT* planner, a
[Bayesian belief network (BBN)](src/risk_informed_planner/bbn/BBN_autonomous_ranked.xdsl),
and a normalized expected-loss cost model. Includes notebooks for planning
experiments, statistics, and figure generation.

## Install

Use Python 3.12 or newer in your preferred environment. Run these commands from
the repository root on Windows, macOS, or Linux:

```sh
python -m pip install -r requirements.txt
python -m pip install -e ".[notebooks]"
```

BBN inference requires a valid PySMILE license at
`src/risk_informed_planner/licenses/pysmile_license.py`. Keep this file private.

## Run notebooks

```sh
python -m jupyterlab notebooks
```

Open `kristiansund_bridge_planning.ipynb` and run all cells using your installed
environment. Keep the kernel working directory in the notebook's directory so
relative map and output paths resolve correctly. The study saves
`results/study_03.pkl` and figures in `plots/`.

Run the planning notebook before its companion figure notebooks. See the
[notebook guide](notebooks/README.md) for other experiments and statistics.
Map datasets are in [notebooks/maps](notebooks/maps).

To run all notebooks sequentially and save their outputs in place:

```sh
python notebooks/run_all.py
```

Each notebook uses a fresh kernel in the current Python environment. Figure
notebooks run after the experiments; failures are reported at the end.

## Tests

```sh
python -m unittest discover -s tests -v
```

The BBN integration tests require a valid PySMILE license.

## Documentation

```sh
python -m pip install -r docs/requirements.txt
python -m sphinx -b html -W --keep-going docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

Open <http://localhost:8000>. API documentation is generated from source
docstrings. See the [publishing guide](docs/publishing.rst) for GitHub Pages.
