Publish on GitHub Pages
=======================

Local preview
-------------

From the repository root:

.. code-block:: console

   python -m pip install -r docs/requirements.txt
   python -m sphinx -b html -W --keep-going docs docs/_build/html
   python -m http.server 8000 --directory docs/_build/html

Open ``http://localhost:8000/``. The generated site includes a sidebar, search,
API navigation, math rendering, and a ``.nojekyll`` marker for GitHub Pages.
Do not commit ``docs/_build``.

One-time repository setup
-------------------------

1. Commit the documentation files, ``.github/workflows/docs.yml``, and the
   accompanying metadata changes to the repository.
2. On GitHub, open **Settings > Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.
4. Push to ``main``, or run the **Documentation** workflow manually from the
   Actions tab while selecting ``main``.
5. Open the URL shown by the workflow's ``github-pages`` deployment.

The expected Pages form for a project repository is
``https://OWNER.github.io/REPOSITORY/``. Use the actual deployed URL rather than
assuming that a repository rename has already changed it.

The workflow is ready to commit; creating the local documentation does not
publish the repository or change its Pages settings.

Workflow behavior
-----------------

The workflow builds on relevant pull requests and pushes to ``main``. It also
supports manual runs. Pull requests run the strict HTML build only; deployment
runs only on ``main`` and uses GitHub's Pages artifact and deployment actions.

The build job has read-only repository permissions. The separate deployment
job has ``pages: write`` and ``id-token: write`` and uses the ``github-pages``
environment. Configure environment protection rules on GitHub if the project
requires them. No custom publishing token or deployment branch is needed.

Documentation requirements are pinned in ``docs/requirements.txt``. CI installs
only those requirements; it does not install the planner or scientific stack,
read a PySMILE license, execute notebooks, or package map images and results.
Only ``docs/_build/html`` is uploaded as the Pages artifact.

If the repository uses a branch other than ``main``, update the workflow's push
filter and deployment condition together.

Repository links and base URL
-----------------------------

On GitHub Actions, repository links are derived from ``GITHUB_REPOSITORY``.
The workflow passes the Pages base URL into Sphinx. This avoids hard-coding an
owner or repository name in the site and supports repository subpaths.

Optional local environment variables are:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - Meaning
   * - ``DOCS_REPOSITORY_URL``
     - Repository URL used for GitHub source links.
   * - ``DOCS_GIT_REF``
     - Branch or tag used for source links; defaults to ``main``.
   * - ``DOCS_BASE_URL``
     - Deployed site base URL, if known.

Troubleshooting
---------------

**Build fails on a warning:** inspect the Sphinx message. Broken internal
references, missing pages, and malformed reStructuredText fail the build.
Reproduce with the local command above.

**Deployment is rejected:** check that Pages uses GitHub Actions as its source,
that the run is on ``main``, and that environment rules permit deployment.

**CSS or search files are missing:** upload the entire generated HTML directory,
including files and directories beginning with an underscore. Use the supplied
Pages workflow rather than copying selected HTML files.

**A newer workflow is waiting:** concurrency serializes publication on the same
branch; inspect the preceding workflow run in Actions.

Further reading
---------------

* `GitHub: using custom workflows with Pages <https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages>`_
* `GitHub: configuring a publishing source <https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site>`_
* `Read the Docs theme installation <https://sphinx-rtd-theme.readthedocs.io/en/stable/installing.html>`_
