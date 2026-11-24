"""Sphinx configuration. Building documentation never imports the planner."""
import os
from datetime import datetime, timezone
from urllib.parse import urlparse

project = "risk_informed_planner"
author = "Ozan Kaya and Emir Cem Gezer"
copyright = f"{datetime.now(timezone.utc).year}, {author}"
extensions = ["sphinx_rtd_theme", "sphinx.ext.githubpages", "sphinx.ext.mathjax",
              "sphinx.ext.autodoc", "sphinx.ext.napoleon", "autoapi.extension"]
# Analyze source without importing PySMILE or rendering its license module.
autoapi_dirs = ["../src/risk_informed_planner"]
autoapi_ignore = ["*/licenses/*", "*/__pycache__/*"]
autoapi_generate_api_docs = True
autoapi_root = "api"
autoapi_add_toctree_entry = True
autoapi_keep_files = False
autoapi_options = ["members", "undoc-members", "show-module-summary"]
autoapi_python_class_content = "class"
autodoc_typehints = "none"
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_use_ivar = True
html_theme = "sphinx_rtd_theme"
html_title = project
html_short_title = project
html_theme_options = {"navigation_depth": 3, "collapse_navigation": False,
                      "sticky_navigation": True, "style_external_links": True}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
exclude_patterns = ["_build", ".venv", "requirements.txt", "presentation", "Thumbs.db", ".DS_Store"]
nitpicky = True
# External annotation types have no local reference pages. Keep internal
# references strict without requiring online inventories or native imports.
nitpick_ignore = [("py:class", "enum.Enum"), ("py:class", "pysmile.Network")]
html_show_sourcelink = False
html_show_sphinx = False
smartquotes = False

# Pages project sites need their repository subpath; relative Sphinx links work
# both locally and at https://OWNER.github.io/REPOSITORY/.
repository_url = os.environ.get("DOCS_REPOSITORY_URL", "").rstrip("/")
if not repository_url and os.environ.get("GITHUB_REPOSITORY"):
    repository_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/") + "/" + os.environ["GITHUB_REPOSITORY"]
if repository_url:
    html_theme_options["vcs_pageview_mode"] = "blob"
    parsed = urlparse(repository_url)
    parts = parsed.path.strip("/").removesuffix(".git").split("/")
    if parsed.hostname == "github.com" and len(parts) == 2:
        html_context = {"display_github": True, "github_user": parts[0], "github_repo": parts[1],
                        "github_version": os.environ.get("DOCS_GIT_REF", "main"), "conf_py_path": "/docs/"}
html_baseurl = os.environ.get("DOCS_BASE_URL", "")
