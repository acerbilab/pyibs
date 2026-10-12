# Configuration file for the Sphinx documentation builder.
#
# For the full list of options, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------

# The package is documented from this checkout, two levels up.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
EXAMPLES = os.path.join(HERE, "..", "..", "examples")

# -- Project information -----------------------------------------------------

project = "PyIBS"
copyright = (
    "2026, Machine and Human Intelligence research group "
    "(PI: Luigi Acerbi, University of Helsinki)"
)

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.extlinks",
]

numpydoc_show_class_members = False
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ["http", "https", "mailto"]
autodoc_default_options = {
    "exclude-members": "__weakref__",
}
# Section labels are prefixed with their document ("index:What is it?"),
# so that pages and notebooks can share section titles.
autosectionlabel_prefix_document = True

# Shorthand for external links:
extlinks = {
    "labrepos": ("https://github.com/acerbilab/%s", None),
    "mainbranch": ("https://github.com/acerbilab/pyibs/blob/main/%s", None),
}

coverage_show_missing_items = True

templates_path = ["_templates"]

# Patterns, relative to the source directory, of files and directories to
# ignore when looking for source files.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "PyIBS"

html_static_path = ["css/custom.css"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_theme_options = {
    "repository_url": "https://github.com/acerbilab/pyibs",
    "repository_branch": "main",
    "path_to_docs": "docsrc/source",
    "launch_buttons": {"colab_url": "https://colab.research.google.com/"},
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "extra_footer": (
        "<p>PyIBS is one of the open-source "
        '<a href="https://acerbilab.org/model-fitting/">tools for fitting '
        "models to data</a> from "
        '<a href="https://www.helsinki.fi/en/researchgroups/'
        "machine-and-human-intelligence\">Luigi Acerbi's group</a> at the "
        "University of Helsinki.</p>"
    ),
}
html_baseurl = "https://acerbilab.github.io/pyibs/"
html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"
]

todo_include_todos = True

# The example notebooks are rendered with their saved outputs, not executed.
nb_execution_mode = "off"

# Download notebooks as .ipynb, not as .ipynb.txt.
html_sourcelink_suffix = ""

# A notebook may repeat a section title, such as "Remarks"; its labels are
# not used.
suppress_warnings = [
    f"autosectionlabel._examples/{os.path.splitext(filename)[0]}"
    for filename in sorted(os.listdir(EXAMPLES))
    if filename.endswith(".ipynb")
]


def notebook_source_links(app, pagename, templatename, context, doctree):
    """Point notebook launch and edit buttons to their repository sources."""
    if not pagename.startswith("_examples/"):
        return
    source = pagename + context["page_source_suffix"]
    staged = app.config.html_theme_options["path_to_docs"] + "/" + source
    original = source.replace("_examples/", "examples/", 1)
    for group in context.get("header_buttons", []):
        for button in group.get("buttons", [group]):
            if "url" in button:
                button["url"] = button["url"].replace(staged, original)


def setup(app):
    # The theme builds its buttons at priority 501. The notebooks are
    # copied into source/_examples for rendering but live in examples/.
    app.connect("html-page-context", notebook_source_links, priority=900)
