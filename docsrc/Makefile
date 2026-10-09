# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# copy the example notebooks to the source folder, replacing any copies
# that a failed build left
# render the docs
# copy the rendered docs and .nojekyll (for github.io) to the docs folder
# remove the example notebooks from the source folder
github:
	@rm -rf source/_examples
	@mkdir -p source/_examples
	@cp ../examples/*.ipynb source/_examples/
	@$(MAKE) html
	@mkdir -p ../docs
	@cp -a _build/html/. ../docs
	@cp .nojekyll ../docs/.nojekyll
	@rm -r source/_examples

clean:
	@rm -rf ../docs/*
	@if [ -d "./_build" ]; then rm -r _build; fi;

.PHONY: help github clean Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
