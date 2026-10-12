# Makefile for extracting code from Jupyter Notebooks: each script is the
# code of the notebook of the same stem in ../, without its markdown cells.
# Needs GNU Make run from a POSIX shell (Git Bash on Windows), and nbconvert,
# IPython (without which nbconvert leaves the code as it is and the last line
# of the script is cut), black and isort (at the pre-commit hook versions)
# in the environment that `python` names.
#
# `make run` executes the notebooks in place, from any shell, with ipykernel,
# matplotlib, PyBADS and PyVBMC too, on the PyIBS, PyBADS and PyVBMC that
# `python` imports. It prints first their locations and the versions of their
# installed metadata, and those of gpyreg, on which PyBADS and PyVBMC both
# depend. As Jupyter's interface does, it merges the consecutive pieces of a
# cell's output stream, which nbconvert alone saves one for each flush of the
# kernel's stream, and it leaves the times of the execution out of the cells'
# metadata.

NOTEBOOKS = $(wildcard ../pyibs_example_*.ipynb)
SCRIPTS = $(patsubst ../%.ipynb,%.py,$(NOTEBOOKS))

all: $(SCRIPTS)

%.py: ../%.ipynb
  # convert NB to script | remove leading lines | remove trailing line > output
	@python -m nbconvert $< --no-prompt --stdout --to script --PythonExporter.exclude_markdown=True | sed -e '1,3d' | head -n -1 > $@
	@python -m isort $@
	@python -m black $@

# PyBADS and PyVBMC remind the user of an installed release that is more than
# a year old at the start of a run in an IPython kernel, which is where the
# notebooks execute: the outputs that ship show no such reminder
run: export PYBADS_NO_UPDATE_REMINDER = 1
run: export PYVBMC_NO_UPDATE_REMINDER = 1
run:
	@python -c "import importlib; from importlib.metadata import version as v; [print(n, v(n), importlib.import_module(n).__file__) for n in ('pyibs', 'pybads', 'pyvbmc', 'gpyreg')]"
  # a preprocessor named here runs after the execution, the default one before
	@python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.record_timing=False --NotebookExporter.preprocessors=nbconvert.preprocessors.CoalesceStreamsPreprocessor $(NOTEBOOKS)

clean:
	@rm -f pyibs_example_*.py

.PHONY: all clean run
