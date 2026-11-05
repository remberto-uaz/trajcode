# Reproduce every result file and figure of the paper.
PY ?= python3
.PHONY: install test data figures theory all clean
install:   ## install the package with test dependencies
	$(PY) -m pip install -e ".[test]"
test:      ## property and regression tests
	$(PY) -m pytest
data:      ## download WordNet 3.0 and check its SHA-256 (data/wordnet/wordnet.zip)
	$(PY) scripts/get_wordnet.py
figures: data  ## experiments E1-E7: results/*.csv, results/*.json and figures/*.pdf
	@for f in experiments/e[0-9]*.py experiments/fig_isit.py; do echo "== $$f"; $(PY) $$f || exit 1; done
	cd theory && $(PY) check_t34.py >/dev/null && $(PY) check_t09c.py
theory:    ## compile the working notes in theory/ (needs latexmk)
	cd theory && for f in 0*.tex; do latexmk -pdf -interaction=nonstopmode -halt-on-error $$f >/dev/null && echo "ok $$f"; done; latexmk -c >/dev/null
all: test figures
clean:
	cd theory && latexmk -c >/dev/null 2>&1 || true
