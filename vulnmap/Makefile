# =============================================================================
# VulnMap Makefile
# =============================================================================
# Usage:
#   make install     - Install Python dependencies
#   make test        - Run unit tests (fast, no data download)
#   make test-slow   - Run all tests including integration (needs AHBA data)
#   make expression  - Run Phase 1 (expression matrix construction)
#   make genesets    - Run Phase 2 (gene set retrieval)
#   make enrichment  - Run Phase 3 (enrichment analysis)
#   make damage      - Run Phase 4 (damage map comparison)
#   make celltypes   - Run Phase 5a (cell type deconvolution)
#   make crossdis    - Run Phase 5b (cross-disorder analysis)
#   make figures     - Run Phase 6 (visualization)
#   make app         - Run Phase 7 (Streamlit app)
#   make report      - Run Phase 8 (thesis write-up and reports)
#   make analysis    - Run Phases 1-5 end to end
#   make all         - Run everything
#   make clean       - Remove cached/interim data

.PHONY: install test test-slow expression genesets enrichment damage celltypes crossdis figures app report analysis all clean

PYTHON ?= python
PYTEST ?= pytest
STREAMLIT ?= streamlit

# --- Setup ---
install:
	pip install -r requirements.txt
	pip install -e .

# --- Testing ---
test:
	cd .. && $(PYTEST) vulnmap/tests/ -v --tb=short

test-slow:
	cd .. && $(PYTEST) vulnmap/tests/ -v --tb=short --runslow

# --- Pipeline phases ---
expression:
	$(PYTHON) -c "from vulnmap.expression import build_expression; r = build_expression(); print(r['params_used'])"

genesets:
	$(PYTHON) -c "from vulnmap.genesets import fetch_all_genesets; fetch_all_genesets()"

enrichment:
	$(PYTHON) -c "from vulnmap.enrichment import run_enrichment_pipeline; run_enrichment_pipeline()"

damage:
	$(PYTHON) -c "from vulnmap.damage import run_damage_comparison; run_damage_comparison()"

celltypes:
	$(PYTHON) -c "from vulnmap.celltypes import run_celltype_analysis; run_celltype_analysis()"

crossdis:
	$(PYTHON) -c "from vulnmap.crossdisorder import run_crossdisorder; run_crossdisorder()"

figures:
	$(PYTHON) -c "from vulnmap.viz import generate_all_figures; generate_all_figures()"

app:
	$(STREAMLIT) run app/app.py

report:
	$(PYTHON) -c "from vulnmap.report import generate_all_reports; generate_all_reports()"

# --- Composite targets ---
analysis: expression genesets enrichment damage celltypes crossdis

all: install test analysis figures

# --- Cleanup ---
clean:
	rm -rf data/interim/*
	rm -rf data/processed/*
	rm -rf reports/figures/*
	rm -rf reports/tables/*
	@echo "Cleaned interim and processed data. Raw data preserved."
