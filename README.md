# VulnMap: Transcriptomic Brain Vulnerability & Spatially Aware Null Models

**Final Year Research Project & Open-Source Computational Neuroscience Tool**

> **Title:** *VulnMap: Does regional brain vulnerability follow risk-gene expression? A multi-disorder test with spatially aware null models, and an open tool for the community.*

---

## 🏛️ Project Architecture

This repository contains two interconnected components:

1. **`vulnmap/` — Core Computational Neuroscience Pipeline & Streamlit Explorer:**
   - Publication-grade Python research package (`src/vulnmap/`) implementing Phases 1 through 8.
   - Robust null model framework: **Naive**, **Matched**, and **Spatially Aware** null models.
   - Neuroimaging integration with **ENIGMA Toolbox** and Alexander-Bloch spherical spin tests.
   - Comprehensive test suite (118 passed unit tests).
   - 300 DPI publication figures (`reports/figures/`) and LaTeX tables (`reports/tables/`).
   - Interactive local **Streamlit Web Application** (`vulnmap/app/app.py`).
   - Full documentation in [`vulnmap/README.md`](vulnmap/README.md).

2. **`src/` — React / Vite Interactive Notebook & Project Presentation Viewer:**
   - Web interface presenting the methodology, scientific rationale, and interactive pipeline notebooks.
   - Built with React 19, TypeScript, Vite, Tailwind CSS, and Lucide Icons.

---

## 🚀 Quick Start

### 1. Run the VulnMap Computational Neuroscience Pipeline & Streamlit App

```bash
cd vulnmap

# Create virtual environment & install dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows (or: source .venv/bin/activate on Unix)
pip install -r requirements.txt
pip install -e .

# Run test suite
pytest tests/ -v

# Launch the Streamlit Interactive Explorer
streamlit run app/app.py
```

### 2. Run the React Presentation Web App

```bash
# In the repository root:
npm install
npm run dev
```

Open `http://localhost:3000` to view the interactive project notebook viewer.

---

## 📖 Scientific Publications & Hypotheses

Detailed hypothesis evaluations, figures, tables, and methodology guides are available in:
- [VulnMap Documentation & Gallery](vulnmap/README.md)
- [Hypotheses Evaluation Table](vulnmap/reports/results_hypotheses_table.md)
- [Thesis Figures Index](vulnmap/reports/thesis_figures_index.md)
- [Methods & Algorithmic Summary](vulnmap/reports/methods_summary.md)
- [Academic References (BibTeX)](vulnmap/reports/references.bib)
