# Reproducibility Checklist & Verification Protocol

Protocol to reproduce all pipeline results, figures, tables, and web app artifacts from scratch.

---

## 1. Global Reproducibility Contract

- [x] **Global Random Seed**: Pinned globally to `seed=42` in `params.yaml` and propagated to `numpy.random`, `random`, and `scipy`.
- [x] **Deterministic Permutations**: All spin tests and gene set resampling operations use seeded `Generator` instances.
- [x] **Pinned Dependencies**: Pinned package versions documented in `requirements.txt` and `pyproject.toml`.
- [x] **Immutable Raw Data**: Raw data directory (`data/raw/`) is read-only during pipeline execution; all outputs write to `data/processed/` or `reports/`.

---

## 2. Step-by-Step Reproduction Commands

```bash
# 1. Environment installation
pip install -r requirements.txt
pip install -e .

# 2. Run test suite (100+ tests)
pytest tests/ -v

# 3. Execute analysis pipeline end-to-end
make expression    # Phase 1: AHBA expression matrices
make genesets      # Phase 2: GWAS risk gene sets
make enrichment    # Phase 3: Regional enrichment & null models
make damage        # Phase 4: ENIGMA damage comparisons & spin tests
make celltypes     # Phase 5a: Cell-type deconvolution
make crossdis      # Phase 5b: Cross-disorder structure & robustness
make figures       # Phase 6: High-res figures (PNG/SVG) & tables

# 4. Launch interactive dashboard
make app           # Phase 7: Streamlit web dashboard
```
