# Methods Summary & Computational Provenance

Automated documentation of pipeline algorithms, parameter specifications, runtime software environment, and output checksums.

---

## 1. Algorithmic Specifications

- **AHBA Microarray Preprocessing**:
  - Parcellation: Desikan-Killiany (82 anatomical regions: 68 cortical + 14 subcortical).
  - Normalization: Scaled Robust Sigmoid (SRS) per donor to eliminate donor batch effects while preserving dynamic range.
  - Probe Selection: Differential Stability ($DS \ge 0.1$) across 6 post-mortem donors.
- **Polygenic Risk Gene Sets**:
  - Sources: GWAS Catalog & MAGMA gene-level scoring.
  - Significance Thresholds: Genome-wide ($p < 5\times 10^{-8}$) and Suggestive ($p < 1\times 10^{-5}$).
  - Harmonization: Ensembl BioMart mapping to AHBA background gene symbols.
- **Statistical Null Models**:
  - Permutations: Mode `FAST` ($N = 1,000$ draws per test).
  - Spatial Null: Preserves regional empirical spatial autocorrelation (Moran's I tolerance: 0.1).
- **Neuroimaging Damage Comparison**:
  - Clinical Maps: ENIGMA Consortium cortical thickness case-control Cohen's $d$.
  - Spatial Significance: Alexander-Bloch spherical spin permutations ($N = 10,000$ spins).
- **Cell-Type Deconvolution**:
  - Regression: Ordinary Least Squares (OLS) against canonical human brain single-cell markers.

---

## 2. Software Runtime Environment

| Package / Resource | Version |
|---|---|
| `os` | `Windows 11 (AMD64)` |
| `python` | `3.13.5` |
| `numpy` | `2.5.3` |
| `scipy` | `1.18.1` |
| `pandas` | `3.0.6` |
| `matplotlib` | `3.11.2` |
| `seaborn` | `0.13.2` |
| `scikit-learn` | `not installed` |
| `statsmodels` | `0.15.0` |
| `nilearn` | `0.14.1` |
| `nibabel` | `5.4.2` |
| `streamlit` | `1.65.0` |
| `plotly` | `7.1.0` |
| `pytest` | `9.1.1` |

---

## 3. Processed Data Artifacts & SHA-256 Checksums

| Artifact File | Size | SHA-256 (first 16 chars) |
|---|---|---|
| `celltype_regression_results.parquet` | 12,703 bytes | `65dcc235b5c24668` |
| `crossdisorder_matrix.parquet` | 64,039 bytes | `416866eab7421d2f` |
| `crossdisorder_similarity.parquet` | 12,730 bytes | `ead23aa043d32096` |
| `damage_comparison_results.parquet` | 9,017 bytes | `7429bf5f62e8ce7b` |
| `enrichment_results.parquet` | 207,445 bytes | `a8049cb5b70b5f1f` |
| `robustness_threshold_results.parquet` | 6,321 bytes | `41e39a9b07fc6586` |
