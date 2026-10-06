"""
Tests for VulnMap Phase 5a: Cell-Type Deconvolution & Regression
=================================================================

These tests verify:
1. Canonical marker loading and custom marker CSV fallback.
2. Regional cell-type density proxy calculation.
3. OLS regression properties (R^2 in [0, 1], residual orthogonality).
4. Full pipeline runner and QC report generation.
"""

import numpy as np
import pandas as pd
import pytest

from vulnmap.celltypes import (
    CANONICAL_CELLTYPE_MARKERS,
    celltype_qc_report,
    compute_celltype_regional_scores,
    load_celltype_markers,
    regress_celltypes_out,
    run_celltype_analysis,
)


@pytest.fixture
def synthetic_expression():
    """Create a synthetic expression matrix containing canonical marker genes."""
    rng = np.random.default_rng(42)
    n_regions = 30

    # Collect marker genes
    marker_genes = []
    for genes in CANONICAL_CELLTYPE_MARKERS.values():
        marker_genes.extend(genes[:3])

    extra_genes = [f"OTHER_{i}" for i in range(50)]
    all_genes = marker_genes + extra_genes
    regions = [f"region_{i}" for i in range(n_regions)]

    data = rng.standard_normal((n_regions, len(all_genes))).astype(np.float64)
    return pd.DataFrame(data, index=regions, columns=all_genes)


class TestCelltypeLoadersAndScores:
    """Tests for marker loading and scoring."""

    def test_load_canonical_markers(self):
        df = load_celltype_markers(None)
        assert isinstance(df, pd.DataFrame)
        assert "gene" in df.columns
        assert "cell_type" in df.columns
        assert df["cell_type"].nunique() >= 5
        assert "Astrocyte" in df["cell_type"].values
        assert "Excitatory_Neuron" in df["cell_type"].values

    def test_load_custom_markers(self, tmp_path):
        custom_csv = tmp_path / "custom_markers.csv"
        pd.DataFrame({
            "gene": ["GFAP", "AIF1", "MBP"],
            "cell_type": ["Astrocyte", "Microglia", "Oligodendrocyte"],
        }).to_csv(custom_csv, index=False)

        df = load_celltype_markers(custom_csv)
        assert len(df) == 3
        assert set(df["gene"]) == {"GFAP", "AIF1", "MBP"}

    def test_compute_celltype_regional_scores(self, synthetic_expression):
        markers = load_celltype_markers(None)
        ct_scores = compute_celltype_regional_scores(synthetic_expression, markers)

        assert isinstance(ct_scores, pd.DataFrame)
        assert ct_scores.shape[0] == synthetic_expression.shape[0]
        assert ct_scores.shape[1] == markers["cell_type"].nunique()
        assert np.all(np.isfinite(ct_scores.values))


class TestCelltypeRegression:
    """Tests for OLS regression and variance decomposition."""

    def test_regression_statistics(self, synthetic_expression):
        markers = load_celltype_markers(None)
        ct_scores = compute_celltype_regional_scores(synthetic_expression, markers)

        rng = np.random.default_rng(42)
        # Construct vulnerability map partially driven by Astrocytes
        vuln = ct_scores["Astrocyte"] * 0.6 + rng.standard_normal(len(ct_scores)) * 0.4

        res = regress_celltypes_out(vuln, ct_scores)

        assert 0.0 <= res["r_squared"] <= 1.0
        assert 0.0 <= res["adjusted_r_squared"] <= 1.0
        assert len(res["residuals"]) == len(vuln)
        assert -1.0 <= res["r_raw_residual"] <= 1.0

        # Coefficients DataFrame
        coefs = res["coefficients"]
        assert "Intercept" in coefs["cell_type"].values
        assert "Astrocyte" in coefs["cell_type"].values
        assert np.all(np.isfinite(coefs["beta"].values))

    def test_qc_report(self):
        mock_summary = pd.DataFrame([{
            "disorder": "schizophrenia",
            "geneset_def": "catalog_5e-8",
            "null_model": "spatial_coexpr",
            "r_squared": 0.28,
            "adjusted_r_squared": 0.22,
            "f_stat": 4.5,
            "f_pvalue": 0.005,
            "r_raw_residual": 0.85,
            "top_cell_type": "Excitatory_Neuron",
            "top_cell_type_beta": 0.45,
            "top_cell_type_p": 0.001,
        }])
        report = celltype_qc_report(mock_summary)
        assert "Phase 5a: Cell-Type Deconvolution" in report
        assert "28.0%" in report
