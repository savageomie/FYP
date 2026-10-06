"""
Tests for VulnMap Phase 3: Enrichment Scoring & Statistical Testing
=====================================================================

These tests verify:
1. compute_region_scores: z-scoring, gene filtering, empty set handling.
2. empirical_pvalue: bounds in (0, 1], symmetry, extreme score behavior.
3. bh_fdr: Benjamini-Hochberg monotonicity, boundary conditions, NaN handling.
4. max_statistic_fwe: family-wise error rate, conservatism relative to empirical p.
5. compute_enrichment: end-to-end result DataFrame structure and column types.
6. compute_shrinkage_summary: summary aggregation across null models.
7. enrichment_qc_report: text report generation.
"""

import numpy as np
import pandas as pd
import pytest

from vulnmap.enrichment import (
    bh_fdr,
    compute_enrichment,
    compute_region_scores,
    compute_shrinkage_summary,
    empirical_pvalue,
    enrichment_qc_report,
    max_statistic_fwe,
)
from vulnmap.nulls import NaiveNull


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_expression():
    """Create a synthetic expression matrix."""
    rng = np.random.default_rng(42)
    n_regions = 25
    n_genes = 80
    regions = [f"region_{i}" for i in range(n_regions)]
    genes = [f"GENE_{i}" for i in range(n_genes)]
    data = rng.standard_normal((n_regions, n_genes)).astype(np.float64)
    # Give the first 5 genes a strong positive enrichment in the first 3 regions
    data[:3, :5] += 3.5
    return pd.DataFrame(data, index=regions, columns=genes)


@pytest.fixture
def sample_gene_set():
    """Gene set consisting of the first 5 genes."""
    return [f"GENE_{i}" for i in range(5)]


# =============================================================================
# Tests: compute_region_scores
# =============================================================================

class TestRegionScores:
    """Tests for regional enrichment score calculation."""

    def test_output_shape(self, synthetic_expression, sample_gene_set):
        scores = compute_region_scores(synthetic_expression, sample_gene_set)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (synthetic_expression.shape[0],)

    def test_top_regions_have_higher_scores(self, synthetic_expression, sample_gene_set):
        scores = compute_region_scores(synthetic_expression, sample_gene_set)
        # First 3 regions had +3.5 added to their genes
        assert scores[0] > scores[5]
        assert scores[1] > scores[5]
        assert scores[2] > scores[5]

    def test_missing_genes_filtered(self, synthetic_expression):
        # Mix of valid and invalid genes
        mixed_genes = ["GENE_0", "GENE_1", "NONEXISTENT_A", "NONEXISTENT_B"]
        scores = compute_region_scores(synthetic_expression, mixed_genes)
        assert scores.shape == (synthetic_expression.shape[0],)
        assert np.all(np.isfinite(scores))

    def test_empty_or_all_missing_genes_returns_zeros(self, synthetic_expression):
        scores_empty = compute_region_scores(synthetic_expression, [])
        assert np.all(scores_empty == 0.0)

        scores_missing = compute_region_scores(synthetic_expression, ["NOPE1", "NOPE2"])
        assert np.all(scores_missing == 0.0)

    def test_invalid_method_raises(self, synthetic_expression, sample_gene_set):
        with pytest.raises(ValueError, match="Unknown scoring method"):
            compute_region_scores(synthetic_expression, sample_gene_set, method="invalid_method")


# =============================================================================
# Tests: Statistical Functions (empirical_pvalue, bh_fdr, max_statistic_fwe)
# =============================================================================

class TestStatisticalFunctions:
    """Tests for empirical p-values, FDR, and FWE correction."""

    def test_empirical_pvalue_bounds(self):
        rng = np.random.default_rng(42)
        n_perm, n_regions = 100, 10
        observed = np.array([5.0] * 5 + [0.0] * 5)
        null_dist = rng.standard_normal((n_perm, n_regions))

        p = empirical_pvalue(observed, null_dist)
        assert p.shape == (n_regions,)
        # Strictly in (0, 1] - never 0!
        assert np.all(p > 0.0)
        assert np.all(p <= 1.0)
        # Extreme observed values should have minimal p-value
        assert p[0] == 1.0 / (n_perm + 1)
        # Observed value 0.0 should have high p-value
        assert p[5] > 0.5

    def test_bh_fdr_properties(self):
        # Basic test vector
        p_vals = np.array([0.001, 0.01, 0.04, 0.05, 0.20, 0.80])
        q_vals = bh_fdr(p_vals)

        assert q_vals.shape == p_vals.shape
        assert np.all(q_vals >= 0.0)
        assert np.all(q_vals <= 1.0)
        # Monotonicity: sorted p-values must yield non-decreasing q-values
        assert np.all(np.diff(q_vals) >= 0.0)
        # q-values >= p-values
        assert np.all(q_vals >= p_vals)

    def test_bh_fdr_edge_cases(self):
        # Empty array
        assert len(bh_fdr(np.array([]))) == 0

        # With NaNs
        p_with_nan = np.array([0.01, np.nan, 0.05])
        q_with_nan = bh_fdr(p_with_nan)
        assert np.isnan(q_with_nan[1])
        assert not np.isnan(q_with_nan[0])
        assert not np.isnan(q_with_nan[2])

    def test_max_statistic_fwe_properties(self):
        rng = np.random.default_rng(42)
        n_perm, n_regions = 200, 8
        observed = rng.standard_normal(n_regions)
        null_dist = rng.standard_normal((n_perm, n_regions))

        p_fwe = max_statistic_fwe(observed, null_dist)
        p_emp = empirical_pvalue(observed, null_dist)

        assert p_fwe.shape == (n_regions,)
        assert np.all(p_fwe > 0.0)
        assert np.all(p_fwe <= 1.0)
        # FWE must be at least as conservative as uncorrected empirical p
        assert np.all(p_fwe >= p_emp)


# =============================================================================
# Tests: compute_enrichment
# =============================================================================

class TestComputeEnrichment:
    """Tests for the primary compute_enrichment() function."""

    def test_result_dataframe_structure(self, synthetic_expression, sample_gene_set):
        null_model = NaiveNull(synthetic_expression, n_perm=50, seed=42)
        res = compute_enrichment(
            expression_df=synthetic_expression,
            gene_set=sample_gene_set,
            null_model=null_model,
            fdr_alpha=0.05,
        )

        assert isinstance(res, pd.DataFrame)
        assert len(res) == synthetic_expression.shape[0]

        expected_cols = [
            "region", "score", "null_mean", "null_sd", "z",
            "p_emp", "p_fdr", "p_fwe", "significant_fdr", "significant_fwe",
            "n_genes", "n_perm",
        ]
        for col in expected_cols:
            assert col in res.columns

        # Verify boolean flags match p-values
        assert (res["significant_fdr"] == (res["p_fdr"] < 0.05)).all()
        assert (res["significant_fwe"] == (res["p_fwe"] < 0.05)).all()

        # Check metadata values
        assert (res["n_perm"] == 50).all()
        assert (res["n_genes"] == len(sample_gene_set)).all()

    def test_detected_enriched_regions(self, synthetic_expression, sample_gene_set):
        """First 3 regions were boosted, so they should have high z-scores and low p-values."""
        null_model = NaiveNull(synthetic_expression, n_perm=100, seed=42)
        res = compute_enrichment(
            expression_df=synthetic_expression,
            gene_set=sample_gene_set,
            null_model=null_model,
        )

        top_regions = res.iloc[:3]
        other_regions = res.iloc[3:]
        assert top_regions["z"].mean() > other_regions["z"].mean()


# =============================================================================
# Tests: Shrinkage Summary & QC Report
# =============================================================================

class TestShrinkageAndQC:
    """Tests for compute_shrinkage_summary() and enrichment_qc_report()."""

    @pytest.fixture
    def mock_results_df(self):
        """Create mock multi-null results table."""
        rows = []
        for null_name, sig_fdr, sig_fwe in [
            ("naive", 10, 5),
            ("matched", 7, 3),
            ("spatial_coexpr", 3, 1),
        ]:
            for r in range(20):
                is_sig_fdr = r < sig_fdr
                is_sig_fwe = r < sig_fwe
                rows.append({
                    "disorder": "schizophrenia",
                    "geneset_def": "catalog_5e-8",
                    "null_model": null_name,
                    "region": f"region_{r}",
                    "score": 1.5,
                    "null_mean": 0.0,
                    "null_sd": 1.0,
                    "z": 1.5,
                    "p_emp": 0.01 if is_sig_fdr else 0.5,
                    "p_fdr": 0.02 if is_sig_fdr else 0.6,
                    "p_fwe": 0.03 if is_sig_fwe else 0.7,
                    "significant_fdr": is_sig_fdr,
                    "significant_fwe": is_sig_fwe,
                    "n_genes": 50,
                    "n_perm": 100,
                })
        return pd.DataFrame(rows)

    def test_shrinkage_summary(self, mock_results_df):
        summary = compute_shrinkage_summary(mock_results_df, fdr_alpha=0.05)
        assert len(summary) == 3
        assert list(summary["null_model"]) == ["matched", "naive", "spatial_coexpr"] or set(
            summary["null_model"]
        ) == {"naive", "matched", "spatial_coexpr"}

        naive_row = summary[summary["null_model"] == "naive"].iloc[0]
        assert naive_row["n_significant_fdr"] == 10
        assert naive_row["n_significant_fwe"] == 5

        spatial_row = summary[summary["null_model"] == "spatial_coexpr"].iloc[0]
        assert spatial_row["n_significant_fdr"] == 3
        assert spatial_row["n_significant_fwe"] == 1

    def test_qc_report_generation(self, mock_results_df):
        report = enrichment_qc_report(mock_results_df)
        assert isinstance(report, str)
        assert "VulnMap Phase 3: Enrichment QC Report" in report
        assert "Shrinkage Summary" in report
        assert "schizophrenia" in report
        assert "LIMITATIONS" in report

    def test_qc_report_empty(self):
        report = enrichment_qc_report(pd.DataFrame())
        assert "No results to report" in report
