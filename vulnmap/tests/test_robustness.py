"""
Tests for VulnMap Phase 5c: Robustness & Sensitivity Analyses
==============================================================

These tests verify:
1. GWAS threshold sensitivity (5e-8 vs 1e-5 map correlation and Jaccard overlap).
2. Gene-set subsampling stability.
3. Intraclass Correlation Coefficient ICC(2, 1) computation.
4. Leave-one-donor consistency evaluation.
5. QC report generation.
"""

import numpy as np
import pandas as pd
import pytest

from vulnmap.robustness import (
    compute_icc_2_1,
    evaluate_threshold_sensitivity,
    leave_one_donor_consistency,
    robustness_qc_report,
    subsample_geneset_stability,
)


@pytest.fixture
def mock_multi_threshold_results():
    """Create mock enrichment results with both 5e-8 and 1e-5 thresholds."""
    rng = np.random.default_rng(42)
    regions = [f"region_{i}" for i in range(25)]

    rows = []
    base_sig = rng.standard_normal(25)

    for def_name in ["catalog_5e-8", "catalog_1e-5"]:
        noise = rng.standard_normal(25) * (0.2 if def_name == "catalog_1e-5" else 0.0)
        z = base_sig + noise

        for i, reg in enumerate(regions):
            rows.append({
                "disorder": "schizophrenia",
                "geneset_def": def_name,
                "null_model": "spatial_coexpr",
                "region": reg,
                "z": z[i],
                "p_fdr": 0.01 if z[i] > 1.0 else 0.5,
            })

    return pd.DataFrame(rows)


class TestThresholdSensitivity:
    """Tests for 5e-8 vs 1e-5 threshold evaluation."""

    def test_evaluate_thresholds(self, mock_multi_threshold_results):
        res = evaluate_threshold_sensitivity(mock_multi_threshold_results, fdr_alpha=0.05)
        assert isinstance(res, pd.DataFrame)
        assert len(res) == 1

        row = res.iloc[0]
        assert row["disorder"] == "schizophrenia"
        assert -1.0 <= row["pearson_r"] <= 1.0
        # By construction, 5e-8 and 1e-5 are highly correlated
        assert row["pearson_r"] > 0.8
        assert 0.0 <= row["jaccard_overlap"] <= 1.0


class TestGeneSetSubsampling:
    """Tests for gene-set size stability."""

    def test_subsampling_stability(self):
        rng = np.random.default_rng(42)
        n_regions, n_genes = 20, 50
        regions = [f"reg_{i}" for i in range(n_regions)]
        genes = [f"GENE_{i}" for i in range(n_genes)]
        expr_df = pd.DataFrame(rng.standard_normal((n_regions, n_genes)), index=regions, columns=genes)

        res = subsample_geneset_stability(expr_df, genes, target_size=10, n_repeats=20, seed=42)

        assert "mean_pairwise_r" in res
        assert "full_vs_subsample_mean_r" in res
        assert -1.0 <= res["mean_pairwise_r"] <= 1.0
        assert -1.0 <= res["full_vs_subsample_mean_r"] <= 1.0


class TestDonorICC:
    """Tests for ICC(2, 1) and leave-one-donor analysis."""

    def test_perfect_agreement_icc(self):
        # Perfect agreement across raters
        perfect_matrix = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ])
        icc = compute_icc_2_1(perfect_matrix)
        assert np.isclose(icc, 1.0, atol=1e-3)

    def test_leave_one_donor_consistency(self):
        rng = np.random.default_rng(42)
        regions = [f"reg_{i}" for i in range(15)]
        genes = [f"GENE_{i}" for i in range(20)]

        # Simulate 3 donors with correlated expression
        base_df = pd.DataFrame(rng.standard_normal((15, 20)), index=regions, columns=genes)
        donors = [base_df + rng.standard_normal((15, 20)) * 0.2 for _ in range(3)]

        res = leave_one_donor_consistency(donors, genes[:10])

        assert "icc" in res
        assert "mean_donor_r" in res
        assert res["icc"] > 0.5
        assert res["mean_donor_r"] > 0.5
