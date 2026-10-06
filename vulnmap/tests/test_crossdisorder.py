"""
Tests for VulnMap Phase 5b: Cross-Disorder Similarity & H3 Testing
===================================================================

These tests verify:
1. Cross-disorder matrix construction from enrichment results.
2. Pairwise similarity matrix properties (symmetry, diagonal = 1, r in [-1, 1]).
3. Hierarchical clustering linkage and cophenetic correlation.
4. PCA score and loading matrices.
5. Hypothesis 3 (H3) permutation test logic and empirical p-value bounds.
"""

import numpy as np
import pandas as pd
import pytest

from vulnmap.crossdisorder import (
    build_crossdisorder_matrix,
    compute_pairwise_similarity,
    crossdisorder_pca,
    crossdisorder_qc_report,
    evaluate_category_similarity_h3,
    hierarchical_clustering,
)


@pytest.fixture
def mock_enrichment_results():
    """Create mock enrichment results for 4 disorders across 20 regions."""
    rng = np.random.default_rng(42)
    disorders = ["schizophrenia", "bipolar_disorder", "mdd", "parkinsons_disease"]
    regions = [f"region_{i}" for i in range(20)]

    rows = []
    # Make psychiatric disorders correlated with each other
    psych_signal = rng.standard_normal(20)
    neuro_signal = rng.standard_normal(20)

    for d in disorders:
        if d in ["schizophrenia", "bipolar_disorder", "mdd"]:
            z = psych_signal + rng.standard_normal(20) * 0.3
        else:
            z = neuro_signal + rng.standard_normal(20) * 0.3

        for i, reg in enumerate(regions):
            rows.append({
                "disorder": d,
                "geneset_def": "catalog_5e-8",
                "null_model": "spatial_coexpr",
                "region": reg,
                "z": z[i],
                "p_fdr": 0.01 if abs(z[i]) > 1.0 else 0.5,
            })

    return pd.DataFrame(rows)


class TestCrossDisorderMatrixAndSimilarity:
    """Tests for cross-disorder matrix and pairwise correlation."""

    def test_build_matrix(self, mock_enrichment_results):
        mat = build_crossdisorder_matrix(mock_enrichment_results)
        assert isinstance(mat, pd.DataFrame)
        assert mat.shape == (4, 20)
        assert set(mat.index) == {"schizophrenia", "bipolar_disorder", "mdd", "parkinsons_disease"}
        assert np.all(np.isfinite(mat.values))

    def test_pairwise_similarity_properties(self, mock_enrichment_results):
        mat = build_crossdisorder_matrix(mock_enrichment_results)
        sim_df, pval_df = compute_pairwise_similarity(mat, spin_test=False)

        assert sim_df.shape == (4, 4)
        assert np.allclose(np.diag(sim_df.values), 1.0)
        # Symmetry
        assert np.allclose(sim_df.values, sim_df.values.T)
        assert np.all(sim_df.values >= -1.0) and np.all(sim_df.values <= 1.0)

        # High similarity between SCZ and BIP
        assert sim_df.loc["schizophrenia", "bipolar_disorder"] > 0.5


class TestClusteringAndPCA:
    """Tests for clustering and dimensionality reduction."""

    def test_hierarchical_clustering(self, mock_enrichment_results):
        mat = build_crossdisorder_matrix(mock_enrichment_results)
        res = hierarchical_clustering(mat)

        assert "linkage" in res
        assert "cophenetic_corr" in res
        assert len(res["ordered_disorders"]) == 4
        assert -1.0 <= res["cophenetic_corr"] <= 1.0

    def test_crossdisorder_pca(self, mock_enrichment_results):
        mat = build_crossdisorder_matrix(mock_enrichment_results)
        pca_res = crossdisorder_pca(mat, n_components=2)

        scores = pca_res["scores"]
        assert scores.shape == (4, 2)
        assert np.sum(pca_res["explained_variance_ratio"]) <= 1.0001
        assert "PC1" in pca_res["top_drivers"]


class TestH3PermutationTest:
    """Tests for H3 permutation test."""

    def test_h3_hypothesis_test(self):
        rng = np.random.default_rng(42)
        psych_disorders = ["schizophrenia", "bipolar_disorder", "mdd", "adhd"]
        neuro_disorders = ["parkinsons_disease", "alzheimers_disease", "als", "frontotemporal_dementia"]
        all_disorders = psych_disorders + neuro_disorders

        # Simulate block-correlated similarity matrix
        n = len(all_disorders)
        R = np.eye(n)
        # High correlation within psychiatric (indices 0..3)
        for i in range(4):
            for j in range(4):
                if i != j:
                    R[i, j] = 0.75 + rng.uniform(-0.05, 0.05)
        # Moderate/high correlation within neurodegenerative (indices 4..7)
        for i in range(4, 8):
            for j in range(4, 8):
                if i != j:
                    R[i, j] = 0.70 + rng.uniform(-0.05, 0.05)
        # Low correlation between psychiatric and neurodegenerative
        for i in range(4):
            for j in range(4, 8):
                val = 0.05 + rng.uniform(-0.05, 0.05)
                R[i, j] = val
                R[j, i] = val

        sim_df = pd.DataFrame(R, index=all_disorders, columns=all_disorders)

        cat_map = {d: "psychiatric" for d in psych_disorders}
        cat_map.update({d: "neurodegenerative" for d in neuro_disorders})

        h3_res = evaluate_category_similarity_h3(sim_df, category_mapping=cat_map, n_perm=500, seed=42)

        assert "delta_obs" in h3_res
        assert "p_value" in h3_res
        assert 0.0 < h3_res["p_value"] <= 1.0
        assert h3_res["delta_obs"] > 0.5
        assert h3_res["supported"]
