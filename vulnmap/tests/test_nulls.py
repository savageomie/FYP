"""
Tests for VulnMap Phase 3: Null Model Classes
==============================================

These tests verify:
1. NaiveNull: random gene-set permutations, size matching, seed reproducibility.
2. MatchedNull: stratification on gene properties, fallback behavior, reproducibility.
3. SpatialNull: spatial weights, Moran's I computation, candidate matching, reproducibility.
4. Factory function create_null_model().
5. Chunked score computation consistency.
"""

import numpy as np
import pandas as pd
import pytest

from vulnmap.nulls import (
    MatchedNull,
    NaiveNull,
    NullModelBase,
    SpatialNull,
    create_null_model,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_expression():
    """Create a synthetic expression matrix for testing."""
    rng = np.random.default_rng(42)
    n_regions = 20
    n_genes = 100
    regions = [f"region_{i}" for i in range(n_regions)]
    genes = [f"GENE_{i}" for i in range(n_genes)]
    data = rng.standard_normal((n_regions, n_genes)).astype(np.float64)
    # Add slight spatial gradient to first few genes
    gradient = np.linspace(-1, 1, n_regions)[:, np.newaxis]
    data[:, :10] += gradient * 2.0
    return pd.DataFrame(data, index=regions, columns=genes)


@pytest.fixture
def synthetic_annotations(synthetic_expression):
    """Create synthetic gene annotations with length and gc_content."""
    rng = np.random.default_rng(42)
    genes = list(synthetic_expression.columns)
    return pd.DataFrame({
        "gene": genes,
        "length": rng.integers(500, 50000, size=len(genes)),
        "gc_content": rng.uniform(0.35, 0.65, size=len(genes)),
    })


@pytest.fixture
def sample_gene_set():
    """A sample gene set of 10 genes."""
    return [f"GENE_{i}" for i in range(10)]


def dummy_score_fn(df: pd.DataFrame, genes: list) -> np.ndarray:
    """Simple scoring function: mean expression across selected genes."""
    valid = [g for g in genes if g in df.columns]
    if not valid:
        return np.zeros(df.shape[0])
    return df[valid].mean(axis=1).values


# =============================================================================
# Tests: NaiveNull
# =============================================================================

class TestNaiveNull:
    """Tests for NaiveNull random permutation model."""

    def test_generate_null_sets_shape_and_types(self, synthetic_expression, sample_gene_set):
        null_model = NaiveNull(synthetic_expression, n_perm=50, seed=42)
        null_sets = null_model.generate_null_gene_sets(sample_gene_set)

        assert len(null_sets) == 50
        for gene_set in null_sets:
            assert len(gene_set) == len(sample_gene_set)
            assert all(isinstance(g, str) for g in gene_set)
            # All genes must belong to expression matrix
            assert all(g in synthetic_expression.columns for g in gene_set)
            # No duplicates within a single set
            assert len(set(gene_set)) == len(gene_set)

    def test_seed_reproducibility(self, synthetic_expression, sample_gene_set):
        model1 = NaiveNull(synthetic_expression, n_perm=20, seed=42)
        model2 = NaiveNull(synthetic_expression, n_perm=20, seed=42)
        model3 = NaiveNull(synthetic_expression, n_perm=20, seed=99)

        sets1 = model1.generate_null_gene_sets(sample_gene_set)
        sets2 = model2.generate_null_gene_sets(sample_gene_set)
        sets3 = model3.generate_null_gene_sets(sample_gene_set)

        assert sets1 == sets2
        assert sets1 != sets3

    def test_compute_null_scores_shape_and_finite(self, synthetic_expression, sample_gene_set):
        null_model = NaiveNull(synthetic_expression, n_perm=30, seed=42, chunk_size=10)
        scores = null_model.compute_null_scores(
            sample_gene_set, score_fn=dummy_score_fn, show_progress=False
        )

        assert scores.shape == (30, synthetic_expression.shape[0])
        assert np.all(np.isfinite(scores))

    def test_custom_k(self, synthetic_expression, sample_gene_set):
        null_model = NaiveNull(synthetic_expression, n_perm=10, seed=42)
        null_sets = null_model.generate_null_gene_sets(sample_gene_set, k=15)
        for s in null_sets:
            assert len(s) == 15


# =============================================================================
# Tests: MatchedNull
# =============================================================================

class TestMatchedNull:
    """Tests for MatchedNull property-matched permutation model."""

    def test_strata_with_annotations(self, synthetic_expression, synthetic_annotations):
        model = MatchedNull(
            synthetic_expression,
            gene_annotations=synthetic_annotations,
            n_perm=20,
            seed=42,
        )
        assert len(model._strata) > 1
        assert len(model._gene_strata) == synthetic_expression.shape[1]

    def test_strata_without_annotations(self, synthetic_expression):
        """Falls back gracefully to mean-expression bins only."""
        model = MatchedNull(
            synthetic_expression,
            gene_annotations=None,
            n_perm=20,
            seed=42,
        )
        assert len(model._strata) > 1
        # All strata labels start with expr_bin
        for s in model._gene_strata:
            assert s.isdigit() or "-" in s

    def test_generate_null_sets(self, synthetic_expression, synthetic_annotations, sample_gene_set):
        model = MatchedNull(
            synthetic_expression,
            gene_annotations=synthetic_annotations,
            n_perm=25,
            seed=42,
        )
        null_sets = model.generate_null_gene_sets(sample_gene_set)

        assert len(null_sets) == 25
        for s in null_sets:
            assert len(s) == len(sample_gene_set)
            assert all(g in synthetic_expression.columns for g in s)

    def test_reproducibility(self, synthetic_expression, synthetic_annotations, sample_gene_set):
        m1 = MatchedNull(
            synthetic_expression, gene_annotations=synthetic_annotations, n_perm=15, seed=42
        )
        m2 = MatchedNull(
            synthetic_expression, gene_annotations=synthetic_annotations, n_perm=15, seed=42
        )
        assert m1.generate_null_gene_sets(sample_gene_set) == m2.generate_null_gene_sets(sample_gene_set)

    def test_unmatched_genes_fallback(self, synthetic_expression, synthetic_annotations):
        """When real genes are not in expression, fallback to naive null."""
        model = MatchedNull(
            synthetic_expression, gene_annotations=synthetic_annotations, n_perm=10, seed=42
        )
        missing_genes = ["UNKNOWN_1", "UNKNOWN_2"]
        null_sets = model.generate_null_gene_sets(missing_genes)
        assert len(null_sets) == 10
        assert len(null_sets[0]) == 2


# =============================================================================
# Tests: SpatialNull
# =============================================================================

class TestSpatialNull:
    """Tests for SpatialNull spatial autocorrelation-matched model."""

    def test_correlation_weights(self, synthetic_expression):
        model = SpatialNull(synthetic_expression, n_perm=10, seed=42)
        w = model._weights
        n_regions = synthetic_expression.shape[0]

        assert w.shape == (n_regions, n_regions)
        # Diagonal should be 0
        assert np.allclose(np.diag(w), 0.0)
        # Rows should sum to ~1.0
        row_sums = w.sum(axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_custom_distance_weights(self, synthetic_expression):
        n_regions = synthetic_expression.shape[0]
        rng = np.random.default_rng(42)
        dist = rng.uniform(1.0, 50.0, size=(n_regions, n_regions))
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0.0)

        model = SpatialNull(synthetic_expression, distance_matrix=dist, n_perm=10, seed=42)
        w = model._weights
        assert np.allclose(np.diag(w), 0.0)
        assert np.allclose(w.sum(axis=1), 1.0)

    def test_morans_i_calculation(self, synthetic_expression):
        n_regions = synthetic_expression.shape[0]
        # Construct 1D linear distance matrix: |i - j|
        coords = np.arange(n_regions)[:, np.newaxis]
        dist_matrix = np.abs(coords - coords.T).astype(np.float64)
        np.fill_diagonal(dist_matrix, 0.0)

        model = SpatialNull(
            synthetic_expression, distance_matrix=dist_matrix, n_perm=10, seed=42
        )
        morans = model.get_precomputed_morans()

        assert len(morans) == synthetic_expression.shape[1]
        assert isinstance(morans, pd.Series)
        assert np.all(np.isfinite(morans.values))
        # First 10 genes had a 1D spatial gradient added, so their Moran's I with distance |i-j| must be high (> 0)
        assert morans.iloc[:10].mean() > 0.1
        # Random noise genes (10:) have Moran's I near 0
        assert morans.iloc[:10].mean() > morans.iloc[10:].mean()

    def test_precomputed_morans_reuse(self, synthetic_expression):
        model1 = SpatialNull(synthetic_expression, n_perm=5, seed=42)
        cached_morans = model1.get_precomputed_morans()

        # Pass cached morans to model2
        model2 = SpatialNull(
            synthetic_expression, precomputed_morans=cached_morans, n_perm=5, seed=42
        )
        assert (model2.get_precomputed_morans() == cached_morans).all()

    def test_generate_null_sets(self, synthetic_expression, sample_gene_set):
        model = SpatialNull(synthetic_expression, n_perm=15, seed=42)
        null_sets = model.generate_null_gene_sets(sample_gene_set)

        assert len(null_sets) == 15
        for s in null_sets:
            assert len(s) == len(sample_gene_set)
            assert all(g in synthetic_expression.columns for g in s)

    def test_spatial_null_reproducibility(self, synthetic_expression, sample_gene_set):
        m1 = SpatialNull(synthetic_expression, n_perm=10, seed=42)
        m2 = SpatialNull(synthetic_expression, n_perm=10, seed=42)
        assert m1.generate_null_gene_sets(sample_gene_set) == m2.generate_null_gene_sets(sample_gene_set)


# =============================================================================
# Tests: Factory Function & Chunking
# =============================================================================

class TestNullModelFactoryAndChunking:
    """Tests for create_null_model() and chunking consistency."""

    def test_factory_creates_correct_types(self, synthetic_expression):
        m_naive = create_null_model("naive", synthetic_expression, n_perm=5)
        assert isinstance(m_naive, NaiveNull)

        m_matched = create_null_model("matched", synthetic_expression, n_perm=5)
        assert isinstance(m_matched, MatchedNull)

        m_spatial = create_null_model("spatial", synthetic_expression, n_perm=5)
        assert isinstance(m_spatial, SpatialNull)

        m_coexpr = create_null_model("spatial_coexpr", synthetic_expression, n_perm=5)
        assert isinstance(m_coexpr, SpatialNull)
        assert m_coexpr.match_coexpr is True

    def test_factory_invalid_name(self, synthetic_expression):
        with pytest.raises(ValueError, match="Unknown null model"):
            create_null_model("nonexistent_model", synthetic_expression)

    def test_chunking_consistency(self, synthetic_expression, sample_gene_set):
        """Chunked computation should produce identical results regardless of chunk size."""
        m_chunk5 = NaiveNull(synthetic_expression, n_perm=20, seed=42, chunk_size=5)
        scores5 = m_chunk5.compute_null_scores(
            sample_gene_set, score_fn=dummy_score_fn, show_progress=False
        )

        m_chunk20 = NaiveNull(synthetic_expression, n_perm=20, seed=42, chunk_size=20)
        scores20 = m_chunk20.compute_null_scores(
            sample_gene_set, score_fn=dummy_score_fn, show_progress=False
        )

        assert np.allclose(scores5, scores20)
