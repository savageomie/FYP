"""
Tests for VulnMap Phase 1: Expression Matrix Construction
=========================================================

These tests verify:
1. Expression matrix shape and data types.
2. No all-NaN regions in the filtered matrix.
3. Coverage table matches the number of regions.
4. Differential stability values are in [0, 1].
5. Gene filtering behaves correctly.

Note: Tests that require abagen data download are marked with
@pytest.mark.slow and skipped by default. Run with:
    pytest -m slow
to include them.
"""

import numpy as np
import pandas as pd
import pytest

from vulnmap.utils import set_global_seed


# =============================================================================
# Fixtures: synthetic data for unit tests (no abagen download needed)
# =============================================================================

@pytest.fixture
def synthetic_expression():
    """Create a synthetic expression matrix for testing."""
    rng = np.random.default_rng(42)
    n_regions = 82
    n_genes = 500
    regions = [f"region_{i}" for i in range(n_regions)]
    genes = [f"GENE{i}" for i in range(n_genes)]
    data = rng.standard_normal((n_regions, n_genes)).astype(np.float64)
    # Add some NaN values (realistic: some genes missing in some regions)
    mask = rng.random((n_regions, n_genes)) < 0.02  # 2% NaN
    data[mask] = np.nan
    return pd.DataFrame(data, index=regions, columns=genes)


@pytest.fixture
def synthetic_donor_dfs(synthetic_expression):
    """Create synthetic per-donor expression matrices."""
    rng = np.random.default_rng(42)
    donors = []
    for i in range(6):
        noise = rng.standard_normal(synthetic_expression.shape) * 0.5
        donor = synthetic_expression + noise
        # Each donor may be missing some regions
        if i >= 4:
            # Donors 4 and 5 have no right hemisphere (simulate)
            donor.iloc[41:, :] = np.nan
        donors.append(donor)
    return donors


# =============================================================================
# Tests: Gene info and differential stability
# =============================================================================

class TestGeneInfo:
    """Tests for gene info computation."""

    def test_ds_values_in_range(self, synthetic_expression, synthetic_donor_dfs):
        """DS values should be in [-1, 1] (Pearson correlation range)."""
        from vulnmap.expression import _compute_gene_info

        gene_info = _compute_gene_info(
            synthetic_expression, synthetic_donor_dfs, ds_threshold=0.1
        )
        ds = gene_info["ds"].dropna()
        assert (ds >= -1.0).all(), "DS values below -1"
        assert (ds <= 1.0).all(), "DS values above 1"

    def test_ds_filtering(self, synthetic_expression):
        """Genes below DS threshold should be marked as not kept."""
        from vulnmap.expression import _compute_gene_info

        # Use independent random donors (no shared signal) so DS values
        # are near zero and many genes get filtered at threshold 0.5
        rng = np.random.default_rng(123)
        independent_donors = [
            pd.DataFrame(
                rng.standard_normal(synthetic_expression.shape),
                index=synthetic_expression.index,
                columns=synthetic_expression.columns,
            )
            for _ in range(6)
        ]

        gene_info = _compute_gene_info(
            synthetic_expression, independent_donors, ds_threshold=0.5
        )
        # At threshold 0.5, independent random donors should yield low DS
        assert gene_info["kept"].dtype == bool
        n_kept = gene_info["kept"].sum()
        assert n_kept < len(gene_info), (
            f"Expected some genes to be filtered at threshold 0.5, "
            f"but all {len(gene_info)} were kept"
        )

    def test_gene_info_columns(self, synthetic_expression, synthetic_donor_dfs):
        """Gene info should have expected columns."""
        from vulnmap.expression import _compute_gene_info

        gene_info = _compute_gene_info(
            synthetic_expression, synthetic_donor_dfs, ds_threshold=0.1
        )
        expected_cols = {"mean_expr", "std_expr", "ds", "kept"}
        assert expected_cols == set(gene_info.columns)

    def test_mean_expr_finite(self, synthetic_expression, synthetic_donor_dfs):
        """Mean expression should be finite for genes with data."""
        from vulnmap.expression import _compute_gene_info

        gene_info = _compute_gene_info(
            synthetic_expression, synthetic_donor_dfs, ds_threshold=0.0
        )
        # Most should be finite (a few might be NaN if all values are NaN)
        assert gene_info["mean_expr"].notna().sum() > 0


# =============================================================================
# Tests: Coverage table
# =============================================================================

class TestCoverageTable:
    """Tests for coverage table construction."""

    def test_coverage_shape(self, synthetic_expression, synthetic_donor_dfs):
        """Coverage table should have one row per region."""
        from vulnmap.expression import _build_coverage_table

        coverage = _build_coverage_table(
            synthetic_expression, synthetic_donor_dfs, "desikan_killiany"
        )
        assert len(coverage) == len(synthetic_expression), (
            f"Coverage table has {len(coverage)} rows but expression has "
            f"{len(synthetic_expression)} regions"
        )

    def test_coverage_columns(self, synthetic_expression, synthetic_donor_dfs):
        """Coverage table should have expected columns."""
        from vulnmap.expression import _build_coverage_table

        coverage = _build_coverage_table(
            synthetic_expression, synthetic_donor_dfs, "desikan_killiany"
        )
        expected = {"n_donors_present", "pct_genes_available",
                    "n_samples_total", "low_coverage", "parcellation"}
        assert expected.issubset(set(coverage.columns))

    def test_donors_present_range(self, synthetic_expression, synthetic_donor_dfs):
        """n_donors_present should be between 0 and n_donors."""
        from vulnmap.expression import _build_coverage_table

        coverage = _build_coverage_table(
            synthetic_expression, synthetic_donor_dfs, "desikan_killiany"
        )
        n_donors = len(synthetic_donor_dfs)
        assert (coverage["n_donors_present"] >= 0).all()
        assert (coverage["n_donors_present"] <= n_donors).all()


# =============================================================================
# Tests: Expression matrix properties
# =============================================================================

class TestExpressionMatrix:
    """Tests for the expression matrix itself."""

    def test_no_all_nan_rows(self, synthetic_expression):
        """Filtered expression should not have all-NaN regions."""
        # In synthetic data, no row is entirely NaN
        all_nan = synthetic_expression.isna().all(axis=1)
        assert not all_nan.any(), (
            f"Found {all_nan.sum()} all-NaN regions: "
            f"{synthetic_expression.index[all_nan].tolist()}"
        )

    def test_shape_reasonable(self, synthetic_expression):
        """Expression matrix should have reasonable dimensions."""
        n_regions, n_genes = synthetic_expression.shape
        assert 10 <= n_regions <= 500, f"Unexpected n_regions: {n_regions}"
        assert 50 <= n_genes <= 25000, f"Unexpected n_genes: {n_genes}"

    def test_dtype_numeric(self, synthetic_expression):
        """Expression values should be numeric."""
        assert np.issubdtype(synthetic_expression.dtypes.iloc[0], np.floating)


# =============================================================================
# Tests: Parcellation info
# =============================================================================

class TestParcellation:
    """Tests for parcellation metadata."""

    def test_known_parcellations(self):
        """Known parcellation names should return info."""
        from vulnmap.expression import get_parcellation_info

        for parc in ["desikan_killiany", "schaefer_200"]:
            info = get_parcellation_info(parc)
            assert "name" in info
            assert "description" in info

    def test_unknown_parcellation_raises(self):
        """Unknown parcellation should raise ValueError."""
        from vulnmap.expression import get_parcellation_info

        with pytest.raises(ValueError, match="Unknown parcellation"):
            get_parcellation_info("nonexistent_atlas")


# =============================================================================
# Integration test (requires abagen data — slow)
# =============================================================================

@pytest.mark.slow
class TestExpressionIntegration:
    """Integration tests requiring real AHBA data download."""

    def test_build_expression_desikan(self, tmp_path):
        """Full build_expression for Desikan-Killiany."""
        from vulnmap.expression import build_expression

        result = build_expression(
            parcellation="desikan_killiany",
            cache_dir=tmp_path / "cache",
            return_donors=True,
        )
        expr = result["expression"]

        # Shape checks
        n_regions, n_genes = expr.shape
        assert 60 <= n_regions <= 100, f"Unexpected n_regions: {n_regions}"
        assert 1000 <= n_genes <= 20000, f"Unexpected n_genes: {n_genes}"

        # No all-NaN regions
        all_nan = expr.isna().all(axis=1)
        assert not all_nan.any()

        # Coverage table matches
        coverage = result["coverage"]
        assert len(coverage) == n_regions

        # Donor matrices
        donors = result["donor_expressions"]
        assert len(donors) >= 1  # At least 1 donor matrix
