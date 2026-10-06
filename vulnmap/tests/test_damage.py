"""
Tests for VulnMap Phase 4: Damage-Map Comparison & Spin Testing
================================================================

These tests verify:
1. Region name normalization across naming conventions.
2. Desikan-Killiany spherical coordinates geometry (unit norm, bilateral symmetry).
3. ENIGMA map loading for published disorders.
4. Custom atrophy loading and error handling.
5. Alexander-Bloch spherical spin permutations (bijectivity, seed reproducibility).
6. Spatial correlation calculations (Pearson, Spearman, parametric, and spin p-values).
7. QC report generation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vulnmap.damage import (
    DK_CORTICAL_REGIONS,
    DK_CORTICAL_REGIONS_LH,
    DK_CORTICAL_REGIONS_RH,
    compute_damage_correlation,
    damage_qc_report,
    get_desikan_killiany_spherical_coords,
    load_custom_atrophy,
    load_enigma_map,
    normalize_region_name,
    spin_test_permutations,
)


# =============================================================================
# Tests: Region Name Normalization
# =============================================================================

class TestNormalization:
    """Tests for brain region name normalization."""

    def test_normalize_freesurfer_aparc_variants(self):
        assert normalize_region_name("ctx-lh-bankssts") == "lh_bankssts"
        assert normalize_region_name("ctx-rh-insula") == "rh_insula"
        assert normalize_region_name("ctx_lh_cuneus") == "lh_cuneus"

    def test_normalize_prefix_and_suffix(self):
        assert normalize_region_name("L_superiorfrontal") == "lh_superiorfrontal"
        assert normalize_region_name("R_superiorfrontal") == "rh_superiorfrontal"
        assert normalize_region_name("bankssts_lh") == "lh_bankssts"
        assert normalize_region_name("fusiform_rh") == "rh_fusiform"

    def test_preserve_standard_names(self):
        assert normalize_region_name("lh_temporalpole") == "lh_temporalpole"
        assert normalize_region_name("Left-Thalamus-Proper") == "Left-Thalamus-Proper"


# =============================================================================
# Tests: Spherical Coordinates
# =============================================================================

class TestSphericalCoordinates:
    """Tests for Desikan-Killiany 3D spherical coordinates."""

    def test_shape_and_coverage(self):
        coords = get_desikan_killiany_spherical_coords()
        assert len(coords) == 68
        assert list(coords.columns) == ["x", "y", "z"]
        # All 68 parcels must be present
        for r in DK_CORTICAL_REGIONS:
            assert r in coords.index

    def test_unit_sphere_norm(self):
        coords = get_desikan_killiany_spherical_coords()
        norms = np.linalg.norm(coords.values, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_bilateral_symmetry(self):
        coords = get_desikan_killiany_spherical_coords()
        for lh_reg in DK_CORTICAL_REGIONS_LH:
            rh_reg = "rh_" + lh_reg[3:]
            lh_pt = coords.loc[lh_reg].values
            rh_pt = coords.loc[rh_reg].values

            # X must be opposite sign, Y and Z identical
            assert np.isclose(lh_pt[0], -rh_pt[0], atol=1e-5)
            assert np.isclose(lh_pt[1], rh_pt[1], atol=1e-5)
            assert np.isclose(lh_pt[2], rh_pt[2], atol=1e-5)


# =============================================================================
# Tests: Map Loaders
# =============================================================================

class TestMapLoaders:
    """Tests for ENIGMA and custom damage map loaders."""

    def test_load_builtin_enigma_disorders(self):
        for disorder in ["schizophrenia", "bipolar_disorder", "mdd", "epilepsy", "ocd"]:
            df = load_enigma_map(disorder)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 68
            assert "region" in df.columns
            assert "effect_size" in df.columns
            assert np.all(np.isfinite(df["effect_size"].values))

    def test_load_unknown_disorder_raises(self):
        with pytest.raises(FileNotFoundError):
            load_enigma_map("completely_unknown_disorder_xyz")

    def test_load_custom_atrophy(self, tmp_path):
        # Create a valid test CSV
        csv_file = tmp_path / "test_atrophy.csv"
        df_dummy = pd.DataFrame({
            "region": ["lh_bankssts", "lh_cuneus", "rh_insula"],
            "effect_size": [-0.35, -0.15, -0.40],
        })
        df_dummy.to_csv(csv_file, index=False)

        loaded = load_custom_atrophy(csv_file)
        assert len(loaded) == 3
        assert list(loaded["region"]) == ["lh_bankssts", "lh_cuneus", "rh_insula"]

    def test_load_custom_atrophy_missing_column(self, tmp_path):
        bad_csv = tmp_path / "bad_atrophy.csv"
        pd.DataFrame({"region": ["lh_cuneus"], "wrong_col": [-0.2]}).to_csv(bad_csv, index=False)

        with pytest.raises(ValueError, match="must contain 'region' and 'effect_size'"):
            load_custom_atrophy(bad_csv)


# =============================================================================
# Tests: Alexander-Bloch Spin Test
# =============================================================================

class TestSpinTest:
    """Tests for Alexander-Bloch spherical spin permutations."""

    def test_spin_permutations_shape_and_bijectivity(self):
        coords = get_desikan_killiany_spherical_coords()
        perms = spin_test_permutations(coords.values, n_perm=30, seed=42)

        assert perms.shape == (30, 68)
        for p in range(30):
            # Must be a bijection (each parcel index appears exactly once)
            assert len(set(perms[p])) == 68
            assert set(perms[p]) == set(range(68))

    def test_seed_reproducibility(self):
        coords = get_desikan_killiany_spherical_coords()
        p1 = spin_test_permutations(coords.values, n_perm=10, seed=42)
        p2 = spin_test_permutations(coords.values, n_perm=10, seed=42)
        p3 = spin_test_permutations(coords.values, n_perm=10, seed=99)

        assert np.array_equal(p1, p2)
        assert not np.array_equal(p1, p3)


# =============================================================================
# Tests: Spatial Correlation
# =============================================================================

class TestSpatialCorrelation:
    """Tests for spatial correlation between enrichment and damage maps."""

    @pytest.fixture
    def mock_aligned_maps(self):
        rng = np.random.default_rng(42)
        regions = DK_CORTICAL_REGIONS
        n = len(regions)
        # Construct correlated maps
        base_signal = rng.standard_normal(n)
        enrichment = pd.Series(base_signal + rng.standard_normal(n) * 0.4, index=regions)
        damage = pd.Series(-base_signal + rng.standard_normal(n) * 0.4, index=regions)
        return enrichment, damage

    def test_correlation_statistics(self, mock_aligned_maps):
        enrichment, damage = mock_aligned_maps
        res = compute_damage_correlation(enrichment, damage, n_perm=50, seed=42)

        assert res["n_regions"] == 68
        assert -1.0 <= res["pearson_r"] <= 1.0
        assert -1.0 <= res["spearman_rho"] <= 1.0
        # Negative correlation by design
        assert res["pearson_r"] < -0.4

        # P-values in (0, 1]
        assert 0.0 < res["pearson_p_param"] <= 1.0
        assert 0.0 < res["pearson_p_spin"] <= 1.0
        assert 0.0 < res["spearman_p_spin"] <= 1.0

    def test_too_few_regions(self):
        few_enrich = pd.Series([1.0, 2.0], index=["reg1", "reg2"])
        few_damage = pd.Series([-1.0, -2.0], index=["reg1", "reg2"])
        res = compute_damage_correlation(few_enrich, few_damage)
        assert res["n_regions"] == 2
        assert np.isnan(res["pearson_r"])


# =============================================================================
# Tests: QC Report
# =============================================================================

class TestDamageQC:
    """Tests for damage comparison QC report."""

    def test_qc_report_format(self):
        mock_df = pd.DataFrame([{
            "disorder": "schizophrenia",
            "geneset_def": "catalog_5e-8",
            "null_model": "spatial_coexpr",
            "damage_dataset": "ENIGMA",
            "measure": "cortical_thickness",
            "corr_type": "pearson",
            "r": -0.42,
            "p_param": 0.001,
            "p_spin": 0.02,
            "p_smash": np.nan,
            "p_fdr": 0.02,
            "n_regions": 68,
        }])
        report = damage_qc_report(mock_df)
        assert isinstance(report, str)
        assert "Phase 4: Damage-Map Comparison" in report
        assert "schizophrenia" in report
        assert "Significant under spin null" in report

    def test_qc_report_empty(self):
        report = damage_qc_report(pd.DataFrame())
        assert "No damage map comparisons available" in report
