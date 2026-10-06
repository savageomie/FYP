"""
Tests for VulnMap Phase 7: Streamlit App & Service Layer
========================================================

Verifies:
1. App service data loading (precomputed tables, manifest, configs).
2. Filter helpers (available disorders, null models).
3. Candidate gene set harmonization (exact match, case insensitivity, unrecognized genes).
4. Live on-the-fly regional vulnerability enrichment calculation.
5. App entry point and page scripts structure.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from vulnmap.app_service import (
    load_app_datasets,
    get_available_disorders,
    get_available_null_models,
    get_expression_matrix,
    harmonize_user_genes,
    run_user_gene_enrichment,
)
from vulnmap.utils import get_project_root


class TestAppService:
    """Test data loaders, helpers, and harmonization."""

    def test_load_app_datasets(self):
        data = load_app_datasets()
        expected_keys = [
            "params", "disorders", "enrichment", "damage",
            "celltypes", "crossdisorder_sim", "crossdisorder_mat",
            "robustness", "manifest"
        ]
        for k in expected_keys:
            assert k in data
        assert isinstance(data["enrichment"], pd.DataFrame)
        assert isinstance(data["damage"], pd.DataFrame)

    def test_get_available_disorders(self):
        df = pd.DataFrame({"disorder": ["schizophrenia", "bipolar_disorder", "schizophrenia"]})
        disorders = get_available_disorders(df)
        assert disorders == ["bipolar_disorder", "schizophrenia"]

    def test_get_available_null_models(self):
        df = pd.DataFrame({"null_model": ["naive", "spatial_coexpr", "matched"]})
        models = get_available_null_models(df)
        assert "spatial_coexpr" in models
        assert "matched" in models
        assert "naive" in models

    def test_harmonize_user_genes(self):
        ref_genes = ["DRD2", "COMT", "GRIN2A", "CACNA1C", "BDNF"]
        user_input = ["drd2", "COMT", "UNKNOWN_XYZ", "bdnf", "DRD2"]

        res = harmonize_user_genes(user_input, ref_genes)
        assert len(res["valid_genes"]) == 3  # DRD2, COMT, BDNF (deduplicated)
        assert "UNKNOWN_XYZ" in res["unrecognized_genes"]
        assert res["n_input"] == 4  # 4 unique input symbols
        assert res["n_valid"] == 3
        assert res["mapping_rate"] == 75.0


class TestLiveEnrichment:
    """Test on-the-fly candidate gene set enrichment."""

    def test_run_user_gene_enrichment_valid(self):
        genes = ["DRD2", "COMT", "GRIN2A", "BDNF"]
        res = run_user_gene_enrichment(
            gene_symbols=genes,
            null_model_name="naive",
            n_perm=100,  # Fast for unit tests
            seed=42,
        )

        assert "results_df" in res
        results_df = res["results_df"]
        assert 68 <= len(results_df) <= 85  # Desikan-Killiany cortical + subcortical regions
        assert "z" in results_df.columns
        assert "p_fdr" in results_df.columns
        assert "significant_fdr" in results_df.columns

        assert len(res["top_vulnerable"]) == 5
        assert len(res["top_resilient"]) == 5
        assert isinstance(res["mean_z"], float)
        assert res["max_z"] >= res["min_z"]

    def test_run_user_gene_enrichment_too_few_genes(self):
        with pytest.raises(ValueError, match="Too few harmonized genes"):
            run_user_gene_enrichment(
                gene_symbols=["UNKNOWN_1", "UNKNOWN_2"],
                null_model_name="naive",
                n_perm=50,
            )


class TestAppFiles:
    """Verify all Streamlit script files exist and are non-empty."""

    def test_app_scripts_exist(self):
        root = get_project_root()
        app_file = root / "app" / "app.py"
        assert app_file.exists()
        assert app_file.stat().st_size > 0

        pages = [
            "1_Regional_Enrichment.py",
            "2_Neuroimaging_Damage.py",
            "3_Cross_Disorder_CellTypes.py",
            "4_Custom_Gene_List.py",
            "5_Robustness_Reports.py",
        ]
        for p in pages:
            page_path = root / "app" / "pages" / p
            assert page_path.exists(), f"Page missing: {page_path}"
            assert page_path.stat().st_size > 0
