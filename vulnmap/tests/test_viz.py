"""
Tests for VulnMap Phase 6: Visualization & Publication Figures
==============================================================

Verifies:
1. Publication styling configuration.
2. Figure 1: Pipeline Overview & Data QC (DS distribution, regional coverage, gene set sizes, Jaccard).
3. Figure 2: Hypothesis 1 — Regional Brain Enrichment & Null Model Shrinkage.
4. Figure 3: Hypothesis 2 — Risk Gene Vulnerability vs Neuroimaging Damage (ENIGMA & spin tests).
5. Figure 4: Hypothesis 3 & Cell Types — Cross-Disorder Similarity, Clustering, PCA, and Deconvolution.
6. Figure 5: Robustness & Sensitivity Analyses.
7. Publication tables generation (Tables 1-6).
8. Master orchestrator (generate_all_figures) and QC manifest.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from vulnmap.viz import (
    generate_all_figures,
    generate_all_tables,
    plot_crossdisorder_and_celltypes,
    plot_damage_comparison,
    plot_enrichment_and_shrinkage,
    plot_expression_and_genesets_qc,
    plot_robustness_evaluation,
    save_figure,
    set_publication_style,
    viz_qc_report,
)


@pytest.fixture
def mock_params(tmp_path):
    """Fixture providing minimal params for testing visualization."""
    fig_dir = tmp_path / "reports" / "figures"
    tbl_dir = tmp_path / "reports" / "tables"
    return {
        "seed": 42,
        "expression": {"ds_threshold": 0.10},
        "viz": {
            "dpi": 100,  # Lower DPI for fast test execution
            "formats": ["png", "svg"],
            "figure_dir": str(fig_dir),
            "table_dir": str(tbl_dir),
        },
    }


@pytest.fixture
def mock_enrichment_df():
    """Create mock enrichment dataframe for 2 disorders and 3 null models."""
    rng = np.random.default_rng(42)
    disorders = ["schizophrenia", "bipolar_disorder"]
    models = ["naive", "matched", "spatial_coexpr"]
    regions = [f"lh_region_{i}" for i in range(20)]

    rows = []
    for d in disorders:
        for m in models:
            for r in regions:
                z = rng.standard_normal()
                p_emp = rng.uniform(0.001, 0.5)
                rows.append({
                    "disorder": d,
                    "geneset_def": "catalog_5e-8",
                    "null_model": m,
                    "region": r,
                    "score": z * 0.5,
                    "null_mean": 0.0,
                    "null_sd": 1.0,
                    "z": z,
                    "p_emp": p_emp,
                    "p_fdr": p_emp * 1.2,
                    "significant_fdr": p_emp < 0.05,
                    "n_genes": 45,
                })
            # Also add catalog_1e-5 definition
            for r in regions:
                z = rng.standard_normal()
                p_emp = rng.uniform(0.001, 0.5)
                rows.append({
                    "disorder": d,
                    "geneset_def": "catalog_1e-5",
                    "null_model": m,
                    "region": r,
                    "score": z * 0.5,
                    "null_mean": 0.0,
                    "null_sd": 1.0,
                    "z": z,
                    "p_emp": p_emp,
                    "p_fdr": p_emp * 1.2,
                    "significant_fdr": p_emp < 0.05,
                    "n_genes": 120,
                })
    return pd.DataFrame(rows)


class TestStyleAndSave:
    """Test publication styling and figure saving helpers."""

    def test_set_publication_style(self):
        set_publication_style(font_scale=1.1)
        assert plt.rcParams["axes.linewidth"] == 0.8
        assert plt.rcParams["legend.frameon"] is True

    def test_save_figure(self, tmp_path, mock_params):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([0, 1], [0, 1])

        out_dir = tmp_path / "figs"
        saved = save_figure(fig, "test_fig", params=mock_params, output_dir=out_dir)
        plt.close(fig)

        assert "png" in saved
        assert "svg" in saved
        assert saved["png"].exists()
        assert saved["svg"].exists()
        assert saved["png"].stat().st_size > 0
        assert saved["svg"].stat().st_size > 0


class TestIndividualFigures:
    """Test individual figure generation functions."""

    def test_plot_expression_and_genesets_qc(self, tmp_path, mock_params):
        out_dir = tmp_path / "fig1"
        paths = plot_expression_and_genesets_qc(params=mock_params, output_dir=out_dir)
        assert "png" in paths
        assert "svg" in paths
        assert paths["png"].exists()
        assert paths["svg"].exists()

    def test_plot_enrichment_and_shrinkage(self, tmp_path, mock_params, mock_enrichment_df):
        out_dir = tmp_path / "fig2"
        paths = plot_enrichment_and_shrinkage(
            enrichment_df=mock_enrichment_df,
            params=mock_params,
            output_dir=out_dir,
        )
        assert "png" in paths
        assert "svg" in paths
        assert paths["png"].exists()
        assert paths["svg"].exists()

    def test_plot_damage_comparison(self, tmp_path, mock_params, mock_enrichment_df):
        out_dir = tmp_path / "fig3"
        dam_df = pd.DataFrame([
            {
                "disorder": "schizophrenia",
                "geneset_def": "catalog_5e-8",
                "null_model": "spatial_coexpr",
                "damage_dataset": "enigma",
                "measure": "cortical_thickness",
                "corr_type": "pearson",
                "r": 0.42,
                "p_param": 0.001,
                "p_spin": 0.02,
                "n_regions": 68,
            }
        ])
        paths = plot_damage_comparison(
            damage_df=dam_df,
            enrichment_df=mock_enrichment_df,
            params=mock_params,
            output_dir=out_dir,
        )
        assert "png" in paths
        assert "svg" in paths
        assert paths["png"].exists()

    def test_plot_crossdisorder_and_celltypes(self, tmp_path, mock_params):
        out_dir = tmp_path / "fig4"
        sim_df = pd.DataFrame(
            [[1.0, 0.85], [0.85, 1.0]],
            index=["schizophrenia", "bipolar_disorder"],
            columns=["schizophrenia", "bipolar_disorder"],
        )
        celltype_df = pd.DataFrame([
            {
                "disorder": "schizophrenia",
                "geneset_def": "catalog_5e-8",
                "null_model": "spatial_coexpr",
                "r_squared": 0.22,
                "top_cell_type": "Astrocyte",
            }
        ])
        paths = plot_crossdisorder_and_celltypes(
            sim_df=sim_df,
            celltype_df=celltype_df,
            params=mock_params,
            output_dir=out_dir,
        )
        assert "png" in paths
        assert "svg" in paths
        assert paths["png"].exists()

    def test_plot_robustness_evaluation(self, tmp_path, mock_params, mock_enrichment_df):
        out_dir = tmp_path / "fig5"
        rob_df = pd.DataFrame([
            {
                "disorder": "schizophrenia",
                "null_model": "spatial_coexpr",
                "pearson_r": 0.95,
                "spearman_rho": 0.94,
                "jaccard_overlap": 0.88,
                "n_sig_5e8": 40,
                "n_sig_1e5": 45,
                "n_regions": 68,
            }
        ])
        paths = plot_robustness_evaluation(
            robustness_df=rob_df,
            enrichment_df=mock_enrichment_df,
            params=mock_params,
            output_dir=out_dir,
        )
        assert "png" in paths
        assert "svg" in paths
        assert paths["png"].exists()


class TestTablesAndOrchestration:
    """Test table generation, master orchestrator, and QC reports."""

    def test_generate_all_tables(self, tmp_path, mock_params):
        table_dir = tmp_path / "tables"
        tables = generate_all_tables(params=mock_params, table_dir=table_dir)
        assert len(tables) == 6
        for name, path in tables.items():
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0
            df = pd.read_csv(path)
            assert not df.empty or "table" in name

    def test_generate_all_figures_end_to_end(self, tmp_path, mock_params):
        fig_dir = tmp_path / "figures"
        tbl_dir = tmp_path / "tables"

        res = generate_all_figures(
            params=mock_params,
            output_dir=fig_dir,
            table_dir=tbl_dir,
        )

        assert "figures" in res
        assert "tables" in res
        assert "manifest" in res

        assert len(res["figures"]) == 5
        for fig_key, path_dict in res["figures"].items():
            assert "png" in path_dict
            assert "svg" in path_dict
            assert path_dict["png"].exists()
            assert path_dict["svg"].exists()

        manifest_file = Path(res["manifest"])
        assert manifest_file.exists()
        with open(manifest_file, "r") as f:
            manifest_data = json.load(f)
        assert "figures" in manifest_data
        assert "tables" in manifest_data
        assert len(manifest_data["figures"]) == 5

        # Test QC report string
        report = viz_qc_report(res)
        assert "Phase 6 QC Report" in report
        assert "fig1" in report
        assert "table1" in report
