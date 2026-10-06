"""
Notebook 03: Enrichment Scoring & Null Model Testing
=====================================================

Phase 3 of the VulnMap pipeline.
Tests regional brain vulnerability to disorder risk-gene expression
under three increasingly stringent null models:
1. NaiveNull (random permutation)
2. MatchedNull (length, GC content, mean expression matched)
3. SpatialNull (spatial autocorrelation & co-expression aware)

Usage:
    python notebooks/03_enrichment.py

Outputs:
    - Regional enrichment scores and p-values
    - Shrinkage summary table (Naive -> Matched -> Spatial)
    - P-value and z-score diagnostic plots
    - Cached results in data/processed/
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd

from vulnmap.enrichment import (
    compute_shrinkage_summary,
    enrichment_qc_report,
    run_enrichment_pipeline,
)
from vulnmap.utils import (
    ensure_dir,
    get_project_root,
    load_cached_dataframe,
    load_cached_json,
    load_params,
    logger,
    set_global_seed,
)


def _generate_synthetic_pilot_data(n_genes: int = 1000):
    """Generate synthetic fallback data if real AHBA / GWAS data not yet cached."""
    logger.info("Using synthetic pilot data for demonstration...")
    from vulnmap.damage import DK_ALL_REGIONS
    regions = DK_ALL_REGIONS
    n_regions = len(regions)
    rng = np.random.default_rng(42)
    genes = [f"GENE_{i}" for i in range(n_genes)]
    expr_vals = rng.standard_normal((n_regions, n_genes)).astype(np.float64)

    # Add spatial structure to first 50 genes
    coords = np.linspace(-1, 1, n_regions)[:, np.newaxis]
    expr_vals[:, :50] += coords * 2.0

    expression_df = pd.DataFrame(expr_vals, index=regions, columns=genes)

    # Synthetic gene sets for schizophrenia (catalog_5e-8 and catalog_1e-5)
    harmonized_genesets = {
        "schizophrenia__catalog_5e-8": genes[:30],
        "schizophrenia__catalog_1e-5": genes[:60],
        "bipolar_disorder__catalog_5e-8": genes[20:50],
    }

    return expression_df, harmonized_genesets


def main():
    """Run Phase 3 enrichment pipeline and produce reports and figures."""
    params = load_params()
    seed = params.get("seed", 42)
    set_global_seed(seed)

    logger.info("=" * 70)
    logger.info("Phase 3: Regional Enrichment & Null Model Audit")
    logger.info("=" * 70)

    # 1. Attempt to load real cached data from Phase 1 and Phase 2
    cache_dir = get_project_root() / "data" / "interim"
    expression_df = load_cached_dataframe(cache_dir, "expression_desikan_killiany")
    harmonized_genesets = load_cached_json(cache_dir, "harmonized_genesets")

    if expression_df is None or harmonized_genesets is None:
        logger.warning(
            "Phase 1 or Phase 2 cached data not found. "
            "Running pilot with synthetic benchmark data."
        )
        expression_df, harmonized_genesets = _generate_synthetic_pilot_data()

    # 2. Run enrichment pipeline for all null models
    null_models = ["naive", "matched", "spatial_coexpr"]
    results = run_enrichment_pipeline(
        expression_df=expression_df,
        harmonized_genesets=harmonized_genesets,
        null_model_names=null_models,
        params=params,
    )

    # 3. Print QC report & shrinkage table
    report = enrichment_qc_report(results, fdr_alpha=params["enrichment"]["fdr_alpha"])
    print("\n" + report + "\n")

    shrinkage = compute_shrinkage_summary(results, fdr_alpha=params["enrichment"]["fdr_alpha"])
    print("--- Shrinkage Summary ---")
    print(shrinkage.to_string(index=False))

    # 4. Generate visual diagnostic plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig_dir = ensure_dir(get_project_root() / "reports" / "figures")

        # Plot A: Shrinkage Bar / Slope Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Bar plot of significant regions per null model
        sns.barplot(
            data=shrinkage,
            x="disorder",
            y="n_significant_fdr",
            hue="null_model",
            palette="Blues_r",
            ax=axes[0],
        )
        axes[0].set_title("Significant Regions by Null Model (FDR < 0.05)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Disorder", fontsize=10)
        axes[0].set_ylabel("Significant Regions", fontsize=10)
        axes[0].tick_params(axis="x", rotation=15)

        # Plot B: P-value Distribution across Null Models
        sns.histplot(
            data=results,
            x="p_emp",
            hue="null_model",
            bins=25,
            multiple="layer",
            palette="Set2",
            alpha=0.6,
            ax=axes[1],
        )
        axes[1].set_title("Empirical P-Value Distribution", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Empirical p-value", fontsize=10)
        axes[1].set_ylabel("Count", fontsize=10)

        plt.tight_layout()
        plot_path = fig_dir / "03_enrichment_null_audit.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Diagnostic plot saved to {plot_path}")

    except Exception as e:
        logger.warning(f"Plot generation skipped: {e}")

    logger.info("=" * 70)
    logger.info("Phase 3 pipeline execution complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
