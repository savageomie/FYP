"""
Notebook 06: Cross-Disorder Structure & Robustness
===================================================

Phase 5b & 5c of the VulnMap pipeline.

Computes pairwise vulnerability similarity across disorders, hierarchical
clustering, PCA decomposition, and Hypothesis 3 (H3) testing. Also evaluates
GWAS threshold sensitivity (5e-8 vs 1e-5).

Usage:
    python notebooks/06_crossdisorder.py

Outputs:
    - Cross-disorder similarity matrix (data/processed/crossdisorder_similarity.parquet)
    - Threshold robustness table (data/processed/robustness_threshold_results.parquet)
    - Diagnostic figures (reports/figures/06_crossdisorder_clustering.png)
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd

from vulnmap.crossdisorder import (
    crossdisorder_qc_report,
    run_crossdisorder_pipeline,
)
from vulnmap.robustness import (
    robustness_qc_report,
    run_robustness_pipeline,
)
from vulnmap.utils import (
    ensure_dir,
    get_project_root,
    load_params,
    logger,
    set_global_seed,
)


def main():
    """Run Phase 5b/5c cross-disorder and robustness pipelines."""
    params = load_params()
    seed = params.get("seed", 42)
    set_global_seed(seed)

    logger.info("=" * 70)
    logger.info("Phase 5b & 5c: Cross-Disorder Structure & Robustness")
    logger.info("=" * 70)

    # 1. Run cross-disorder pipeline
    cross_results = run_crossdisorder_pipeline(params=params)

    # Print cross-disorder QC report
    print("\n" + crossdisorder_qc_report(cross_results) + "\n")

    # 2. Run robustness sensitivity pipeline
    robust_df = run_robustness_pipeline(params=params)

    # Print robustness QC report
    print("\n" + robustness_qc_report(robust_df) + "\n")

    # 3. Generate diagnostic visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig_dir = ensure_dir(get_project_root() / "reports" / "figures")

        sim_df = cross_results["sim_matrix"]
        pca_scores = cross_results["pca"]["scores"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

        # Panel A: Pairwise Correlation Heatmap
        sns.heatmap(
            sim_df,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            square=True,
            cbar_kws={"label": "Pearson r"},
            ax=axes[0],
        )
        axes[0].set_title("Cross-Disorder Vulnerability Similarity", fontsize=11, fontweight="bold")
        axes[0].tick_params(axis="x", rotation=20)

        # Panel B: Threshold Sensitivity (Genome-Wide 5e-8 vs Suggestive 1e-5)
        if not robust_df.empty:
            sns.barplot(
                data=robust_df,
                x="disorder",
                y="pearson_r",
                hue="null_model",
                palette="Blues_r",
                ax=axes[1],
            )
            axes[1].set_title("GWAS Threshold Stability (5e-8 vs 1e-5)", fontsize=11, fontweight="bold")
            axes[1].set_xlabel("Disorder", fontsize=10)
            axes[1].set_ylabel("Pearson Correlation (r)", fontsize=10)
            axes[1].set_ylim(0.0, 1.05)
            axes[1].tick_params(axis="x", rotation=15)
        elif len(pca_scores) >= 3 and pca_scores.shape[1] >= 2:
            sns.scatterplot(
                data=pca_scores,
                x="PC1",
                y="PC2",
                s=120,
                color="#2980b9",
                ax=axes[1],
            )
            axes[1].set_title("PCA: Principal Axes of Vulnerability", fontsize=11, fontweight="bold")

        plt.tight_layout()
        plot_path = fig_dir / "06_crossdisorder_clustering.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"[OK] Cross-disorder diagnostic plot saved to {plot_path}")

    except Exception as e:
        logger.warning(f"Plot generation skipped: {e}")

    logger.info("=" * 70)
    logger.info("Phase 5b & 5c execution complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
