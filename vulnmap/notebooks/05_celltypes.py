"""
Notebook 05: Cell-Type Deconvolution & Regression
===================================================

Phase 5a of the VulnMap pipeline.

Regresses regional cell-type density proxy scores (Astrocytes, Microglia,
Oligodendrocytes, Neurons, etc.) out of regional genetic vulnerability maps.
Quantifies variance explained (R^2) and identifies cellular drivers.

Usage:
    python notebooks/05_celltypes.py

Outputs:
    - Cell-type regression summary (data/processed/celltype_regression_results.parquet)
    - Diagnostic figures (reports/figures/05_celltype_decomposition.png)
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd

from vulnmap.celltypes import (
    celltype_qc_report,
    compute_celltype_regional_scores,
    load_celltype_markers,
    regress_celltypes_out,
    run_celltype_analysis,
)
from vulnmap.utils import (
    ensure_dir,
    get_project_root,
    load_cached_dataframe,
    load_params,
    logger,
    set_global_seed,
)


def main():
    """Run Phase 5a cell-type regression pipeline and produce reports and figures."""
    params = load_params()
    seed = params.get("seed", 42)
    set_global_seed(seed)

    logger.info("=" * 70)
    logger.info("Phase 5a: Cell-Type Deconvolution & Regression")
    logger.info("=" * 70)

    # 1. Run cell-type regression across all disorders
    summary_df = run_celltype_analysis(params=params)

    # 2. Print QC report
    report = celltype_qc_report(summary_df)
    print("\n" + report + "\n")

    # 3. Generate diagnostic visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig_dir = ensure_dir(get_project_root() / "reports" / "figures")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

        # Panel A: R^2 variance explained by cell types
        spatial_subset = summary_df[summary_df["null_model"] == "spatial_coexpr"].copy()
        if spatial_subset.empty:
            spatial_subset = summary_df.copy()

        spatial_subset["r2_pct"] = spatial_subset["r_squared"] * 100.0

        sns.barplot(
            data=spatial_subset,
            x="disorder",
            y="r2_pct",
            hue="geneset_def",
            palette="Blues_r",
            ax=axes[0],
        )
        axes[0].set_title("Variance Explained (R^2) by Cell-Type Composition", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Disorder", fontsize=10)
        axes[0].set_ylabel("Variance Explained (%)", fontsize=10)
        axes[0].tick_params(axis="x", rotation=15)

        # Panel B: Dominant driver cell types
        sns.barplot(
            data=spatial_subset,
            x="disorder",
            y="r_raw_residual",
            hue="top_cell_type",
            palette="Set2",
            ax=axes[1],
        )
        axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        axes[1].set_title("Vulnerability Map Preservation (Raw vs Residual r)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Disorder", fontsize=10)
        axes[1].set_ylabel("Pearson Correlation (Raw vs Residual)", fontsize=10)
        axes[1].tick_params(axis="x", rotation=15)

        plt.tight_layout()
        plot_path = fig_dir / "05_celltype_decomposition.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"[OK] Cell-type diagnostic plot saved to {plot_path}")

    except Exception as e:
        logger.warning(f"Plot generation skipped: {e}")

    logger.info("=" * 70)
    logger.info("Phase 5a execution complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
