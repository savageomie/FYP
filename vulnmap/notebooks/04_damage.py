"""
Notebook 04: Damage-Map Comparison & Spatial Spin Tests
========================================================

Phase 4 of the VulnMap pipeline.

Compares regional risk-gene enrichment z-scores from Phase 3 against
empirical neuroimaging damage maps from ENIGMA meta-analyses. Tests
whether genetic vulnerability predicts regional cortical thinning using
Alexander-Bloch spherical spin permutations.

Usage:
    python notebooks/04_damage.py

Outputs:
    - Damage comparison results table (data/processed/damage_comparison_results.parquet)
    - QC report comparing parametric vs spin-test p-values
    - Scatter and correlation plots (reports/figures/04_damage_comparison_pilot.png)
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd

from vulnmap.damage import (
    damage_qc_report,
    load_enigma_map,
    run_damage_comparison,
)
from vulnmap.utils import (
    ensure_dir,
    get_project_root,
    load_params,
    logger,
    set_global_seed,
)


def main():
    """Run Phase 4 damage comparison pipeline and produce reports and figures."""
    params = load_params()
    seed = params.get("seed", 42)
    set_global_seed(seed)

    logger.info("=" * 70)
    logger.info("Phase 4: Damage-Map Comparison & Spatial Null Testing")
    logger.info("=" * 70)

    # 1. Run damage comparison across disorders
    results = run_damage_comparison(params=params)

    if results.empty:
        logger.warning("No damage comparisons could be computed. Exiting.")
        return

    # 2. Print QC report
    report = damage_qc_report(results)
    print("\n" + report + "\n")

    # 3. Generate diagnostic figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig_dir = ensure_dir(get_project_root() / "reports" / "figures")

        # Load schizophrenia data for scatter plot
        enrich_file = get_project_root() / "data" / "processed" / "enrichment_results.parquet"
        enrich_df = pd.read_parquet(enrich_file)

        scz_enrich = enrich_df[
            (enrich_df["disorder"] == "schizophrenia") &
            (enrich_df["geneset_def"] == "catalog_5e-8") &
            (enrich_df["null_model"] == "spatial_coexpr")
        ].set_index("region")["z"]

        scz_damage = load_enigma_map("schizophrenia", "cortical_thickness").set_index("region")["effect_size"]

        common_regions = [r for r in scz_enrich.index if r in scz_damage.index]
        x_vals = scz_enrich.loc[common_regions].values
        y_vals = scz_damage.loc[common_regions].values

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

        # Panel A: Scatter Plot of Enrichment vs Empirical Damage
        sns.regplot(
            x=x_vals,
            y=y_vals,
            scatter_kws={"alpha": 0.7, "color": "#2c3e50", "s": 45},
            line_kws={"color": "#e74c3c", "linewidth": 2},
            ax=axes[0],
        )

        scz_row = results[
            (results["disorder"] == "schizophrenia") &
            (results["geneset_def"] == "catalog_5e-8") &
            (results["null_model"] == "spatial_coexpr") &
            (results["corr_type"] == "pearson")
        ].iloc[0]

        r_val = scz_row["r"]
        p_spin = scz_row["p_spin"]
        p_param = scz_row["p_param"]

        axes[0].set_title(
            f"Schizophrenia: Genetic Vulnerability vs Cortical Thinning\n"
            f"r = {r_val:.3f} (p_spin = {p_spin:.4f}, p_param = {p_param:.4f})",
            fontsize=11,
            fontweight="bold",
        )
        axes[0].set_xlabel("Genetic Enrichment z-score (Spatial Null)", fontsize=10)
        axes[0].set_ylabel("ENIGMA Cortical Thickness (Cohen's d)", fontsize=10)

        # Panel B: Correlation comparison across null models
        pearson_subset = results[results["corr_type"] == "pearson"]
        sns.barplot(
            data=pearson_subset,
            x="disorder",
            y="r",
            hue="null_model",
            palette="Blues_r",
            ax=axes[1],
        )
        axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[1].set_title("Spatial Correlation across Null Models", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Disorder", fontsize=10)
        axes[1].set_ylabel("Pearson Correlation (r)", fontsize=10)
        axes[1].tick_params(axis="x", rotation=15)

        plt.tight_layout()
        plot_path = fig_dir / "04_damage_comparison_pilot.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"[OK] Diagnostic plot saved to {plot_path}")

    except Exception as e:
        logger.warning(f"Plot generation skipped: {e}")

    logger.info("=" * 70)
    logger.info("Phase 4 pipeline execution complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
