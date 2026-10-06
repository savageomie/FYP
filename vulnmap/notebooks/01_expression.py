"""
Notebook 01: Expression Matrix Construction
============================================

Thin wrapper around vulnmap.expression.build_expression().

Usage:
    python notebooks/01_expression.py

Outputs:
    - Shape report and QC summary
    - Coverage heatmap
    - Differential stability histogram
    - Flagged low-coverage regions
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from vulnmap.utils import load_params, set_global_seed, logger
from vulnmap.expression import build_expression, expression_qc_report


def main():
    """Run Phase 1 expression pipeline and produce QC report."""
    # Load parameters
    params = load_params()
    seed = params.get("seed", 42)
    rng = set_global_seed(seed)

    # Build expression matrix (Desikan-Killiany, primary)
    logger.info("=" * 70)
    logger.info("Phase 1: Expression Matrix Construction")
    logger.info("=" * 70)

    result = build_expression(params=params)

    # Print QC report
    report = expression_qc_report(result)
    print(report)

    # Summary statistics
    expr = result["expression"]
    gene_info = result["gene_info"]
    coverage = result["coverage"]

    print(f"\n--- Quick Summary ---")
    print(f"Expression matrix: {expr.shape[0]} regions x {expr.shape[1]} genes")
    print(f"Donors: {len(result.get('donor_expressions', []) or [])}")
    print(f"Genes retained (DS >= {params['expression']['ds_threshold']}): "
          f"{gene_info['kept'].sum()}")
    print(f"Low coverage regions: {coverage['low_coverage'].sum()}")

    # Attempt basic visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Panel A: DS histogram
        ds_vals = gene_info["ds"].dropna()
        if len(ds_vals) > 0:
            axes[0].hist(ds_vals, bins=50, color="#3498db", edgecolor="white", alpha=0.7)
            axes[0].axvline(
                x=params["expression"]["ds_threshold"],
                color="red", linestyle="--", linewidth=2,
                label=f"Threshold = {params['expression']['ds_threshold']}"
            )
            axes[0].set_xlabel("Differential Stability (DS)")
            axes[0].set_ylabel("Gene Count")
            axes[0].set_title("Gene Differential Stability Distribution")
            axes[0].legend()

        # Panel B: Coverage bar
        if "n_donors_present" in coverage.columns:
            donor_counts = coverage["n_donors_present"].value_counts().sort_index()
            axes[1].bar(
                donor_counts.index.astype(int),
                donor_counts.values,
                color="#2ecc71", edgecolor="white"
            )
            axes[1].set_xlabel("Number of Donors with Data")
            axes[1].set_ylabel("Number of Regions")
            axes[1].set_title("Regional Sample Coverage")

        plt.tight_layout()
        from vulnmap.utils import get_project_root, ensure_dir
        fig_dir = get_project_root() / "reports" / "figures"
        ensure_dir(fig_dir)
        fig_path = fig_dir / "01_expression_qc.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"\n[OK] QC figure saved to: {fig_path}")
        plt.close()

    except ImportError:
        print("\n[INFO] matplotlib not available - skipping visualizations.")

    print("\n[OK] Phase 1 complete. Ready for Phase 2 (02_genesets).")


if __name__ == "__main__":
    main()
