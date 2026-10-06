"""
Notebook 02: Gene Set Retrieval & Harmonization
================================================

Thin wrapper around vulnmap.genesets.fetch_all_genesets().

Usage:
    python notebooks/02_genesets.py

Outputs:
    - Summary table (disorder, category, n_genes per definition)
    - Jaccard overlap heatmap
    - Underpowered disorder warnings
    - Harmonized gene set cache
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from vulnmap.utils import load_params, set_global_seed, logger
from vulnmap.genesets import fetch_all_genesets, genesets_qc_report


def main():
    """Run Phase 2 gene set pipeline and produce QC report."""
    # Load parameters
    params = load_params()
    seed = params.get("seed", 42)
    rng = set_global_seed(seed)

    logger.info("=" * 70)
    logger.info("Phase 2: Gene Set Retrieval & Harmonization")
    logger.info("=" * 70)

    # Fetch all gene sets
    result = fetch_all_genesets(params=params)

    # Print QC report
    report = genesets_qc_report(result)
    print(report)

    # Summary
    summary = result["summary_table"]
    if not summary.empty:
        print("\n--- Disorder Gene Counts ---")
        display_cols = ["disorder", "category", "n_genes_5e8", "n_genes_1e5"]
        if "n_harmonized_5e8" in summary.columns:
            display_cols.append("n_harmonized_5e8")
        if "n_magma_fdr" in summary.columns:
            display_cols.append("n_magma_fdr")
        print(summary[display_cols].to_string(index=False))

    # Attempt Jaccard heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        jaccard = result.get("jaccard_5e8")
        if jaccard is not None and len(jaccard) > 1:
            fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
            sns.heatmap(
                jaccard,
                annot=True, fmt=".2f",
                cmap="YlOrRd",
                vmin=0, vmax=1,
                linewidths=0.5,
                ax=ax,
            )
            ax.set_title("Disorder Gene Set Overlap (Jaccard, p < 5e-8)")
            plt.tight_layout()

            from vulnmap.utils import get_project_root, ensure_dir
            fig_dir = get_project_root() / "reports" / "figures"
            ensure_dir(fig_dir)
            fig_path = fig_dir / "02_jaccard_heatmap.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"\n[OK] Jaccard heatmap saved to: {fig_path}")
            plt.close()

    except ImportError:
        print("\n[INFO] matplotlib/seaborn not available - skipping visualizations.")

    # Warnings
    if result.get("all_warnings"):
        print("\n--- WARNINGS ---")
        for w in result["all_warnings"]:
            print(f"  [WARN] {w}")

    print(f"\n[OK] Phase 2 complete. Ready for Phase 3 (03_enrichment).")


if __name__ == "__main__":
    main()
