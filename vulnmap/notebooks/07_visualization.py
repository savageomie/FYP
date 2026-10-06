"""
Notebook 07: Publication Visualization & Figures
=================================================

Phase 6 of the VulnMap pipeline.

Generates publication-quality, multi-panel figures and summary tables
synthesizing the entire transcriptomic vulnerability analysis:
- Figure 1: Pipeline Overview & Data Quality Control (AHBA Expression & Gene Sets)
- Figure 2: Hypothesis 1 — Regional Brain Enrichment & Null Model Shrinkage
- Figure 3: Hypothesis 2 — Risk Gene Vulnerability vs Empirical Neuroimaging Damage
- Figure 4: Hypothesis 3 & Cell Types — Cross-Disorder Topography & Cellular Deconvolution
- Figure 5: Robustness & Sensitivity Analyses (GWAS Thresholds & Donor Stability)
- Tables 1-6: Summary CSV tables in reports/tables/
- Manifest: reports/figures/figures_manifest.json

Usage:
    python notebooks/07_visualization.py
    # or:
    make figures

Outputs:
    - reports/figures/fig1_data_qc.png (.svg)
    - reports/figures/fig2_regional_enrichment_shrinkage.png (.svg)
    - reports/figures/fig3_damage_alignment.png (.svg)
    - reports/figures/fig4_crossdisorder_and_celltypes.png (.svg)
    - reports/figures/fig5_robustness_sensitivity.png (.svg)
    - reports/figures/figures_manifest.json
    - reports/tables/table1_expression_genesets_qc.csv
    - reports/tables/table2_enrichment_shrinkage.csv
    - reports/tables/table3_damage_comparison.csv
    - reports/tables/table4_celltype_regression.csv
    - reports/tables/table5_crossdisorder_similarity.csv
    - reports/tables/table6_robustness_summary.csv
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from vulnmap.utils import (
    ensure_dir,
    get_project_root,
    load_params,
    logger,
    set_global_seed,
)
from vulnmap.viz import (
    generate_all_figures,
    viz_qc_report,
)


def main():
    """Run Phase 6 publication visualization pipeline."""
    params = load_params()
    seed = params.get("seed", 42)
    set_global_seed(seed)

    logger.info("=" * 70)
    logger.info("Phase 6: Publication Visualizations & Summary Tables")
    logger.info("=" * 70)

    # 1. Run all figure generation
    results = generate_all_figures(params=params)

    # 2. Print QC report
    report_text = viz_qc_report(results)
    print("\n" + report_text + "\n")

    print("[OK] Phase 6 complete. All publication figures and tables generated.")


if __name__ == "__main__":
    main()
