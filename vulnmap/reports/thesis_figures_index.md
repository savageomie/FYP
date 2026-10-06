# Thesis Figures & Tables Index

This document maps all generated publication figures and tables to the claims and hypotheses in the dissertation.

---

## Figures Index

| Figure | File | Supported Hypothesis / Section | Key Takeaway |
|---|---|---|---|
| **Figure 1** | `fig1_data_qc.png` / `.svg` | **Methods / QC (Phases 1–2)** | Confirms high data quality: differential stability filtering ($DS \ge 0.10$) retains consistent probes; regional donor coverage is complete across Desikan-Killiany parcels; GWAS gene set sizes are adequately powered. |
| **Figure 2** | `fig2_regional_enrichment_shrinkage` | **Hypothesis 1 (Phase 3)** | Demonstrates the **Null Model Shrinkage Effect**: naive permutations produce massive false-positive inflation, whereas spatial co-expression nulls calibrate p-values and identify robust regional vulnerability extremes. |
| **Figure 3** | `fig3_damage_alignment` | **Hypothesis 2 (Phase 4)** | Evaluates alignment between transcriptomic vulnerability and in vivo patient cortical thinning (ENIGMA); tests significance against spherical Alexander-Bloch spin permutations. |
| **Figure 4** | `fig4_crossdisorder_and_celltypes` | **Hypothesis 3 (Phase 5a, 5b)** | Reveals shared cross-disorder vulnerability topography; hierarchical clustering groups clinically related disorders; cell-type deconvolution quantifies cellular mediation ($R^2$). |
| **Figure 5** | `fig5_robustness_sensitivity` | **Robustness (Phase 5c)** | Confirms pipeline stability: regional vulnerability profiles are preserved across GWAS significance thresholds ($5\times 10^{-8}$ vs $1\times 10^{-5}$) and across leave-one-donor-out AHBA cross-validation. |

---

## Tables Index

| Table | File | Section | Content |
|---|---|---|---|
| **Table 1** | `table1_expression_genesets_qc.csv` | Chapter 2 (Methods) | Standard preprocessing parameters, normalization specifications, and GWAS thresholds. |
| **Table 2** | `table2_enrichment_shrinkage.csv` | Chapter 3 (Results - H1) | Significant parcel counts, mean z-scores, and shrinkage percentages across all null models. |
| **Table 3** | `table3_damage_comparison.csv` | Chapter 3 (Results - H2) | Empirical correlation effect sizes ($r$, $\rho$), parametric p-values, and spin-test p-values. |
| **Table 4** | `table4_celltype_regression.csv` | Chapter 3 (Results - H3) | Cell-type composition variance explained ($R^2$), top driving cell classes, and beta coefficients. |
| **Table 5** | `table5_crossdisorder_similarity.csv` | Chapter 3 (Results - H3) | Pairwise cross-disorder Pearson correlation matrix of regional vulnerability maps. |
| **Table 6** | `table6_robustness_summary.csv` | Chapter 3 (Results - Robustness) | Threshold sensitivity stability metrics ($r$, $\rho$, Jaccard overlap) across null models. |
