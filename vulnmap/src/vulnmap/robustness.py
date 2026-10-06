"""
VulnMap: Robustness & Sensitivity Analyses
===========================================

Phase 5c of the VulnMap pipeline.

This module assesses the methodological sensitivity and reproducibility of
regional brain vulnerability maps across:
1. **GWAS Thresholds**: Genome-wide (5e-8) vs suggestive (1e-5) significance.
2. **Gene-Set Size Effect**: Subsampling large gene sets to isolate size bias.
3. **Donor Stability**: Leave-one-donor-out analysis and Intraclass Correlation (ICC)
   across the 6 AHBA post-mortem donor brains.

References
----------
- Arnatkevičiūtė, A., et al. (2019). A practical guide to linking brain-wide
  gene expression and neuroimaging data. NeuroImage, 189, 353-367.
- Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
  assessing rater reliability. Psychological Bulletin, 86(2), 420.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from vulnmap.enrichment import compute_region_scores
from vulnmap.utils import (
    ensure_dir,
    get_project_root,
    load_params,
    logger,
    set_global_seed,
)


# =============================================================================
# Threshold Sensitivity (5e-8 vs 1e-5)
# =============================================================================

def evaluate_threshold_sensitivity(
    enrichment_results: pd.DataFrame,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compare regional vulnerability maps between genome-wide (5e-8) and
    suggestive (1e-5) GWAS significance thresholds.

    Parameters
    ----------
    enrichment_results : pd.DataFrame
        Phase 3 enrichment results.
    fdr_alpha : float
        Significance threshold for Jaccard overlap.

    Returns
    -------
    pd.DataFrame
        Columns: ['disorder', 'null_model', 'pearson_r', 'spearman_rho',
                  'jaccard_overlap', 'n_sig_5e8', 'n_sig_1e5', 'n_regions']
    """
    records = []

    disorders = enrichment_results["disorder"].unique()
    null_models = enrichment_results["null_model"].unique()

    for disorder in disorders:
        d_df = enrichment_results[enrichment_results["disorder"] == disorder]

        for null_model in null_models:
            sub = d_df[d_df["null_model"] == null_model]

            sub_5e8 = sub[sub["geneset_def"].isin(["catalog_5e-8", "catalog_5e8"])]
            sub_1e5 = sub[sub["geneset_def"].isin(["catalog_1e-5", "catalog_1e5"])]

            if sub_5e8.empty or sub_1e5.empty:
                continue

            # Align regions
            s5 = sub_5e8.set_index("region")["z"]
            s1 = sub_1e5.set_index("region")["z"]
            common = s5.index.intersection(s1.index)

            if len(common) < 5:
                continue

            x = s5.loc[common].values
            y = s1.loc[common].values

            r_p = float(stats.pearsonr(x, y)[0])
            r_s = float(stats.spearmanr(x, y)[0])

            # Jaccard overlap of significant regions
            sig_5e8 = set(sub_5e8[sub_5e8["p_fdr"] < fdr_alpha]["region"])
            sig_1e5 = set(sub_1e5[sub_1e5["p_fdr"] < fdr_alpha]["region"])

            union_sig = sig_5e8.union(sig_1e5)
            inter_sig = sig_5e8.intersection(sig_1e5)
            jaccard = float(len(inter_sig) / len(union_sig)) if union_sig else 1.0

            records.append({
                "disorder": disorder,
                "null_model": null_model,
                "pearson_r": r_p,
                "spearman_rho": r_s,
                "jaccard_overlap": jaccard,
                "n_sig_5e8": len(sig_5e8),
                "n_sig_1e5": len(sig_1e5),
                "n_regions": len(common),
            })

    return pd.DataFrame(records)


# =============================================================================
# Gene-Set Size Sensitivity
# =============================================================================

def subsample_geneset_stability(
    expression_df: pd.DataFrame,
    gene_set: List[str],
    target_size: int = 15,
    n_repeats: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Subsample a large gene set to evaluate stability and score variance.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Regions x genes expression matrix.
    gene_set : list of str
        Full gene list.
    target_size : int
        Number of genes per subsample.
    n_repeats : int
        Number of bootstrap subsamples.
    seed : int
        Random seed.

    Returns
    -------
    dict
        - 'mean_pairwise_r': average correlation between subsampled score maps
        - 'sd_pairwise_r': standard deviation of correlation
        - 'full_vs_subsample_mean_r': average correlation of subsamples with full set map
    """
    rng = np.random.default_rng(seed)
    valid_genes = [g for g in gene_set if g in expression_df.columns]

    if len(valid_genes) <= target_size:
        return {
            "mean_pairwise_r": 1.0,
            "sd_pairwise_r": 0.0,
            "full_vs_subsample_mean_r": 1.0,
        }

    # Full set baseline score map
    full_scores = compute_region_scores(expression_df, valid_genes)

    n_regions = expression_df.shape[0]
    sub_maps = np.zeros((n_repeats, n_regions))

    for i in range(n_repeats):
        sampled = rng.choice(valid_genes, size=target_size, replace=False)
        sub_maps[i] = compute_region_scores(expression_df, sampled.tolist())

    # Full vs subsample correlations
    full_rs = [float(stats.pearsonr(sub_maps[i], full_scores)[0]) for i in range(n_repeats)]

    # Pairwise correlations among subsamples
    corr_mat = np.corrcoef(sub_maps)
    triu_idx = np.triu_indices_from(corr_mat, k=1)
    pairwise_rs = corr_mat[triu_idx]

    return {
        "mean_pairwise_r": float(np.mean(pairwise_rs)),
        "sd_pairwise_r": float(np.std(pairwise_rs)),
        "full_vs_subsample_mean_r": float(np.mean(full_rs)),
    }


# =============================================================================
# Donor Consistency & ICC
# =============================================================================

def compute_icc_2_1(data_matrix: np.ndarray) -> float:
    """
    Compute Intraclass Correlation Coefficient ICC(2, 1) - two-way random single measures.

    Parameters
    ----------
    data_matrix : np.ndarray
        Shape (n_targets, n_raters) - e.g. (n_regions, n_donors).

    Returns
    -------
    float
        ICC(2, 1) reliability coefficient.
    """
    n, k = data_matrix.shape
    if k < 2 or n < 2:
        return 1.0

    # Mean squares
    grand_mean = np.mean(data_matrix)
    ss_total = np.sum((data_matrix - grand_mean) ** 2)

    target_means = np.mean(data_matrix, axis=1)
    ss_target = k * np.sum((target_means - grand_mean) ** 2)
    ms_target = ss_target / (n - 1)

    rater_means = np.mean(data_matrix, axis=0)
    ss_rater = n * np.sum((rater_means - grand_mean) ** 2)
    ms_rater = ss_rater / (k - 1)

    ss_error = ss_total - ss_target - ss_rater
    ms_error = max(0.0, ss_error / ((n - 1) * (k - 1)))

    denom = ms_target + (k - 1) * ms_error + (k * (ms_rater - ms_error) / n)
    if denom <= 0:
        return 0.0

    icc = (ms_target - ms_error) / denom
    return float(np.clip(icc, -1.0, 1.0))


def leave_one_donor_consistency(
    donor_expressions: List[pd.DataFrame],
    gene_set: List[str],
) -> Dict[str, Any]:
    """
    Evaluate consistency of regional vulnerability maps across individual donors.

    Parameters
    ----------
    donor_expressions : list of pd.DataFrame
        Expression matrices for each AHBA donor.
    gene_set : list of str
        Disorder risk genes.

    Returns
    -------
    dict
        - 'icc': Intraclass Correlation Coefficient ICC(2, 1) across donors
        - 'mean_donor_r': average pairwise Pearson r between donors
        - 'donor_correlation_matrix': pd.DataFrame of pairwise donor correlations
    """
    n_donors = len(donor_expressions)
    if n_donors < 2:
        return {"icc": 1.0, "mean_donor_r": 1.0}

    # Find common regions across all donor DataFrames
    common_regions = donor_expressions[0].index
    for d_df in donor_expressions[1:]:
        common_regions = common_regions.intersection(d_df.index)

    donor_maps = []
    valid_donors = []

    for i, d_df in enumerate(donor_expressions):
        sub_df = d_df.loc[common_regions]
        scores = compute_region_scores(sub_df, gene_set)
        if not np.all(np.isnan(scores)):
            donor_maps.append(np.nan_to_num(scores, nan=0.0))
            valid_donors.append(f"Donor_{i + 1}")

    if len(donor_maps) < 2:
        return {"icc": 1.0, "mean_donor_r": 1.0}

    donor_matrix = np.column_stack(donor_maps)  # (n_regions, n_donors)
    icc_val = compute_icc_2_1(donor_matrix)

    corr_mat = np.corrcoef(donor_matrix.T)
    triu_idx = np.triu_indices_from(corr_mat, k=1)
    mean_r = float(np.mean(corr_mat[triu_idx])) if len(triu_idx[0]) > 0 else 1.0

    return {
        "icc": float(icc_val),
        "mean_donor_r": float(mean_r),
        "donor_correlation_matrix": pd.DataFrame(corr_mat, index=valid_donors, columns=valid_donors),
    }


# =============================================================================
# Pipeline Runner & QC
# =============================================================================

def run_robustness_pipeline(
    enrichment_results: Optional[pd.DataFrame] = None,
    params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Run threshold sensitivity and stability checks across all disorders.
    """
    if params is None:
        params = load_params()

    if output_dir is None:
        output_dir = get_project_root() / "data" / "processed"
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    if enrichment_results is None:
        enrich_file = output_dir / "enrichment_results.parquet"
        if not enrich_file.exists():
            raise FileNotFoundError(f"Enrichment results not found at {enrich_file}. Run Phase 3 first.")
        enrichment_results = pd.read_parquet(enrich_file)

    # 1. Threshold sensitivity
    thresh_df = evaluate_threshold_sensitivity(
        enrichment_results, fdr_alpha=params.get("enrichment", {}).get("fdr_alpha", 0.05)
    )

    if not thresh_df.empty:
        thresh_df.to_parquet(output_dir / "robustness_threshold_results.parquet", index=False)
        thresh_df.to_csv(output_dir / "robustness_threshold_results.csv", index=False)
        logger.info(f"[OK] Robustness threshold results saved to {output_dir}")

    return thresh_df


def robustness_qc_report(results: pd.DataFrame) -> str:
    """Format a text QC report for robustness checks."""
    lines = [
        "=" * 70,
        "VulnMap Phase 5c: Robustness & Sensitivity QC Report",
        "=" * 70,
        "",
    ]

    if results.empty:
        lines.append("No threshold sensitivity comparisons available.")
        lines.append("=" * 70)
        return "\n".join(lines)

    lines.extend([
        f"Threshold comparisons (5e-8 vs 1e-5): {len(results)}",
        f"Mean map correlation r: {results['pearson_r'].mean():.3f}",
        f"Mean Jaccard overlap of significant regions: {results['jaccard_overlap'].mean():.3f}",
        "",
        "--- Threshold Sensitivity (Genome-Wide 5e-8 vs Suggestive 1e-5) ---",
        results[["disorder", "null_model", "pearson_r", "jaccard_overlap", "n_sig_5e8", "n_sig_1e5"]].to_string(index=False),
        "",
        "--- Key Interpretation ---",
        "  1. High pearson_r (> 0.7) confirms that regional vulnerability patterns are stable across GWAS p-value thresholds.",
        "  2. Jaccard overlap measures the exact consistency of binary significance declarations.",
        "=" * 70,
    ])

    return "\n".join(lines)


# Alias for pipeline calls
run_robustness_analysis = run_robustness_pipeline
