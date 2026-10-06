"""
VulnMap: Enrichment Scoring & Statistical Testing
===================================================

Phase 3b of the VulnMap pipeline.

This module computes regional enrichment scores for disorder gene sets
and tests their significance using null models from nulls.py.

Enrichment score
----------------
For each brain region r, the enrichment score is the mean z-scored
expression of the gene set genes in that region:

    score(r) = mean( z_g(r) for g in gene_set )

where z_g(r) is the z-score of gene g's expression across regions.

Statistical testing
-------------------
- Empirical p-value: p = (b + 1) / (N + 1), where b is the number of
  null scores >= observed, N is the total permutations.
- BH-FDR: Benjamini-Hochberg correction across regions.
- Max-statistic FWE: family-wise error rate using the maximum null score.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from vulnmap.nulls import NullModelBase, create_null_model
from vulnmap.utils import (
    Timer,
    cache_dataframe,
    cache_json,
    ensure_dir,
    load_params,
    logger,
)


# =============================================================================
# Enrichment scoring
# =============================================================================

def compute_region_scores(
    expression_df: pd.DataFrame,
    gene_set: List[str],
    method: str = "mean_zscore",
) -> np.ndarray:
    """
    Compute enrichment scores for each brain region.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Group expression matrix (regions × genes).
    gene_set : list of str
        Gene symbols in the set.
    method : str
        Scoring method. Currently only 'mean_zscore' supported.

    Returns
    -------
    np.ndarray
        1D array of region scores (length = n_regions).
    """
    # Filter to genes present in expression matrix
    valid_genes = [g for g in gene_set if g in expression_df.columns]
    if not valid_genes:
        return np.zeros(expression_df.shape[0])

    subset = expression_df[valid_genes].values.astype(np.float64)

    if method == "mean_zscore":
        # Z-score each gene across regions
        means = np.nanmean(subset, axis=0, keepdims=True)
        stds = np.nanstd(subset, axis=0, keepdims=True)
        stds[stds == 0] = 1.0  # Avoid division by zero
        stds[np.isnan(stds)] = 1.0
        z_scored = (subset - np.nan_to_num(means, nan=0.0)) / stds
        # Region score = mean z-score across genes
        scores = np.nan_to_num(np.nanmean(z_scored, axis=1), nan=0.0)
    else:
        raise ValueError(f"Unknown scoring method: {method}")

    return scores


# =============================================================================
# Statistical testing
# =============================================================================

def empirical_pvalue(
    observed: np.ndarray,
    null_distribution: np.ndarray,
) -> np.ndarray:
    """
    Compute two-sided empirical p-values.

    p = (b + 1) / (N + 1) where b = number of null scores with
    |null| >= |observed|.

    Parameters
    ----------
    observed : np.ndarray
        1D array of observed scores (n_regions).
    null_distribution : np.ndarray
        2D array of null scores (n_perm × n_regions).

    Returns
    -------
    np.ndarray
        1D array of empirical p-values.
    """
    n_perm = null_distribution.shape[0]
    # Two-sided: count nulls with abs(null) >= abs(observed)
    obs_abs = np.abs(observed)
    null_abs = np.abs(null_distribution)

    b = np.sum(null_abs >= obs_abs[np.newaxis, :], axis=0)
    p = (b + 1) / (n_perm + 1)
    return p


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : np.ndarray
        1D array of p-values.

    Returns
    -------
    np.ndarray
        FDR-adjusted q-values.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.array([])

    # Handle NaN
    valid_mask = ~np.isnan(p)
    q = np.full_like(p, np.nan)

    valid_p = p[valid_mask]
    n_valid = len(valid_p)
    if n_valid == 0:
        return q

    order = np.argsort(valid_p)
    ranked = valid_p[order]
    bh_q = ranked * n_valid / (np.arange(1, n_valid + 1))
    # Enforce monotonicity (right to left)
    bh_q = np.minimum.accumulate(bh_q[::-1])[::-1]
    bh_q = np.clip(bh_q, 0.0, 1.0)

    result = np.empty_like(bh_q)
    result[order] = bh_q
    q[valid_mask] = result

    return q


def max_statistic_fwe(
    observed: np.ndarray,
    null_distribution: np.ndarray,
) -> np.ndarray:
    """
    Family-wise error rate using max-statistic correction.

    For each permutation, take the maximum absolute score across regions.
    The p-value for region r is the fraction of permutations where the
    max null score >= |observed(r)|.

    Parameters
    ----------
    observed : np.ndarray
        1D array of observed scores.
    null_distribution : np.ndarray
        2D array of null scores (n_perm × n_regions).

    Returns
    -------
    np.ndarray
        FWE-corrected p-values.
    """
    n_perm = null_distribution.shape[0]
    max_null = np.max(np.abs(null_distribution), axis=1)  # (n_perm,)
    obs_abs = np.abs(observed)

    p_fwe = np.array([
        (np.sum(max_null >= obs_abs[r]) + 1) / (n_perm + 1)
        for r in range(len(observed))
    ])
    return p_fwe


# =============================================================================
# Main enrichment pipeline
# =============================================================================

def compute_enrichment(
    expression_df: pd.DataFrame,
    gene_set: List[str],
    null_model: NullModelBase,
    score_method: str = "mean_zscore",
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run enrichment analysis for a single gene set against a null model.

    Parameters
    ----------
    expression_df : pd.DataFrame
        Group expression matrix (regions × genes).
    gene_set : list of str
        Gene symbols to test.
    null_model : NullModelBase
        Initialized null model instance.
    score_method : str
        Scoring method for regions.
    fdr_alpha : float
        FDR threshold for significance.

    Returns
    -------
    pd.DataFrame
        One row per region with columns:
        - region, score, null_mean, null_sd, z, p_emp, p_fdr, p_fwe,
          significant_fdr, significant_fwe, n_genes, n_perm
    """
    # Observed scores
    score_fn = lambda df, genes: compute_region_scores(df, genes, method=score_method)
    observed = score_fn(expression_df, gene_set)

    # Null distribution
    logger.info(
        f"Computing null distribution: {null_model.name}, "
        f"{null_model.n_perm} permutations, "
        f"{len(gene_set)} genes"
    )
    with Timer(f"Null distribution ({null_model.name})"):
        null_scores = null_model.compute_null_scores(
            gene_set, score_fn=score_fn
        )

    # Statistics
    null_mean = null_scores.mean(axis=0)
    null_sd = null_scores.std(axis=0)
    null_sd[null_sd == 0] = 1.0  # Avoid division by zero

    z_scores = (observed - null_mean) / null_sd
    p_emp = empirical_pvalue(observed, null_scores)
    p_fdr = bh_fdr(p_emp)
    p_fwe = max_statistic_fwe(observed, null_scores)

    # Build results DataFrame
    regions = expression_df.index
    results = pd.DataFrame({
        "region": regions,
        "score": observed,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "z": z_scores,
        "p_emp": p_emp,
        "p_fdr": p_fdr,
        "p_fwe": p_fwe,
        "significant_fdr": p_fdr < fdr_alpha,
        "significant_fwe": p_fwe < fdr_alpha,
        "n_genes": len([g for g in gene_set if g in expression_df.columns]),
        "n_perm": null_model.n_perm,
    })

    # Summary
    n_sig_fdr = results["significant_fdr"].sum()
    n_sig_fwe = results["significant_fwe"].sum()
    logger.info(
        f"  Results: {n_sig_fdr} regions significant (FDR < {fdr_alpha}), "
        f"{n_sig_fwe} regions significant (FWE < {fdr_alpha})"
    )

    return results


def run_enrichment_pipeline(
    expression_df: Optional[pd.DataFrame] = None,
    harmonized_genesets: Optional[Dict[str, List[str]]] = None,
    null_model_names: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Run enrichment analysis for all disorders and null models.

    Parameters
    ----------
    expression_df : pd.DataFrame, optional
        Group expression matrix. If None, loads from Phase 1 cache.
    harmonized_genesets : dict, optional
        Mapping from (disorder, geneset_def) to gene list.
        If None, loads from Phase 2 cache.
    null_model_names : list of str, optional
        Null models to use. If None, reads from params.yaml.
    params : dict, optional
        Pipeline parameters.
    output_dir : str or Path, optional
        Directory to save results.

    Returns
    -------
    pd.DataFrame
        Long-format results table with all disorders, gene set
        definitions, and null models.
    """
    if params is None:
        params = load_params()

    enrichment_params = params.get("enrichment", {})
    score_method = enrichment_params.get("score_method", "mean_zscore")
    fdr_alpha = enrichment_params.get("fdr_alpha", 0.05)
    seed = params.get("seed", 42)

    if null_model_names is None:
        null_model_names = enrichment_params.get(
            "null_models", ["naive", "matched", "spatial_coexpr"]
        )

    # Determine permutation count
    from vulnmap.utils import get_n_perm
    n_perm = get_n_perm(params)

    # Load expression data if not provided
    if expression_df is None:
        from vulnmap.utils import get_project_root, load_cached_dataframe
        cache_dir = get_project_root() / "data" / "interim"
        expression_df = load_cached_dataframe(cache_dir, "expression_desikan_killiany")
        if expression_df is None:
            raise FileNotFoundError(
                "Phase 1 expression matrix not found. Run Phase 1 first."
            )

    # Load gene sets if not provided
    if harmonized_genesets is None:
        from vulnmap.utils import get_project_root, load_cached_json
        cache_dir = get_project_root() / "data" / "interim"
        gs_data = load_cached_json(cache_dir, "harmonized_genesets")
        if gs_data is None:
            raise FileNotFoundError(
                "Phase 2 harmonized gene sets not found. Run Phase 2 first."
            )
        harmonized_genesets = gs_data

    # Setup output
    if output_dir is None:
        from vulnmap.utils import get_project_root
        output_dir = get_project_root() / "data" / "processed"
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    # Run enrichment for each (disorder, geneset_def, null_model)
    all_results = []

    for gs_key, gene_list in harmonized_genesets.items():
        # Parse key: "disorder__definition"
        if isinstance(gs_key, str) and "__" in gs_key:
            disorder, geneset_def = gs_key.split("__", 1)
        elif isinstance(gs_key, (list, tuple)) and len(gs_key) == 2:
            disorder, geneset_def = gs_key
        else:
            disorder = str(gs_key)
            geneset_def = "unknown"

        if len(gene_list) < 1:
            logger.warning(f"Skipping {disorder}/{geneset_def}: empty gene set")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Disorder: {disorder} | Definition: {geneset_def}")
        logger.info(f"Gene set size: {len(gene_list)}")
        logger.info(f"{'='*60}")

        for null_name in null_model_names:
            logger.info(f"  Null model: {null_name}")

            try:
                null_model = create_null_model(
                    name=null_name,
                    expression_df=expression_df,
                    n_perm=n_perm,
                    seed=seed,
                    params=params,
                )

                result = compute_enrichment(
                    expression_df=expression_df,
                    gene_set=gene_list,
                    null_model=null_model,
                    score_method=score_method,
                    fdr_alpha=fdr_alpha,
                )

                # Add metadata columns
                result.insert(0, "disorder", disorder)
                result.insert(1, "geneset_def", geneset_def)
                result.insert(2, "null_model", null_name)

                all_results.append(result)

            except Exception as e:
                logger.error(
                    f"  FAILED for {disorder}/{geneset_def}/{null_name}: {e}"
                )
                continue

    if not all_results:
        logger.warning("No enrichment results produced!")
        return pd.DataFrame()

    # Concatenate all results
    full_results = pd.concat(all_results, ignore_index=True)

    # Save
    results_path = output_dir / "enrichment_results.parquet"
    full_results.to_parquet(results_path, index=False)
    logger.info(f"\n[OK] Full enrichment results saved to {results_path}")
    logger.info(f"  Shape: {full_results.shape}")
    logger.info(f"  Disorders tested: {full_results['disorder'].nunique()}")
    logger.info(f"  Null models: {full_results['null_model'].unique().tolist()}")

    # Also save as CSV for convenience
    csv_path = output_dir / "enrichment_results.csv"
    full_results.to_csv(csv_path, index=False)

    return full_results


# =============================================================================
# Shrinkage summary
# =============================================================================

def compute_shrinkage_summary(
    results: pd.DataFrame,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute the "shrinkage" in significant region counts as null models
    become more stringent.

    Parameters
    ----------
    results : pd.DataFrame
        Full enrichment results from run_enrichment_pipeline().
    fdr_alpha : float
        FDR threshold.

    Returns
    -------
    pd.DataFrame
        Summary with columns: disorder, geneset_def, null_model,
        n_significant_fdr, n_significant_fwe, n_regions_total.
    """
    summary_rows = []
    for (disorder, gs_def, null_name), group in results.groupby(
        ["disorder", "geneset_def", "null_model"]
    ):
        summary_rows.append({
            "disorder": disorder,
            "geneset_def": gs_def,
            "null_model": null_name,
            "n_significant_fdr": (group["p_fdr"] < fdr_alpha).sum(),
            "n_significant_fwe": (group["p_fwe"] < fdr_alpha).sum(),
            "n_regions_total": len(group),
            "mean_z": group["z"].mean(),
            "max_z": group["z"].max(),
        })

    return pd.DataFrame(summary_rows)


# =============================================================================
# QC report
# =============================================================================

def enrichment_qc_report(
    results: pd.DataFrame,
    fdr_alpha: float = 0.05,
) -> str:
    """
    Generate a text QC report for enrichment results.

    Parameters
    ----------
    results : pd.DataFrame
        Output of run_enrichment_pipeline().
    fdr_alpha : float
        FDR threshold.

    Returns
    -------
    str
        Multi-line QC report.
    """
    lines = [
        "=" * 70,
        "VulnMap Phase 3: Enrichment QC Report",
        "=" * 70,
        "",
    ]

    if results.empty:
        lines.append("No results to report.")
        return "\n".join(lines)

    shrinkage = compute_shrinkage_summary(results, fdr_alpha)

    lines.extend([
        f"Total result rows: {len(results):,}",
        f"Disorders: {results['disorder'].nunique()}",
        f"Gene set definitions: {results['geneset_def'].nunique()}",
        f"Null models: {results['null_model'].nunique()}",
        "",
        "--- Shrinkage Summary (Naive -> Matched -> Spatial) ---",
        shrinkage.to_string(index=False),
        "",
        "--- Key Check: Does significance shrink? ---",
    ])

    # Check shrinkage pattern per (disorder, geneset_def)
    for (disorder, gs_def), d_shrink in shrinkage.groupby(["disorder", "geneset_def"]):
        null_order = ["naive", "matched", "spatial_coexpr", "spatial_surrogate"]
        d_shrink_ordered = d_shrink.set_index("null_model").reindex(
            [n for n in null_order if n in d_shrink["null_model"].values]
        )
        if len(d_shrink_ordered) > 1:
            fdr_counts = d_shrink_ordered["n_significant_fdr"].values
            label = f"{disorder} ({gs_def})"
            if all(fdr_counts[i] >= fdr_counts[i + 1] for i in range(len(fdr_counts) - 1)):
                lines.append(f"  [OK] {label}: monotonic shrinkage (expected)")
            else:
                lines.append(f"  [!] {label}: non-monotonic shrinkage (investigate)")

    lines.extend([
        "",
        "--- P-value Distributions ---",
        f"  p_emp range: [{results['p_emp'].min():.4f}, {results['p_emp'].max():.4f}]",
        f"  p_fdr range: [{results['p_fdr'].min():.4f}, {results['p_fdr'].max():.4f}]",
        f"  p_fwe range: [{results['p_fwe'].min():.4f}, {results['p_fwe'].max():.4f}]",
        "",
        "--- LIMITATIONS ---",
        "  1. p-values are empirical (resolution limited by n_perm).",
        "  2. Spatial null requires reliable distance/weight matrix.",
        "  3. Results depend on gene set definition (Catalog vs MAGMA).",
        "  4. FWE is conservative; FDR is more liberal.",
        "=" * 70,
    ])

    return "\n".join(lines)
