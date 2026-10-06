"""
VulnMap: Cross-Disorder Vulnerability & Clustering
===================================================

Phase 5b of the VulnMap pipeline.

This module investigates shared vs distinct patterns of regional brain
vulnerability across 15 psychiatric, neurodegenerative, and neurological
disorders.

Key components
--------------
1. **Cross-Disorder Matrix**: Builds a disorder x region vulnerability matrix
   from regional enrichment z-scores.
2. **Pairwise Similarity**: Computes Pearson correlation and Alexander-Bloch
   spatial spin-test p-values between all disorder pairs.
3. **Hierarchical Clustering**: Agglomerative clustering with correlation distance
   and average linkage, evaluating cophenetic correlation.
4. **PCA Decomposition**: Principal component analysis of vulnerability maps,
   identifying primary axes of brain vulnerability and regional loadings.
5. **Hypothesis 3 (H3) Permutation Test**: Tests whether within-psychiatric
   vulnerability similarity is significantly greater than psychiatric-vs-
   neurodegenerative similarity via label permutation (10,000 perms).

References
----------
- Cross-Disorder Group of the Psychiatric Genomics Consortium. (2019).
  Genomic relationships, novel loci, and pleiotropic mechanisms across eight
  psychiatric disorders. Cell, 179(7), 1469-1482.
- Gandal, M. J., et al. (2018). Shared molecular neuropathology across major
  psychiatric disorders parallels polygenic overlap. Science, 359(6376), 693-697.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from vulnmap.damage import (
    compute_damage_correlation,
    get_desikan_killiany_spherical_coords,
    spin_test_permutations,
)
from vulnmap.utils import (
    ensure_dir,
    get_project_root,
    load_disorders,
    load_params,
    logger,
    set_global_seed,
)


# =============================================================================
# Matrix Construction
# =============================================================================

def build_crossdisorder_matrix(
    enrichment_results: pd.DataFrame,
    geneset_def: Optional[str] = None,
    null_model: str = "spatial_coexpr",
) -> pd.DataFrame:
    """
    Build a disorders x regions vulnerability z-score matrix.

    Parameters
    ----------
    enrichment_results : pd.DataFrame
        Phase 3 enrichment results.
    geneset_def : str, optional
        Gene set definition to use (e.g. 'catalog_5e-8'). If None, prefers
        'catalog_5e-8', falling back to first available definition per disorder.
    null_model : str
        Null model results to use. Default: 'spatial_coexpr'.

    Returns
    -------
    pd.DataFrame
        Index: disorder names. Columns: brain region names.
        Values: enrichment z-scores.
    """
    df = enrichment_results[enrichment_results["null_model"] == null_model].copy()
    if df.empty:
        # Fall back to any available null model
        df = enrichment_results.copy()
        logger.warning(f"Null model '{null_model}' not found in enrichment results. Using all available.")

    disorders = df["disorder"].unique()
    rows = []

    for d in disorders:
        d_df = df[df["disorder"] == d]
        if geneset_def is not None and geneset_def in d_df["geneset_def"].values:
            sub = d_df[d_df["geneset_def"] == geneset_def]
        elif "catalog_5e-8" in d_df["geneset_def"].values:
            sub = d_df[d_df["geneset_def"] == "catalog_5e-8"]
        else:
            first_def = d_df["geneset_def"].iloc[0]
            sub = d_df[d_df["geneset_def"] == first_def]

        # Region series
        series = sub.set_index("region")["z"]
        series.name = d
        rows.append(series)

    cross_df = pd.DataFrame(rows)
    # Fill remaining NaNs with 0.0
    cross_df = cross_df.fillna(0.0)
    logger.info(f"Built cross-disorder matrix: {cross_df.shape[0]} disorders x {cross_df.shape[1]} regions.")
    return cross_df


# =============================================================================
# Pairwise Similarity & Spin Testing
# =============================================================================

def compute_pairwise_similarity(
    cross_matrix: pd.DataFrame,
    spin_test: bool = True,
    n_perm: int = 1000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise Pearson correlations and spin-test p-values between disorders.

    Parameters
    ----------
    cross_matrix : pd.DataFrame
        Disorders x regions matrix.
    spin_test : bool
        If True, computes spatial spin-test p-values for each pair.
    n_perm : int
        Number of spin permutations.
    seed : int
        Random seed.

    Returns
    -------
    sim_matrix : pd.DataFrame
        Disorders x disorders Pearson r matrix.
    pval_matrix : pd.DataFrame
        Disorders x disorders spin-test empirical p-values.
    """
    disorders = list(cross_matrix.index)
    n_d = len(disorders)
    regions = list(cross_matrix.columns)

    # 1. Pearson correlation matrix
    r_matrix = np.corrcoef(cross_matrix.values)
    np.fill_diagonal(r_matrix, 1.0)
    sim_df = pd.DataFrame(r_matrix, index=disorders, columns=disorders)

    # 2. Spin test p-values
    pval_matrix = np.ones((n_d, n_d), dtype=np.float64)
    np.fill_diagonal(pval_matrix, 0.0)

    if spin_test and n_d > 1:
        coords_df = get_desikan_killiany_spherical_coords()
        # Find common regions that have spherical coordinates
        common_regs = [r for r in regions if r in coords_df.index]

        if len(common_regs) >= 10:
            coords = coords_df.loc[common_regs].values
            spin_perms = spin_test_permutations(coords, n_perm=n_perm, seed=seed)

            for i in range(n_d):
                x = cross_matrix.loc[disorders[i], common_regs].values
                for j in range(i + 1, n_d):
                    y = cross_matrix.loc[disorders[j], common_regs].values
                    r_obs = stats.pearsonr(x, y)[0]

                    # Permutation null
                    null_rs = np.zeros(n_perm)
                    for p in range(n_perm):
                        x_perm = x[spin_perms[p]]
                        null_rs[p] = stats.pearsonr(x_perm, y)[0]

                    p_val = (np.sum(np.abs(null_rs) >= np.abs(r_obs)) + 1.0) / (n_perm + 1.0)
                    pval_matrix[i, j] = p_val
                    pval_matrix[j, i] = p_val
        else:
            # Fallback to parametric p-values
            for i in range(n_d):
                for j in range(i + 1, n_d):
                    p_val = stats.pearsonr(cross_matrix.iloc[i], cross_matrix.iloc[j])[1]
                    pval_matrix[i, j] = p_val
                    pval_matrix[j, i] = p_val

    pval_df = pd.DataFrame(pval_matrix, index=disorders, columns=disorders)
    return sim_df, pval_df


# =============================================================================
# Hierarchical Clustering
# =============================================================================

def hierarchical_clustering(
    cross_matrix: pd.DataFrame,
    metric: str = "correlation",
    method: str = "average",
) -> Dict[str, Any]:
    """
    Perform hierarchical agglomerative clustering of disorders.

    Parameters
    ----------
    cross_matrix : pd.DataFrame
        Disorders x regions matrix.
    metric : str
        Distance metric ('correlation' or 'euclidean').
    method : str
        Linkage criterion ('average', 'complete', 'ward').

    Returns
    -------
    dict
        - 'linkage': linkage matrix Z
        - 'cophenetic_corr': cophenetic correlation coefficient
        - 'ordered_disorders': list of disorders in dendrogram order
        - 'distance_matrix': squareform distance matrix
    """
    disorders = list(cross_matrix.index)
    if len(disorders) < 2:
        return {
            "linkage": np.array([]),
            "cophenetic_corr": 1.0,
            "ordered_disorders": disorders,
            "distance_matrix": pd.DataFrame([[0.0]], index=disorders, columns=disorders),
        }

    # Pairwise condensed distance
    condensed_dist = pdist(cross_matrix.values, metric=metric)
    # Ensure non-negative distances (handle precision issues)
    condensed_dist = np.clip(condensed_dist, 0.0, 2.0)

    # Linkage matrix
    Z = hierarchy.linkage(condensed_dist, method=method)

    # Cophenetic correlation
    coph_corr, _ = hierarchy.cophenet(Z, condensed_dist)

    # Leaf order
    dendro = hierarchy.dendrogram(Z, no_plot=True)
    ordered_indices = dendro["leaves"]
    ordered_disorders = [disorders[i] for i in ordered_indices]

    dist_matrix = pd.DataFrame(
        squareform(condensed_dist), index=disorders, columns=disorders
    )

    logger.info(
        f"Hierarchical clustering complete (method={method}, metric={metric}). "
        f"Cophenetic r = {coph_corr:.3f}"
    )

    return {
        "linkage": Z,
        "cophenetic_corr": float(coph_corr),
        "ordered_disorders": ordered_disorders,
        "distance_matrix": dist_matrix,
    }


# =============================================================================
# PCA Decomposition
# =============================================================================

def crossdisorder_pca(
    cross_matrix: pd.DataFrame,
    n_components: int = 3,
) -> Dict[str, Any]:
    """
    Perform Principal Component Analysis across disorders.

    Parameters
    ----------
    cross_matrix : pd.DataFrame
        Disorders x regions matrix.
    n_components : int
        Number of principal components to extract.

    Returns
    -------
    dict
        - 'scores': pd.DataFrame of disorder coordinates in PC space
        - 'loadings': pd.DataFrame of region contributions to each PC
        - 'explained_variance_ratio': np.ndarray of variance per PC
        - 'top_regions_pc1': dict of top positive/negative loading regions for PC1
        - 'top_regions_pc2': dict of top positive/negative loading regions for PC2
    """
    n_d, n_r = cross_matrix.shape
    k = min(n_components, n_d, n_r)

    pca = PCA(n_components=k)
    scores_arr = pca.fit_transform(cross_matrix.values)

    pc_names = [f"PC{i + 1}" for i in range(k)]
    scores_df = pd.DataFrame(scores_arr, index=cross_matrix.index, columns=pc_names)

    # Loadings (regions x PCs)
    loadings_df = pd.DataFrame(pca.components_.T, index=cross_matrix.columns, columns=pc_names)

    # Top drivers for PC1 and PC2
    top_drivers = {}
    for pc_i, pc_col in enumerate(pc_names[:2]):
        sorted_loadings = loadings_df[pc_col].sort_values()
        top_drivers[pc_col] = {
            "top_positive": sorted_loadings.tail(5).to_dict(),
            "top_negative": sorted_loadings.head(5).to_dict(),
        }

    return {
        "scores": scores_df,
        "loadings": loadings_df,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "top_drivers": top_drivers,
    }


# =============================================================================
# Hypothesis 3 (H3) Permutation Test
# =============================================================================

def evaluate_category_similarity_h3(
    sim_matrix: pd.DataFrame,
    category_mapping: Optional[Dict[str, str]] = None,
    n_perm: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Permutation test for Hypothesis 3 (H3):
    Is within-psychiatric similarity significantly greater than
    psychiatric-vs-neurodegenerative similarity?

    Test statistic:
        Delta = mean(r_within_psychiatric) - mean(r_psych_vs_neuro)

    Parameters
    ----------
    sim_matrix : pd.DataFrame
        Disorders x disorders correlation matrix.
    category_mapping : dict, optional
        Mapping from disorder name to category ('psychiatric' or 'neurodegenerative').
        If None, loads categories from config/disorders.yaml.
    n_perm : int
        Number of label permutations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        - 'delta_obs': observed difference in mean correlation
        - 'mean_within_psychiatric': observed within-psychiatric mean r
        - 'mean_psych_vs_neuro': observed psychiatric vs neurodegenerative mean r
        - 'p_value': empirical permutation p-value
        - 'n_perm': number of permutations executed
        - 'supported': bool (True if p_value < 0.05 and delta_obs > 0)
    """
    rng = np.random.default_rng(seed)

    if category_mapping is None:
        try:
            disorders_cfg = load_disorders()
            category_mapping = {
                d: cfg.get("category", "other").lower()
                for d, cfg in disorders_cfg.get("disorders", {}).items()
            }
        except Exception:
            category_mapping = {}

    # Identify disorders present in matrix
    disorders = list(sim_matrix.index)
    categories = [category_mapping.get(d, "other").lower() for d in disorders]

    psych_indices = [i for i, c in enumerate(categories) if c == "psychiatric"]
    neuro_indices = [i for i, c in enumerate(categories) if c == "neurodegenerative"]

    # If insufficient categories in test set, return inconclusive
    if len(psych_indices) < 2 or len(neuro_indices) < 1:
        logger.warning(
            f"Insufficient categories for H3 test (psych={len(psych_indices)}, neuro={len(neuro_indices)}). "
            f"At least 2 psychiatric and 1 neurodegenerative required."
        )
        return {
            "delta_obs": 0.0,
            "mean_within_psychiatric": np.nan,
            "mean_psych_vs_neuro": np.nan,
            "p_value": 1.0,
            "n_perm": n_perm,
            "supported": False,
            "status": "Inconclusive (insufficient category coverage)",
        }

    R = sim_matrix.values

    # Helper function to compute statistic
    def _compute_delta(psych_idx, neuro_idx):
        within_vals = []
        for i in range(len(psych_idx)):
            for j in range(i + 1, len(psych_idx)):
                within_vals.append(R[psych_idx[i], psych_idx[j]])

        between_vals = []
        for i in psych_idx:
            for j in neuro_idx:
                between_vals.append(R[i, j])

        mean_w = np.mean(within_vals) if within_vals else 0.0
        mean_b = np.mean(between_vals) if between_vals else 0.0
        return mean_w - mean_b, mean_w, mean_b

    delta_obs, mean_within, mean_between = _compute_delta(psych_indices, neuro_indices)

    # Permutation test: shuffle category labels among tested disorders
    combined_idx = psych_indices + neuro_indices
    n_psych = len(psych_indices)
    null_deltas = np.zeros(n_perm)

    for p in range(n_perm):
        perm_idx = rng.permutation(combined_idx)
        p_psych = perm_idx[:n_psych]
        p_neuro = perm_idx[n_psych:]
        null_deltas[p], _, _ = _compute_delta(p_psych, p_neuro)

    # One-sided test (H3: within > between, i.e. Delta > 0)
    b = np.sum(null_deltas >= delta_obs)
    p_val = (b + 1.0) / (n_perm + 1.0)

    supported = bool((p_val < 0.05) and (delta_obs > 0.0))

    return {
        "delta_obs": float(delta_obs),
        "mean_within_psychiatric": float(mean_within),
        "mean_psych_vs_neuro": float(mean_between),
        "p_value": float(p_val),
        "n_perm": n_perm,
        "supported": supported,
        "status": "Supported" if supported else "Not Supported (or Inconclusive)",
    }


test_category_similarity_h3 = evaluate_category_similarity_h3


# =============================================================================
# Pipeline Runner & QC
# =============================================================================

def run_crossdisorder_pipeline(
    enrichment_results: Optional[pd.DataFrame] = None,
    params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Run full cross-disorder clustering, PCA, and H3 testing.
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

    seed = params.get("seed", 42)
    h3_n_perm = params.get("crossdisorder", {}).get("category_test_n_perm", 1000)

    # 1. Build matrix
    cross_matrix = build_crossdisorder_matrix(enrichment_results)

    # 2. Pairwise similarity & spin test
    sim_df, pval_df = compute_pairwise_similarity(cross_matrix, n_perm=500, seed=seed)

    # 3. Hierarchical clustering
    cluster_res = hierarchical_clustering(cross_matrix)

    # 4. PCA
    pca_res = crossdisorder_pca(cross_matrix, n_components=min(3, len(cross_matrix)))

    # 5. H3 Permutation test
    h3_res = test_category_similarity_h3(sim_df, n_perm=h3_n_perm, seed=seed)

    # Save outputs
    cross_matrix.to_parquet(output_dir / "crossdisorder_matrix.parquet")
    sim_df.to_parquet(output_dir / "crossdisorder_similarity.parquet")
    sim_df.to_csv(output_dir / "crossdisorder_similarity.csv")

    logger.info(f"[OK] Cross-disorder matrices saved to {output_dir}")

    return {
        "cross_matrix": cross_matrix,
        "sim_matrix": sim_df,
        "pval_matrix": pval_df,
        "clustering": cluster_res,
        "pca": pca_res,
        "h3_test": h3_res,
    }


def crossdisorder_qc_report(results: Dict[str, Any]) -> str:
    """Format a text QC report for cross-disorder analysis."""
    lines = [
        "=" * 70,
        "VulnMap Phase 5b: Cross-Disorder Similarity & H3 Test QC Report",
        "=" * 70,
        "",
    ]

    sim_df = results["sim_matrix"]
    lines.extend([
        f"Disorders analyzed: {len(sim_df)}",
        f"Mean pairwise correlation: {sim_df.values[np.triu_indices_from(sim_df.values, k=1)].mean():.3f}",
        f"Hierarchical clustering cophenetic r: {results['clustering']['cophenetic_corr']:.3f}",
        f"PC1 explained variance: {results['pca']['explained_variance_ratio'][0] * 100:.1f}%",
        "",
        "--- Pairwise Correlation Matrix ---",
        sim_df.round(3).to_string(),
        "",
        "--- Hypothesis 3 (H3) Test: Psychiatric vs Neurodegenerative ---",
        f"  Within-psychiatric mean r: {results['h3_test']['mean_within_psychiatric']:.3f}",
        f"  Psychiatric vs Neurodegenerative mean r: {results['h3_test']['mean_psych_vs_neuro']:.3f}",
        f"  Delta (Observed difference): {results['h3_test']['delta_obs']:.3f}",
        f"  Permutation p-value (10,000 perms): {results['h3_test']['p_value']:.4f}",
        f"  Outcome: {results['h3_test']['status']}",
        "=" * 70,
    ])

    return "\n".join(lines)


# Alias for Makefile and external invocations
run_crossdisorder = run_crossdisorder_pipeline
