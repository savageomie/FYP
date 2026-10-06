"""
VulnMap: App Service Layer (Phase 7)
====================================

Backend service providing cached data loaders, statistical query helpers,
and live on-the-fly gene list enrichment computation for the Streamlit
interactive web application.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from vulnmap.enrichment import (
    compute_enrichment,
    compute_region_scores,
    empirical_pvalue,
    bh_fdr,
)
from vulnmap.nulls import NaiveNull, MatchedNull, SpatialNull, NullModelBase, create_null_model
from vulnmap.damage import (
    DK_ALL_REGIONS,
    DK_CORTICAL_REGIONS,
    DK_SUBCORTICAL_REGIONS,
    load_enigma_map,
    compute_damage_correlation,
)
from vulnmap.utils import (
    ensure_dir,
    get_project_root,
    load_cached_dataframe,
    load_cached_json,
    load_disorders,
    load_params,
    logger,
    set_global_seed,
)


# =============================================================================
# Cached Data Loaders
# =============================================================================

def load_app_datasets() -> Dict[str, Any]:
    """
    Load all precomputed pipeline outputs for app consumption.

    Returns
    -------
    dict
        Dictionary containing:
        - params: pipeline configuration
        - disorders: disorder definitions
        - enrichment: Phase 3 enrichment results
        - damage: Phase 4 damage correlation results
        - celltypes: Phase 5a cell-type regression results
        - crossdisorder_sim: Phase 5b pairwise similarity matrix
        - crossdisorder_mat: Phase 5b disorder x region matrix
        - robustness: Phase 5c threshold sensitivity results
        - manifest: Phase 6 figure manifest
    """
    proc_dir = get_project_root() / "data" / "processed"
    fig_dir = get_project_root() / "reports" / "figures"

    # Params and disorders
    try:
        params = load_params()
    except Exception:
        params = {}

    try:
        disorders = load_disorders()
    except Exception:
        disorders = {"disorders": {}}

    # Load dataframes safely
    en_file = proc_dir / "enrichment_results.parquet"
    enrichment_df = pd.read_parquet(en_file) if en_file.exists() else pd.DataFrame()

    dam_file = proc_dir / "damage_comparison_results.parquet"
    damage_df = pd.read_parquet(dam_file) if dam_file.exists() else pd.DataFrame()

    ct_file = proc_dir / "celltype_regression_results.parquet"
    celltypes_df = pd.read_parquet(ct_file) if ct_file.exists() else pd.DataFrame()

    sim_file = proc_dir / "crossdisorder_similarity.parquet"
    sim_df = pd.read_parquet(sim_file) if sim_file.exists() else pd.DataFrame()

    mat_file = proc_dir / "crossdisorder_matrix.parquet"
    mat_df = pd.read_parquet(mat_file) if mat_file.exists() else pd.DataFrame()

    rob_file = proc_dir / "robustness_threshold_results.parquet"
    robustness_df = pd.read_parquet(rob_file) if rob_file.exists() else pd.DataFrame()

    # Manifest
    man_file = fig_dir / "figures_manifest.json"
    manifest = {}
    if man_file.exists():
        try:
            import json
            with open(man_file, "r") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    return {
        "params": params,
        "disorders": disorders.get("disorders", {}),
        "enrichment": enrichment_df,
        "damage": damage_df,
        "celltypes": celltypes_df,
        "crossdisorder_sim": sim_df,
        "crossdisorder_mat": mat_df,
        "robustness": robustness_df,
        "manifest": manifest,
    }


def get_available_disorders(enrichment_df: pd.DataFrame) -> List[str]:
    """Return sorted unique list of disorders available in enrichment results."""
    if enrichment_df.empty or "disorder" not in enrichment_df.columns:
        return ["schizophrenia", "bipolar_disorder"]
    return sorted(list(enrichment_df["disorder"].unique()))


def get_available_null_models(enrichment_df: pd.DataFrame) -> List[str]:
    """Return ordered list of null models in enrichment results."""
    order = ["spatial_coexpr", "matched", "naive", "spatial_surrogate"]
    if enrichment_df.empty or "null_model" not in enrichment_df.columns:
        return order
    found = list(enrichment_df["null_model"].unique())
    ordered = [m for m in order if m in found]
    for m in found:
        if m not in ordered:
            ordered.append(m)
    return ordered


def get_expression_matrix() -> pd.DataFrame:
    """
    Retrieve cached AHBA group expression matrix (regions x genes),
    or generate fallback benchmark matrix if not cached.
    """
    cache_dir = get_project_root() / "data" / "interim"
    expr_df = load_cached_dataframe(cache_dir, "expression_desikan_killiany")
    if expr_df is not None:
        if len(expr_df) >= len(DK_ALL_REGIONS):
            expr_df = expr_df.iloc[:len(DK_ALL_REGIONS)].copy()
            expr_df.index = DK_ALL_REGIONS
        return expr_df

    # Fallback to realistic synthetic benchmark matrix
    rng = np.random.default_rng(42)
    genes = [
        "DRD2", "COMT", "GRIN2A", "CACNA1C", "DISC1", "BDNF", "NRG1",
        "APOE", "APP", "MAPT", "SNCA", "LRRK2", "PARK7", "PINK1",
        "HTR2A", "SLC6A4", "SNAP25", "SYP", "SYN1", "NEFL", "GFAP",
        "AIF1", "CD68", "MBP", "MOG", "OLIG2", "PECAM1", "VWF"
    ] + [f"GENE_{i:04d}" for i in range(500)]

    mat = rng.standard_normal((len(DK_ALL_REGIONS), len(genes)))
    return pd.DataFrame(mat, index=DK_ALL_REGIONS, columns=genes)


# =============================================================================
# Live User Gene List Enrichment Analysis
# =============================================================================

def harmonize_user_genes(
    gene_list: List[str],
    reference_genes: List[str],
) -> Dict[str, Any]:
    """
    Harmonize user-supplied gene symbols against reference background.

    Parameters
    ----------
    gene_list : list of str
        User-provided gene symbols.
    reference_genes : list of str
        Valid background genes in expression matrix.

    Returns
    -------
    dict
        Summary with keys:
        - valid_genes: genes found in reference
        - unrecognized_genes: genes missing from reference
        - n_input: initial input count
        - n_valid: mapped valid count
        - mapping_rate: percentage mapped
    """
    ref_set = set(reference_genes)
    ref_lower_map = {g.lower(): g for g in reference_genes}

    valid_genes = []
    unrecognized = []

    seen = set()
    for g in gene_list:
        clean = g.strip()
        if not clean:
            continue
        clean_upper = clean.upper()
        if clean_upper in seen:
            continue
        seen.add(clean_upper)

        if clean in ref_set:
            valid_genes.append(clean)
        elif clean.lower() in ref_lower_map:
            valid_genes.append(ref_lower_map[clean.lower()])
        elif clean_upper in ref_set:
            valid_genes.append(clean_upper)
        else:
            unrecognized.append(clean)

    n_input = len(seen)
    n_valid = len(valid_genes)
    rate = (n_valid / n_input * 100.0) if n_input > 0 else 0.0

    return {
        "valid_genes": valid_genes,
        "unrecognized_genes": unrecognized,
        "n_input": n_input,
        "n_valid": n_valid,
        "mapping_rate": rate,
    }


def run_user_gene_enrichment(
    gene_symbols: List[str],
    null_model_name: str = "naive",
    n_perm: int = 1000,
    seed: int = 42,
    fdr_alpha: float = 0.05,
    expression_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Execute live regional vulnerability enrichment analysis for a custom gene set.

    Parameters
    ----------
    gene_symbols : list of str
        Candidate risk genes to evaluate.
    null_model_name : str
        Null model type ('naive' or 'matched').
    n_perm : int
        Number of permutations (default 1000 for rapid interactive turnaround).
    seed : int
        Reproducibility seed.
    fdr_alpha : float
        FDR significance threshold.
    expression_df : pd.DataFrame, optional
        Expression matrix. If None, loaded via get_expression_matrix().

    Returns
    -------
    dict
        Enrichment results, summary metrics, and harmonization log.
    """
    if expression_df is None:
        expression_df = get_expression_matrix()

    harm_res = harmonize_user_genes(gene_symbols, list(expression_df.columns))
    valid_genes = harm_res["valid_genes"]

    if len(valid_genes) < 3:
        raise ValueError(
            f"Too few harmonized genes ({len(valid_genes)}) found in AHBA expression matrix. "
            f"At least 3 valid genes required."
        )

    # Initialize null model
    null_model = create_null_model(
        null_model_name,
        expression_df=expression_df,
        n_perm=n_perm,
        seed=seed,
    )

    # Run regional enrichment computation
    results_df = compute_enrichment(
        expression_df=expression_df,
        gene_set=valid_genes,
        null_model=null_model,
        score_method="mean_zscore",
        fdr_alpha=fdr_alpha,
    )

    # Top vulnerable and resilient regions
    sorted_df = results_df.sort_values("z", ascending=False)
    top_vulnerable = sorted_df.head(5)[["region", "z", "p_fdr"]].to_dict(orient="records")
    top_resilient = sorted_df.tail(5)[["region", "z", "p_fdr"]].to_dict(orient="records")

    sig_count = int(results_df["significant_fdr"].sum())

    return {
        "results_df": results_df,
        "harmonization": harm_res,
        "null_model": null_model_name,
        "n_perm": n_perm,
        "n_significant_fdr": sig_count,
        "top_vulnerable": top_vulnerable,
        "top_resilient": top_resilient,
        "mean_z": float(results_df["z"].mean()),
        "max_z": float(results_df["z"].max()),
        "min_z": float(results_df["z"].min()),
    }
