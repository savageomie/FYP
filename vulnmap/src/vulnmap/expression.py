"""
VulnMap: Expression Matrix Construction & Quality Control
=========================================================

Phase 1 of the VulnMap pipeline.

This module builds regional transcriptomic matrices from the
Allen Human Brain Atlas (AHBA) using the ``abagen`` processing framework.

Scientific context
------------------
- AHBA provides microarray data from 6 donors (Hawrylycz et al., 2012, 2015).
- ``abagen`` (Markello et al., 2021; Arnatkevičiūtė et al., 2019) implements
  a standardized processing pipeline with validated defaults.
- Desikan-Killiany is the primary parcellation because it matches ENIGMA
  Toolbox damage maps, enabling direct comparison without re-parcellation.
- Schaefer-200 is the robustness parcellation (finer cortical resolution).

Key caveats (encoded as warnings)
----------------------------------
- Only 2 of 6 donors have right-hemisphere samples. Default: lr_mirror='bidirectional'.
- Subcortical, brainstem and cerebellar sampling is sparse.
  Substantia nigra may be combined into a broader midbrain label.
- AHBA is healthy adult tissue: results show where risk genes normally act,
  not disease-state expression.
- Differential stability (DS) filtering removes genes with inconsistent
  expression across donors. Default threshold: DS > 0.1.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from vulnmap.utils import (
    Timer,
    cache_dataframe,
    cache_json,
    ensure_dir,
    load_cached_dataframe,
    load_cached_json,
    load_params,
    log_run,
    logger,
    set_global_seed,
)

# Monkeypatch pandas DataFrame.set_axis to handle inplace argument removed in pandas >= 2.1
try:
    _orig_set_axis = pd.DataFrame.set_axis

    def _compat_set_axis(self, labels, *args, **kwargs):
        kwargs.pop("inplace", None)
        return _orig_set_axis(self, labels, *args, **kwargs)

    pd.DataFrame.set_axis = _compat_set_axis
except Exception:
    pass

# Monkeypatch pandas DataFrame.append removed in pandas >= 2.0
if not hasattr(pd.DataFrame, "append"):
    def _df_append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        if other is None:
            return self
        to_concat = [self] + (list(other) if isinstance(other, (list, tuple)) else [other])
        return pd.concat(to_concat, ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort)

    pd.DataFrame.append = _df_append



# =============================================================================
# Atlas / parcellation helpers
# =============================================================================

def get_parcellation_info(parcellation: str) -> Dict[str, Any]:
    """
    Return metadata about a supported parcellation.

    Parameters
    ----------
    parcellation : str
        One of 'desikan_killiany' or 'schaefer_200'.

    Returns
    -------
    dict
        Keys: 'name', 'description', 'atlas_source', 'expected_n_regions',
        'includes_subcortical'.

    Raises
    ------
    ValueError
        If parcellation is not recognized.
    """
    info = {
        "desikan_killiany": {
            "name": "Desikan-Killiany",
            "description": (
                "68 cortical regions (34 per hemisphere) from the FreeSurfer "
                "Desikan-Killiany atlas, plus ~14 subcortical structures "
                "(FreeSurfer aseg). Primary parcellation chosen to match "
                "ENIGMA Toolbox case-control maps."
            ),
            "atlas_source": "fetched via abagen (FreeSurfer aparc + aseg)",
            "expected_n_regions_range": (68, 85),  # cortical only to cortical+subcortical
            "includes_subcortical": True,
        },
        "schaefer_200": {
            "name": "Schaefer-200",
            "description": (
                "200 cortical parcels from Schaefer et al., 2018 "
                "(7-network resolution). Robustness parcellation for "
                "testing parcellation dependence."
            ),
            "atlas_source": "fetched via abagen/nilearn",
            "expected_n_regions_range": (200, 220),
            "includes_subcortical": False,
        },
    }
    if parcellation not in info:
        raise ValueError(
            f"Unknown parcellation: '{parcellation}'. "
            f"Supported: {list(info.keys())}"
        )
    return info[parcellation]


# =============================================================================
# Expression matrix construction
# =============================================================================

def build_expression(
    parcellation: str = "desikan_killiany",
    probe_selection: str = "diff_stability",
    normalization: str = "scaled_robust_sigmoid",
    lr_mirror: str = "bidirectional",
    return_donors: bool = True,
    ds_threshold: float = 0.1,
    data_dir: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    force_recompute: bool = False,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the regional expression matrix from AHBA data using abagen.

    This is the main entry point for Phase 1. It:
    1. Fetches parcellation labels (atlas) for abagen.
    2. Calls abagen.get_expression_data() with the specified parameters.
    3. Computes differential stability (DS) and filters genes.
    4. Builds a coverage table (samples per region per donor).
    5. Caches all outputs.

    Parameters
    ----------
    parcellation : str
        'desikan_killiany' or 'schaefer_200'.
    probe_selection : str
        abagen probe selection method. Default: 'diff_stability'.
    normalization : str
        abagen normalization method. Default: 'scaled_robust_sigmoid'.
    lr_mirror : str
        How to handle left/right hemisphere. Default: 'bidirectional'.
        CAVEAT: Only 2/6 donors have right-hemisphere data.
    return_donors : bool
        If True, also return per-donor expression matrices.
    ds_threshold : float
        Minimum differential stability to retain a gene. Default: 0.1.
    data_dir : str or Path, optional
        Directory for raw AHBA data. If None, uses abagen default.
    cache_dir : str or Path, optional
        Directory for cached outputs. If None, uses data/interim/.
    force_recompute : bool
        If True, ignore cached results and recompute.
    params : dict, optional
        Full params dict (overrides individual arguments).

    Returns
    -------
    dict
        Keys:
        - 'expression': pd.DataFrame (regions × genes), group-level
        - 'donor_expressions': list of pd.DataFrame (if return_donors)
        - 'coverage': pd.DataFrame (regions × ['n_samples', 'n_donors', ...])
        - 'gene_info': pd.DataFrame (gene × ['mean_expr', 'ds', 'kept'])
        - 'parcellation_info': dict
        - 'params_used': dict
    """
    # Import abagen here to give a clear error if not installed
    try:
        import abagen
    except ImportError:
        raise ImportError(
            "abagen is required for Phase 1. Install with:\n"
            "  pip install abagen\n"
            "If on Windows and installation fails, try WSL or conda."
        )

    # Override from params dict if provided
    if params is not None:
        expr_params = params.get("expression", {})
        parcellation = expr_params.get("parcellation", parcellation)
        probe_selection = expr_params.get("probe_selection", probe_selection)
        normalization = expr_params.get("normalization", normalization)
        lr_mirror = expr_params.get("lr_mirror", lr_mirror)
        return_donors = expr_params.get("return_donors", return_donors)
        ds_threshold = expr_params.get("ds_threshold", ds_threshold)

    # Setup paths
    if cache_dir is None:
        from vulnmap.utils import get_project_root
        cache_dir = get_project_root() / "data" / "interim"
    cache_dir = Path(cache_dir)
    cache_name = f"expression_{parcellation}"

    # Check cache
    if not force_recompute:
        cached = load_cached_dataframe(cache_dir, cache_name)
        if cached is not None:
            logger.info(f"Using cached expression matrix for {parcellation}")
            # Also load cached metadata
            meta = load_cached_json(cache_dir, f"{cache_name}_meta")
            gene_info = load_cached_dataframe(cache_dir, f"{cache_name}_gene_info")
            coverage = load_cached_dataframe(cache_dir, f"{cache_name}_coverage")
            donor_dfs = []
            if return_donors:
                i = 0
                while True:
                    d = load_cached_dataframe(cache_dir, f"{cache_name}_donor_{i}")
                    if d is None:
                        break
                    donor_dfs.append(d)
                    i += 1
            return {
                "expression": cached,
                "donor_expressions": donor_dfs if return_donors else None,
                "coverage": coverage,
                "gene_info": gene_info,
                "parcellation_info": get_parcellation_info(parcellation),
                "params_used": meta if meta else {},
            }

    logger.info(f"Building expression matrix for parcellation: {parcellation}")
    parc_info = get_parcellation_info(parcellation)

    # --- Fetch the atlas for abagen ---
    atlas_img, atlas_info = _fetch_atlas(parcellation)

    # --- Build abagen kwargs ---
    abagen_kwargs = {
        "probe_selection": probe_selection,
        "lr_mirror": lr_mirror,
        "norm_matched": True if normalization == "scaled_robust_sigmoid" else False,
        "norm_structures": True if normalization == "scaled_robust_sigmoid" else False,
        "return_donors": return_donors,
    }
    # abagen's normalization parameter name depends on version
    # The 'scaled_robust_sigmoid' is the default in abagen >= 0.1
    # We pass sample_norm and gene_norm explicitly
    abagen_kwargs["sample_norm"] = "scaled_robust_sigmoid"
    abagen_kwargs["gene_norm"] = "scaled_robust_sigmoid"

    if data_dir is not None:
        abagen_kwargs["data_dir"] = str(data_dir)

    logger.info(f"Calling abagen.get_expression_data() with: {abagen_kwargs}")
    logger.warning(
        "CAVEAT: AHBA has 6 donors; only 2 have right-hemisphere samples. "
        f"Using lr_mirror='{lr_mirror}'."
    )
    logger.warning(
        "CAVEAT: AHBA is healthy adult tissue. Expression patterns reflect "
        "normal gene activity, not disease-state changes."
    )

    with Timer("abagen.get_expression_data"):
        result = abagen.get_expression_data(
            atlas_img,
            atlas_info=atlas_info,
            **abagen_kwargs,
        )

    # --- Parse results ---
    if return_donors:
        # abagen returns a dict of DataFrames when return_donors=True
        # The group expression is the first element, donors follow
        if isinstance(result, tuple) and len(result) == 2:
            group_expr, donor_dict = result
        elif isinstance(result, dict):
            # Some abagen versions return dict keyed by donor ID
            donor_dict = result
            # Compute group mean
            all_donor_dfs = list(result.values())
            group_expr = pd.concat(all_donor_dfs).groupby(level=0).mean()
        elif isinstance(result, pd.DataFrame):
            # return_donors might not split in some versions
            group_expr = result
            donor_dict = {}
            logger.warning(
                "abagen returned a single DataFrame despite return_donors=True. "
                "Per-donor matrices not available. Check abagen version."
            )
        else:
            raise TypeError(f"Unexpected abagen output type: {type(result)}")

        if isinstance(donor_dict, dict):
            donor_dfs = list(donor_dict.values())
        elif isinstance(donor_dict, (list, tuple)):
            donor_dfs = list(donor_dict)
        else:
            donor_dfs = []
    else:
        group_expr = result
        donor_dfs = []

    logger.info(f"Group expression matrix shape: {group_expr.shape}")
    logger.info(f"  Regions: {group_expr.shape[0]}, Genes: {group_expr.shape[1]}")
    if donor_dfs:
        logger.info(f"  Per-donor matrices: {len(donor_dfs)}")

    # --- Quality checks ---
    n_regions, n_genes = group_expr.shape

    # Check for all-NaN regions
    all_nan_regions = group_expr.index[group_expr.isna().all(axis=1)].tolist()
    if all_nan_regions:
        logger.warning(
            f"WARNING: {len(all_nan_regions)} regions have ALL NaN values: "
            f"{all_nan_regions}. These will be excluded from downstream analysis."
        )

    # Check for all-NaN genes
    all_nan_genes = group_expr.columns[group_expr.isna().all(axis=0)].tolist()
    if all_nan_genes:
        logger.warning(
            f"WARNING: {len(all_nan_genes)} genes have ALL NaN values "
            f"(removed by abagen QC)."
        )

    # --- Differential Stability ---
    gene_info = _compute_gene_info(group_expr, donor_dfs, ds_threshold)
    logger.info(
        f"Differential stability filtering (threshold={ds_threshold}): "
        f"{gene_info['kept'].sum()}/{len(gene_info)} genes retained"
    )

    # Filter expression matrix to retained genes
    kept_genes = gene_info.index[gene_info["kept"]].tolist()
    group_expr_filtered = group_expr[kept_genes]
    donor_dfs_filtered = [d[kept_genes] for d in donor_dfs] if donor_dfs else []

    # --- Coverage table ---
    coverage = _build_coverage_table(group_expr, donor_dfs, parcellation)
    low_coverage = coverage[
        coverage["n_samples_total"] < 5  # Hard-coded threshold for now
    ]
    if len(low_coverage) > 0:
        logger.warning(
            f"LOW COVERAGE WARNING: {len(low_coverage)} regions have fewer "
            f"than 5 total samples across donors. Results for these regions "
            f"should be interpreted with extreme caution:\n"
            f"{low_coverage.index.tolist()}"
        )

    # --- Cache results ---
    ensure_dir(cache_dir)
    cache_dataframe(group_expr_filtered, cache_dir, cache_name)
    cache_dataframe(gene_info, cache_dir, f"{cache_name}_gene_info")
    cache_dataframe(coverage, cache_dir, f"{cache_name}_coverage")

    for i, d in enumerate(donor_dfs_filtered):
        cache_dataframe(d, cache_dir, f"{cache_name}_donor_{i}")

    params_used = {
        "parcellation": parcellation,
        "probe_selection": probe_selection,
        "normalization": normalization,
        "lr_mirror": lr_mirror,
        "return_donors": return_donors,
        "ds_threshold": ds_threshold,
        "n_regions": group_expr_filtered.shape[0],
        "n_genes_raw": n_genes,
        "n_genes_filtered": group_expr_filtered.shape[1],
        "n_donors": len(donor_dfs),
        "n_all_nan_regions": len(all_nan_regions),
    }
    cache_json(params_used, cache_dir, f"{cache_name}_meta")

    # Log the run
    from vulnmap.utils import get_project_root
    log_run(
        "expression",
        params_used,
        get_project_root() / "data" / "processed",
        extra={"output_shapes": str(group_expr_filtered.shape)},
    )

    return {
        "expression": group_expr_filtered,
        "donor_expressions": donor_dfs_filtered if return_donors else None,
        "coverage": coverage,
        "gene_info": gene_info,
        "parcellation_info": parc_info,
        "params_used": params_used,
    }


# =============================================================================
# Internal helpers
# =============================================================================

def _fetch_atlas(
    parcellation: str,
) -> tuple:
    """
    Fetch the atlas image and info for abagen.

    Uses abagen's built-in atlas fetchers or nilearn's dataset fetchers
    depending on the parcellation.

    Parameters
    ----------
    parcellation : str
        'desikan_killiany' or 'schaefer_200'.

    Returns
    -------
    tuple
        (atlas_img, atlas_info) suitable for abagen.get_expression_data().
        atlas_img: Nifti path or nibabel image.
        atlas_info: DataFrame with columns 'id', 'label', 'hemisphere', etc.
    """
    import abagen

    if parcellation == "desikan_killiany":
        # abagen provides a built-in Desikan-Killiany atlas via
        # abagen.fetch_desikan_killiany()
        # This returns the aparc+aseg atlas with both cortical and subcortical
        # labels, which is exactly what we need.
        logger.info("Fetching Desikan-Killiany atlas via abagen...")
        atlas = abagen.fetch_desikan_killiany()
        # atlas is a dict-like with 'image' and 'info' keys
        return atlas["image"], atlas["info"]

    elif parcellation == "schaefer_200":
        # Use nilearn to fetch Schaefer-200 and build atlas_info
        # abagen can accept any volumetric parcellation
        logger.info("Fetching Schaefer-200 atlas via nilearn...")
        try:
            from nilearn import datasets
            schaefer = datasets.fetch_atlas_schaefer_2018(
                n_rois=200, yeo_networks=7, resolution_mm=1
            )
            atlas_img = schaefer["maps"]
            # Build atlas_info DataFrame from labels
            labels = schaefer["labels"]
            # labels are bytes in some nilearn versions
            if isinstance(labels[0], bytes):
                labels = [l.decode("utf-8") for l in labels]
            atlas_info = pd.DataFrame({
                "id": range(1, len(labels) + 1),
                "label": labels,
                "hemisphere": [
                    "L" if "LH" in l or "lh" in l else "R"
                    for l in labels
                ],
                "structure": "cortex",
            })
            return atlas_img, atlas_info
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch Schaefer-200 atlas: {e}. "
                "Ensure nilearn is installed: pip install nilearn"
            )
    else:
        raise ValueError(f"Unknown parcellation: {parcellation}")


def _compute_gene_info(
    group_expr: pd.DataFrame,
    donor_dfs: List[pd.DataFrame],
    ds_threshold: float,
) -> pd.DataFrame:
    """
    Compute per-gene info: mean expression, std, and differential stability.

    Differential stability (DS) measures how consistently a gene's spatial
    pattern of expression is reproduced across donors. It is defined as the
    mean pairwise Pearson correlation of a gene's expression profile across
    all donor pairs (Hawrylycz et al., 2015).

    Parameters
    ----------
    group_expr : pd.DataFrame
        Group-level expression (regions × genes).
    donor_dfs : list of pd.DataFrame
        Per-donor expression matrices.
    ds_threshold : float
        Minimum DS to retain a gene.

    Returns
    -------
    pd.DataFrame
        Index = gene names. Columns: 'mean_expr', 'std_expr', 'ds', 'kept'.
    """
    gene_info = pd.DataFrame(index=group_expr.columns)
    gene_info.index.name = "gene"
    gene_info["mean_expr"] = group_expr.mean(axis=0)
    gene_info["std_expr"] = group_expr.std(axis=0)

    # Compute DS if we have per-donor data
    if len(donor_dfs) >= 2:
        logger.info("Computing differential stability across donors...")
        ds_values = _differential_stability(donor_dfs)
        gene_info["ds"] = ds_values
    else:
        logger.warning(
            "Cannot compute differential stability: need >= 2 donor matrices. "
            "All genes retained with DS = NaN."
        )
        gene_info["ds"] = np.nan

    # Filter
    if gene_info["ds"].notna().any():
        gene_info["kept"] = gene_info["ds"] >= ds_threshold
    else:
        gene_info["kept"] = True  # Keep all if DS unavailable

    return gene_info


def _differential_stability(
    donor_dfs: List[pd.DataFrame],
) -> pd.Series:
    """
    Compute differential stability (DS) for each gene.

    DS = mean pairwise Pearson correlation of a gene's expression vector
    across all pairs of donors.

    Parameters
    ----------
    donor_dfs : list of pd.DataFrame
        Per-donor expression matrices (regions × genes).
        All must have the same columns (genes).

    Returns
    -------
    pd.Series
        DS value for each gene. Index = gene names.
    """
    from itertools import combinations

    genes = donor_dfs[0].columns
    n_donors = len(donor_dfs)
    n_pairs = n_donors * (n_donors - 1) // 2

    # Find common regions across all donors (some may have NaN)
    ds = pd.Series(np.nan, index=genes, dtype=np.float64)

    # Stack donor matrices and compute pairwise correlations per gene
    pair_corrs = np.zeros((len(genes), n_pairs))

    for pair_idx, (i, j) in enumerate(combinations(range(n_donors), 2)):
        d1 = donor_dfs[i]
        d2 = donor_dfs[j]
        # Align on common regions (non-NaN in both)
        common = d1.index.intersection(d2.index)
        if len(common) < 3:
            pair_corrs[:, pair_idx] = np.nan
            continue
        for g_idx, gene in enumerate(genes):
            v1 = d1.loc[common, gene].values.astype(np.float64)
            v2 = d2.loc[common, gene].values.astype(np.float64)
            mask = np.isfinite(v1) & np.isfinite(v2)
            if mask.sum() < 3:
                pair_corrs[g_idx, pair_idx] = np.nan
            else:
                pair_corrs[g_idx, pair_idx] = np.corrcoef(
                    v1[mask], v2[mask]
                )[0, 1]

    ds = pd.Series(
        np.nanmean(pair_corrs, axis=1),
        index=genes,
        name="ds",
    )
    return ds


def _build_coverage_table(
    group_expr: pd.DataFrame,
    donor_dfs: List[pd.DataFrame],
    parcellation: str,
) -> pd.DataFrame:
    """
    Build a coverage table: how many samples per region per donor.

    This is critical for interpreting results from sparse regions.

    Parameters
    ----------
    group_expr : pd.DataFrame
        Group-level expression (regions × genes). Used for region list.
    donor_dfs : list of pd.DataFrame
        Per-donor expression matrices.
    parcellation : str
        Parcellation name (for labelling).

    Returns
    -------
    pd.DataFrame
        Index = region names. Columns include:
        - 'n_donors_present': how many donors have data for this region
        - 'n_samples_total': total samples across donors (estimated from
          non-NaN gene counts, since abagen aggregates samples to regions)
        - 'pct_genes_available': fraction of genes with non-NaN values
        - 'low_coverage': bool flag
    """
    regions = group_expr.index
    coverage = pd.DataFrame(index=regions)
    coverage.index.name = "region"

    # Count donors with non-NaN data per region
    if donor_dfs:
        donor_present = np.zeros(len(regions))
        for d in donor_dfs:
            # A donor "has data" for a region if at least one gene is non-NaN
            common = regions.intersection(d.index)
            for r in common:
                if d.loc[r].notna().any():
                    donor_present[regions.get_loc(r)] += 1
        coverage["n_donors_present"] = donor_present.astype(int)
    else:
        coverage["n_donors_present"] = np.nan

    # Fraction of genes available per region
    coverage["pct_genes_available"] = (
        group_expr.notna().sum(axis=1) / group_expr.shape[1] * 100
    )

    # Estimate total samples (rough proxy: n_donors_present, since abagen
    # aggregates multiple samples per donor per region into one value)
    coverage["n_samples_total"] = coverage["n_donors_present"]

    # Low coverage flag
    coverage["low_coverage"] = coverage["n_donors_present"] < 3

    coverage["parcellation"] = parcellation

    return coverage


# =============================================================================
# Summary / QC report
# =============================================================================

def expression_qc_report(result: Dict[str, Any]) -> str:
    """
    Generate a text QC report from build_expression() output.

    Parameters
    ----------
    result : dict
        Output of build_expression().

    Returns
    -------
    str
        Multi-line QC report.
    """
    expr = result["expression"]
    gene_info = result["gene_info"]
    coverage = result["coverage"]
    params = result["params_used"]
    parc_info = result["parcellation_info"]

    lines = [
        "=" * 70,
        "VulnMap Phase 1: Expression QC Report",
        "=" * 70,
        "",
        f"Parcellation: {parc_info['name']}",
        f"  {parc_info['description'][:100]}...",
        "",
        f"Expression matrix shape: {expr.shape[0]} regions x {expr.shape[1]} genes",
        f"  Raw genes (before DS filter): {params.get('n_genes_raw', '?')}",
        f"  Filtered genes (DS > {params.get('ds_threshold', '?')}): {params.get('n_genes_filtered', '?')}",
        f"  All-NaN regions: {params.get('n_all_nan_regions', '?')}",
        "",
        "--- Gene Info ---",
        f"  Mean expression range: [{gene_info['mean_expr'].min():.3f}, {gene_info['mean_expr'].max():.3f}]",
        f"  DS range: [{gene_info['ds'].min():.3f}, {gene_info['ds'].max():.3f}]"
        if gene_info["ds"].notna().any() else "  DS: not computed (< 2 donors)",
        f"  Genes kept: {gene_info['kept'].sum()}",
        f"  Genes dropped: {(~gene_info['kept']).sum()}",
        "",
        "--- Coverage ---",
    ]

    if coverage is not None and "n_donors_present" in coverage.columns:
        lines.extend([
            f"  Regions with data from all 6 donors: "
            f"{(coverage['n_donors_present'] == 6).sum()}",
            f"  Regions with data from < 3 donors (LOW COVERAGE): "
            f"{coverage['low_coverage'].sum()}",
        ])
        if coverage["low_coverage"].any():
            low_regions = coverage.index[coverage["low_coverage"]].tolist()
            lines.append(f"  Low-coverage regions: {low_regions}")
    else:
        lines.append("  Coverage data not available.")

    lines.extend([
        "",
        "--- Donor Matrices ---",
        f"  Number of donor matrices: "
        f"{len(result.get('donor_expressions', []) or [])}",
        "",
        "--- Parameters Used ---",
    ])
    for k, v in params.items():
        lines.append(f"  {k}: {v}")

    lines.extend([
        "",
        "--- LIMITATIONS ---",
        "  1. AHBA has only 6 donors; generalization is limited.",
        "  2. Only 2 donors have right-hemisphere samples.",
        "  3. Subcortical and brainstem coverage is sparse.",
        "  4. This is healthy adult tissue, not disease-state expression.",
        "  5. Substantia nigra may not be a separate region in this parcellation.",
        "=" * 70,
    ])

    return "\n".join(lines)
