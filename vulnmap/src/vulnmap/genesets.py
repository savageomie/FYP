"""
VulnMap: Gene Set Retrieval & Harmonization
============================================

Phase 2 of the VulnMap pipeline.

This module:
1. Fetches disorder-associated genes from the GWAS Catalog REST API.
2. Harmonizes gene symbols against the AHBA expression gene list (Phase 1).
3. Optionally integrates MAGMA gene-level results.
4. Filters out underpowered disorders (< min_genes after harmonization).
5. Computes Jaccard overlap between all disorder gene sets.

Data sources & caveats
----------------------
- GWAS Catalog REST API: https://www.ebi.ac.uk/gwas/rest/api/
  Rate-limited; all responses cached to data/raw/gwas/.
- MAPPED_GENES uses nearest-gene heuristics (linear genomic distance).
  In non-coding risk loci, enhancer-promoter looping often connects
  risk SNPs to distant genes, not the nearest one.
- MAGMA gene-level analysis is more defensible but requires the user
  to run MAGMA externally with GWAS summary statistics.
"""

import json
import re
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from vulnmap.utils import (
    Timer,
    cache_dataframe,
    cache_json,
    ensure_dir,
    load_cached_dataframe,
    load_cached_json,
    load_disorders,
    load_params,
    logger,
)


# =============================================================================
# GWAS Catalog REST API
# =============================================================================

GWAS_API_BASE = "https://www.ebi.ac.uk/gwas/rest/api/v2"
# Rate limiting: GWAS Catalog recommends max 1-2 requests per second
_RATE_LIMIT_SECONDS = 0.4


def _gwas_api_request(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
    max_retries: int = 3,
) -> Optional[dict]:
    """
    Make a GET request to the GWAS Catalog REST API with caching and retry logic.

    Parameters
    ----------
    endpoint : str
        API endpoint path (e.g., '/associations').
    params : dict, optional
        Query parameters for the GET request.
    cache_dir : Path, optional
        Directory to cache JSON responses. If None, no caching.
    force_refresh : bool
        If True, ignore cache.
    max_retries : int
        Maximum retries for 429 / transient errors.

    Returns
    -------
    dict or None
        Parsed JSON response, or None if request failed.
    """
    import requests

    url = f"{GWAS_API_BASE}{endpoint}" if not endpoint.startswith("http") else endpoint

    # Check cache first
    if cache_dir is not None and not force_refresh:
        param_parts = [f"{k}_{v}" for k, v in sorted((params or {}).items())]
        cache_key = f"{endpoint}_{'_'.join(param_parts)}" if param_parts else endpoint
        safe_name = re.sub(r'[^\w\-.]', '_', cache_key.strip('/'))
        cache_file = Path(cache_dir) / f"{safe_name}.json"
        if cache_file.exists():
            logger.info(f"Using cached GWAS API response: {cache_file.name}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)

    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching from GWAS API: {url} (params: {params})")
            response = requests.get(
                url,
                params=params,
                headers={"Accept": "application/json"},
                timeout=60,
            )
            if response.status_code == 429:
                wait_time = (attempt + 1) * 2
                logger.warning(
                    f"GWAS API rate limit (429) hit. Waiting {wait_time}s "
                    f"before retry ({attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            # Cache the response
            if cache_dir is not None:
                ensure_dir(cache_dir)
                param_parts = [f"{k}_{v}" for k, v in sorted((params or {}).items())]
                cache_key = f"{endpoint}_{'_'.join(param_parts)}" if param_parts else endpoint
                safe_name = re.sub(r'[^\w\-.]', '_', cache_key.strip('/'))
                cache_file = Path(cache_dir) / f"{safe_name}.json"
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)

            time.sleep(_RATE_LIMIT_SECONDS)
            return data

        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"GWAS API request failed for {url}: {e}")
                return None
            time.sleep((attempt + 1) * 2)

    return None


def _fetch_associations_paginated(
    efo_id: str,
    gwas_trait: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> List[dict]:
    """
    Fetch all associations for a trait, handling pagination and caching.

    Supports GWAS Catalog v2 API using efo_trait filters and local cache.
    """
    cache_ident = gwas_trait or efo_id
    safe_cache_key = re.sub(r'[^\w\-.]', '_', cache_ident)

    # Check for a cached complete result
    if cache_dir is not None and not force_refresh:
        for cname in [f"{safe_cache_key}_all_associations.json", f"{efo_id}_all_associations.json"]:
            complete_cache = Path(cache_dir) / cname
            if complete_cache.exists():
                logger.info(f"Using cached complete associations for {cache_ident}")
                with open(complete_cache, 'r', encoding='utf-8') as f:
                    return json.load(f)

    all_associations = []
    page = 0
    page_size = 500  # Max batch size

    trait_query = gwas_trait or efo_id

    while True:
        data = _gwas_api_request(
            "/associations",
            params={"efo_trait": trait_query, "size": page_size, "page": page},
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )
        if data is None:
            # If trait query returned nothing and efo_id differs, try efo_id
            if page == 0 and gwas_trait and gwas_trait != efo_id:
                logger.info(f"Trying fallback query with EFO ID: {efo_id}")
                data = _gwas_api_request(
                    "/associations",
                    params={"efo_trait": efo_id, "size": page_size, "page": page},
                    cache_dir=cache_dir,
                    force_refresh=force_refresh,
                )
            if data is None:
                logger.warning(f"Failed to fetch page {page} for {trait_query}")
                break

        embedded = data.get("_embedded", {})
        associations = embedded.get("associations", [])
        if not associations:
            break

        all_associations.extend(associations)
        logger.info(
            f"  {trait_query}: fetched page {page}, "
            f"{len(associations)} associations (total: {len(all_associations)})"
        )

        page_info = data.get("page", {})
        total_pages = page_info.get("totalPages", 1)
        if page + 1 >= total_pages or page >= 10:  # Cap at top 5,000 associations per trait
            break
        page += 1
        time.sleep(_RATE_LIMIT_SECONDS)

    # Cache the complete result
    if cache_dir is not None and all_associations:
        ensure_dir(cache_dir)
        complete_cache = Path(cache_dir) / f"{safe_cache_key}_all_associations.json"
        with open(complete_cache, 'w', encoding='utf-8') as f:
            json.dump(all_associations, f, indent=2)

    return all_associations


# =============================================================================
# Gene extraction from associations
# =============================================================================

def extract_genes_from_associations(
    associations: List[dict],
    p_threshold: float = 5e-8,
) -> Dict[str, Any]:
    """
    Extract gene symbols from GWAS Catalog association records.

    Supports both API v2 and legacy/mock formats.
    """
    mapped_genes: Set[str] = set()
    reported_genes: Set[str] = set()
    snps: List[str] = []
    n_significant = 0

    for assoc in associations:
        # Extract p-value: support direct float p_value (v2) or mantissa/exponent (v1)
        p_val = assoc.get("p_value")
        if p_val is None:
            try:
                p_mantissa = float(assoc.get("pvalueMantissa", assoc.get("pvalue_mantissa", 1)))
                p_exponent = int(assoc.get("pvalueExponent", assoc.get("pvalue_exponent", 0)))
                p_val = p_mantissa * (10 ** p_exponent)
            except (TypeError, ValueError):
                continue

        try:
            p_value = float(p_val)
        except (TypeError, ValueError):
            continue

        if p_value > p_threshold:
            continue

        n_significant += 1

        # Extract SNP IDs
        for snp_entry in assoc.get("snps", []):
            rs_id = snp_entry.get("rsId", "")
            if rs_id:
                snps.append(rs_id)
        for snp_entry in assoc.get("snp_allele", []):
            rs_id = snp_entry.get("rs_id", "")
            if rs_id:
                snps.append(rs_id)

        # Extract reported genes
        for locus in assoc.get("loci", []):
            for author_gene in locus.get("authorReportedGenes", []):
                gene_name = author_gene.get("geneName", "").strip()
                if gene_name and gene_name != "NR" and gene_name != "intergenic":
                    for g in re.split(r'[\s,;-]+', gene_name):
                        g = g.strip()
                        if g and len(g) > 1 and not g.startswith("LOC"):
                            reported_genes.add(g.upper())

        # Extract mapped genes: v2 direct mapped_genes list of strings
        for g in assoc.get("mapped_genes", []):
            if isinstance(g, str):
                g = g.strip()
                if g and not g.startswith("LOC"):
                    mapped_genes.add(g.upper())

        # Extract mapped genes: v1 nested genomic context
        for snp_entry in assoc.get("snps", []):
            for gene_entry in snp_entry.get("genes", []):
                gene_name = gene_entry.get("geneName", "").strip()
                if gene_name and not gene_name.startswith("LOC"):
                    mapped_genes.add(gene_name.upper())

    all_genes = mapped_genes | reported_genes

    return {
        "mapped_genes": mapped_genes,
        "reported_genes": reported_genes,
        "all_genes": all_genes,
        "n_associations": len(associations),
        "n_significant": n_significant,
        "snps": list(set(snps)),
    }



# =============================================================================
# Gene symbol harmonization
# =============================================================================

def harmonize_genes(
    query_genes: Set[str],
    reference_genes: List[str],
) -> Dict[str, Any]:
    """
    Harmonize query gene symbols against the AHBA reference gene list.

    Matching is case-insensitive. Reports matched, unmatched, and
    potential alias issues.

    Parameters
    ----------
    query_genes : set of str
        Gene symbols from GWAS/MAGMA.
    reference_genes : list of str
        Gene symbols from the AHBA expression matrix (Phase 1).

    Returns
    -------
    dict
        Keys:
        - 'matched': list of gene symbols present in both sets
        - 'unmatched': list of query genes not found in reference
        - 'n_input': number of input query genes
        - 'n_matched': number of matched genes
        - 'match_rate': fraction matched
    """
    # Build case-insensitive lookup
    ref_upper = {g.upper(): g for g in reference_genes}
    query_upper = {g.upper() for g in query_genes}

    matched = []
    unmatched = []

    for q in sorted(query_upper):
        if q in ref_upper:
            matched.append(ref_upper[q])
        else:
            unmatched.append(q)

    n_input = len(query_upper)
    n_matched = len(matched)

    if n_input > 0:
        match_rate = n_matched / n_input
    else:
        match_rate = 0.0

    if match_rate < 0.5:
        logger.warning(
            f"Low harmonization rate: {match_rate:.1%} ({n_matched}/{n_input}). "
            f"This may indicate gene symbol version mismatch or non-standard "
            f"gene naming in the GWAS source."
        )

    return {
        "matched": matched,
        "unmatched": unmatched,
        "n_input": n_input,
        "n_matched": n_matched,
        "match_rate": match_rate,
    }


# =============================================================================
# MAGMA integration
# =============================================================================

def load_magma_genes(
    magma_file: Union[str, Path],
    fdr_threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Load MAGMA gene-level results and extract significant genes.

    Expected file format: MAGMA .genes.out file with columns including
    GENE, ZSTAT, P. We apply BH-FDR correction.

    Parameters
    ----------
    magma_file : str or Path
        Path to MAGMA .genes.out file.
    fdr_threshold : float
        FDR threshold for significance. Default: 0.05.

    Returns
    -------
    dict
        Keys:
        - 'genes': set of significant gene symbols
        - 'n_total': total genes tested
        - 'n_significant': genes passing FDR
        - 'full_results': DataFrame with all MAGMA results
    """
    from scipy import stats as scipy_stats

    path = Path(magma_file)
    if not path.exists():
        logger.warning(f"MAGMA file not found: {path}")
        return {
            "genes": set(),
            "n_total": 0,
            "n_significant": 0,
            "full_results": None,
        }

    logger.info(f"Loading MAGMA results from {path}")

    # MAGMA .genes.out is whitespace-delimited
    try:
        df = pd.read_csv(path, sep=r'\s+', comment='#')
    except Exception as e:
        logger.error(f"Failed to parse MAGMA file: {e}")
        return {
            "genes": set(),
            "n_total": 0,
            "n_significant": 0,
            "full_results": None,
        }

    # Find the p-value column (MAGMA uses 'P')
    p_col = None
    for candidate in ['P', 'p', 'PVAL', 'pvalue', 'P_VALUE']:
        if candidate in df.columns:
            p_col = candidate
            break

    if p_col is None:
        logger.error(f"No p-value column found in MAGMA file. Columns: {df.columns.tolist()}")
        return {"genes": set(), "n_total": 0, "n_significant": 0, "full_results": df}

    # Find the gene column
    gene_col = None
    for candidate in ['GENE', 'gene', 'Gene', 'SYMBOL', 'gene_symbol']:
        if candidate in df.columns:
            gene_col = candidate
            break

    if gene_col is None:
        logger.error(f"No gene column found in MAGMA file. Columns: {df.columns.tolist()}")
        return {"genes": set(), "n_total": 0, "n_significant": 0, "full_results": df}

    # BH-FDR correction
    from vulnmap.utils import logger as _logger  # already imported
    p_values = df[p_col].values.astype(float)
    n = len(p_values)

    # Benjamini-Hochberg
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    bh_q = sorted_p * n / (np.arange(1, n + 1))
    bh_q = np.minimum.accumulate(bh_q[::-1])[::-1]
    bh_q = np.clip(bh_q, 0.0, 1.0)
    fdr_values = np.empty_like(bh_q)
    fdr_values[sorted_idx] = bh_q

    df['fdr'] = fdr_values
    significant = df[df['fdr'] < fdr_threshold]
    sig_genes = set(significant[gene_col].astype(str).str.upper().tolist())

    logger.info(
        f"MAGMA: {len(sig_genes)} genes significant at FDR < {fdr_threshold} "
        f"(out of {n} tested)"
    )

    return {
        "genes": sig_genes,
        "n_total": n,
        "n_significant": len(sig_genes),
        "full_results": df,
    }


# =============================================================================
# Jaccard overlap
# =============================================================================

def compute_jaccard_matrix(
    gene_sets: Dict[str, Set[str]],
) -> pd.DataFrame:
    """
    Compute pairwise Jaccard similarity between all gene sets.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|.

    Parameters
    ----------
    gene_sets : dict
        Mapping from disorder name to set of gene symbols.

    Returns
    -------
    pd.DataFrame
        Symmetric matrix of Jaccard similarities.
    """
    names = sorted(gene_sets.keys())
    n = len(names)
    matrix = np.zeros((n, n))

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i == j:
                matrix[i, j] = 1.0
            elif i < j:
                set_i = gene_sets[name_i]
                set_j = gene_sets[name_j]
                union = len(set_i | set_j)
                if union > 0:
                    jaccard = len(set_i & set_j) / union
                else:
                    jaccard = 0.0
                matrix[i, j] = jaccard
                matrix[j, i] = jaccard

    return pd.DataFrame(matrix, index=names, columns=names)


# =============================================================================
# Main pipeline entry point
# =============================================================================

def fetch_disorder_genes(
    disorder_name: str,
    efo_id: str,
    gwas_trait: Optional[str] = None,
    p_genome_wide: float = 5e-8,
    p_suggestive: float = 1e-5,
    reference_genes: Optional[List[str]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    magma_dir: Optional[Union[str, Path]] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Fetch and process gene sets for a single disorder.

    Parameters
    ----------
    disorder_name : str
        Human-readable disorder name (e.g., 'schizophrenia').
    efo_id : str
        EFO identifier for GWAS Catalog query.
    gwas_trait : str, optional
        Trait name for GWAS Catalog v2 query.
    p_genome_wide : float
        Genome-wide significance threshold. Default: 5e-8.
    p_suggestive : float
        Suggestive significance threshold. Default: 1e-5.
    reference_genes : list of str, optional
        AHBA gene list for harmonization. If None, no harmonization.
    cache_dir : str or Path, optional
        Cache directory for GWAS API responses.
    magma_dir : str or Path, optional
        Directory containing MAGMA .genes.out files.
    force_refresh : bool
        If True, re-fetch from API.

    Returns
    -------
    dict
        Keys:
        - 'disorder': disorder name
        - 'efo_id': EFO identifier
        - 'catalog_5e8': dict with gene set at genome-wide threshold
        - 'catalog_1e5': dict with gene set at suggestive threshold
        - 'magma_fdr': dict with MAGMA gene set (if available)
        - 'harmonization': dict with harmonization results per definition
        - 'warnings': list of warning strings
    """
    gwas_cache = Path(cache_dir) if cache_dir else None
    result_warnings = []

    # Fetch associations from GWAS Catalog
    logger.info(f"Fetching GWAS associations for {disorder_name} ({gwas_trait or efo_id})")
    associations = _fetch_associations_paginated(
        efo_id=efo_id,
        gwas_trait=gwas_trait,
        cache_dir=gwas_cache,
        force_refresh=force_refresh,
    )

    if not associations:
        result_warnings.append(
            f"No associations found for {efo_id}. "
            f"Verify the EFO ID at: https://www.ebi.ac.uk/gwas/efotraits/{efo_id}"
        )
        return {
            "disorder": disorder_name,
            "efo_id": efo_id,
            "catalog_5e8": {"genes": set(), "n_significant": 0},
            "catalog_1e5": {"genes": set(), "n_significant": 0},
            "magma_fdr": None,
            "harmonization": {},
            "warnings": result_warnings,
        }

    # Extract genes at both thresholds
    genes_5e8 = extract_genes_from_associations(associations, p_threshold=p_genome_wide)
    genes_1e5 = extract_genes_from_associations(associations, p_threshold=p_suggestive)

    logger.info(
        f"  {disorder_name}: "
        f"{len(genes_5e8['all_genes'])} genes at p<{p_genome_wide}, "
        f"{len(genes_1e5['all_genes'])} genes at p<{p_suggestive}"
    )

    # Harmonize with AHBA gene list if provided
    harmonization = {}
    if reference_genes is not None:
        for label, gene_result in [
            ("catalog_5e8", genes_5e8),
            ("catalog_1e5", genes_1e5),
        ]:
            harm = harmonize_genes(gene_result["all_genes"], reference_genes)
            harmonization[label] = harm
            logger.info(
                f"  {disorder_name} [{label}]: "
                f"{harm['n_matched']}/{harm['n_input']} genes matched to AHBA "
                f"({harm['match_rate']:.1%})"
            )

    # MAGMA integration (optional)
    magma_result = None
    if magma_dir is not None:
        magma_path = Path(magma_dir) / f"{disorder_name}.genes.out"
        if magma_path.exists():
            magma_result = load_magma_genes(magma_path)
            if reference_genes is not None and magma_result["genes"]:
                harm = harmonize_genes(magma_result["genes"], reference_genes)
                harmonization["magma_fdr"] = harm
                logger.info(
                    f"  {disorder_name} [magma_fdr]: "
                    f"{harm['n_matched']}/{harm['n_input']} MAGMA genes matched"
                )
        else:
            logger.info(f"  No MAGMA file found at {magma_path}")

    return {
        "disorder": disorder_name,
        "efo_id": efo_id,
        "catalog_5e8": genes_5e8,
        "catalog_1e5": genes_1e5,
        "magma_fdr": magma_result,
        "harmonization": harmonization,
        "warnings": result_warnings,
    }


def fetch_all_genesets(
    params: Optional[Dict[str, Any]] = None,
    disorders: Optional[Dict[str, Any]] = None,
    reference_genes: Optional[List[str]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Fetch gene sets for all disorders defined in disorders.yaml.

    This is the main entry point for Phase 2.

    Parameters
    ----------
    params : dict, optional
        Pipeline parameters. If None, loads from params.yaml.
    disorders : dict, optional
        Disorder definitions. If None, loads from disorders.yaml.
    reference_genes : list of str, optional
        AHBA gene list. If None, attempts to load from Phase 1 cache.
    cache_dir : str or Path, optional
        Cache directory. If None, uses data/raw/gwas/.
    force_refresh : bool
        If True, re-fetch all data from API.

    Returns
    -------
    dict
        Keys:
        - 'disorder_results': dict mapping disorder name to fetch results
        - 'summary_table': pd.DataFrame with gene count summary
        - 'jaccard_5e8': pd.DataFrame Jaccard matrix at genome-wide threshold
        - 'jaccard_1e5': pd.DataFrame Jaccard matrix at suggestive threshold
        - 'underpowered': list of disorder names with too few genes
        - 'harmonized_genesets': dict mapping (disorder, definition) to gene list
    """
    # Load configs
    if params is None:
        params = load_params()
    if disorders is None:
        disorders = load_disorders()

    gs_params = params.get("genesets", {})
    p_genome_wide = gs_params.get("gwas_p_genome_wide", 5e-8)
    p_suggestive = gs_params.get("gwas_p_suggestive", 1e-5)
    min_genes = gs_params.get("min_genes", 10)
    magma_dir = gs_params.get("magma_dir")

    if cache_dir is None:
        from vulnmap.utils import get_project_root
        cache_dir = get_project_root() / gs_params.get("gwas_cache_dir", "data/raw/gwas")

    if magma_dir is not None:
        from vulnmap.utils import get_project_root
        magma_dir = get_project_root() / magma_dir

    # Load reference genes from Phase 1 if not provided
    if reference_genes is None:
        try:
            from vulnmap.utils import get_project_root, load_cached_dataframe
            expr_cache = get_project_root() / "data" / "interim"
            expr_df = load_cached_dataframe(expr_cache, "expression_desikan_killiany")
            if expr_df is not None:
                reference_genes = expr_df.columns.tolist()
                logger.info(f"Loaded {len(reference_genes)} reference genes from Phase 1 cache")
            else:
                logger.warning(
                    "Phase 1 expression matrix not found in cache. "
                    "Gene harmonization will be skipped."
                )
        except Exception as e:
            logger.warning(f"Could not load Phase 1 data: {e}")

    # Fetch all disorders
    disorder_defs = disorders.get("disorders", {})
    disorder_results = {}
    all_warnings = []

    for disorder_name, dinfo in disorder_defs.items():
        efo_id = dinfo.get("efo_id", "")
        if not efo_id:
            logger.warning(f"Skipping {disorder_name}: no EFO ID defined")
            continue

        result = fetch_disorder_genes(
            disorder_name=disorder_name,
            efo_id=efo_id,
            gwas_trait=dinfo.get("gwas_trait"),
            p_genome_wide=p_genome_wide,
            p_suggestive=p_suggestive,
            reference_genes=reference_genes,
            cache_dir=cache_dir,
            magma_dir=magma_dir,
            force_refresh=force_refresh,
        )
        disorder_results[disorder_name] = result
        all_warnings.extend(result.get("warnings", []))

    # Build summary table
    summary_rows = []
    for name, res in disorder_results.items():
        dinfo = disorder_defs.get(name, {})
        row = {
            "disorder": name,
            "category": dinfo.get("category", "unknown"),
            "efo_id": res["efo_id"],
            "n_genes_5e8": len(res["catalog_5e8"].get("all_genes", set())),
            "n_genes_1e5": len(res["catalog_1e5"].get("all_genes", set())),
            "n_assoc_significant_5e8": res["catalog_5e8"].get("n_significant", 0),
        }
        # Add harmonized counts if available
        harm = res.get("harmonization", {})
        if "catalog_5e8" in harm:
            row["n_harmonized_5e8"] = harm["catalog_5e8"]["n_matched"]
            row["match_rate_5e8"] = harm["catalog_5e8"]["match_rate"]
        if "catalog_1e5" in harm:
            row["n_harmonized_1e5"] = harm["catalog_1e5"]["n_matched"]

        # MAGMA
        if res.get("magma_fdr") and res["magma_fdr"].get("genes"):
            row["n_magma_fdr"] = res["magma_fdr"]["n_significant"]
            if "magma_fdr" in harm:
                row["n_magma_harmonized"] = harm["magma_fdr"]["n_matched"]
        else:
            row["n_magma_fdr"] = 0

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("n_genes_5e8", ascending=False)

    # Identify underpowered disorders
    underpowered = []
    harmonized_col = "n_harmonized_5e8" if "n_harmonized_5e8" in summary_df.columns else "n_genes_5e8"
    for _, row in summary_df.iterrows():
        gene_count = row.get(harmonized_col, 0)
        if gene_count < min_genes:
            underpowered.append(row["disorder"])
            logger.warning(
                f"UNDERPOWERED: {row['disorder']} has only "
                f"{gene_count} genes (threshold: {min_genes}). "
                f"Excluded from headline results."
            )

    # Build harmonized gene set dictionary
    harmonized_genesets = {}
    for name, res in disorder_results.items():
        harm = res.get("harmonization", {})
        for defn in ["catalog_5e8", "catalog_1e5", "magma_fdr"]:
            if defn in harm and harm[defn]["matched"]:
                harmonized_genesets[(name, defn)] = harm[defn]["matched"]

    # Compute Jaccard matrices
    def _build_jaccard(definition: str) -> Optional[pd.DataFrame]:
        sets = {}
        for name, res in disorder_results.items():
            harm = res.get("harmonization", {})
            if definition in harm:
                sets[name] = set(harm[definition]["matched"])
            else:
                raw_genes = res.get(definition, {}).get("all_genes", set())
                if raw_genes:
                    sets[name] = raw_genes
        if len(sets) < 2:
            return None
        return compute_jaccard_matrix(sets)

    jaccard_5e8 = _build_jaccard("catalog_5e8")
    jaccard_1e5 = _build_jaccard("catalog_1e5")

    # Cache results
    from vulnmap.utils import get_project_root
    output_dir = get_project_root() / "data" / "interim"
    ensure_dir(output_dir)

    if not summary_df.empty:
        cache_dataframe(summary_df, output_dir, "geneset_summary")
    if jaccard_5e8 is not None:
        cache_dataframe(jaccard_5e8, output_dir, "jaccard_5e8")
    if jaccard_1e5 is not None:
        cache_dataframe(jaccard_1e5, output_dir, "jaccard_1e5")

    # Save underpowered list
    if underpowered:
        und_df = pd.DataFrame({"disorder": underpowered})
        und_path = output_dir / "underpowered_disorders.csv"
        und_df.to_csv(und_path, index=False)
        logger.info(f"Underpowered disorders saved to {und_path}")

    # Save harmonized gene sets
    if harmonized_genesets:
        gs_data = {
            f"{name}__{defn}": genes
            for (name, defn), genes in harmonized_genesets.items()
        }
        cache_json(gs_data, output_dir, "harmonized_genesets")

    return {
        "disorder_results": disorder_results,
        "summary_table": summary_df,
        "jaccard_5e8": jaccard_5e8,
        "jaccard_1e5": jaccard_1e5,
        "underpowered": underpowered,
        "harmonized_genesets": harmonized_genesets,
        "all_warnings": all_warnings,
    }


# =============================================================================
# QC report
# =============================================================================

def genesets_qc_report(result: Dict[str, Any]) -> str:
    """
    Generate a text QC report from fetch_all_genesets() output.

    Parameters
    ----------
    result : dict
        Output of fetch_all_genesets().

    Returns
    -------
    str
        Multi-line QC report.
    """
    summary = result["summary_table"]
    underpowered = result.get("underpowered", [])

    lines = [
        "=" * 70,
        "VulnMap Phase 2: Gene Sets QC Report",
        "=" * 70,
        "",
        f"Total disorders queried: {len(result['disorder_results'])}",
        f"Underpowered (< min_genes): {len(underpowered)}",
        "",
    ]

    if underpowered:
        lines.append("Underpowered disorders (excluded from headline results):")
        for d in underpowered:
            lines.append(f"  - {d}")
        lines.append("")

    if not summary.empty:
        lines.append("--- Gene Set Summary ---")
        lines.append(summary.to_string(index=False))
        lines.append("")

    # Jaccard summary
    j5 = result.get("jaccard_5e8")
    if j5 is not None:
        # Find highest off-diagonal Jaccard
        mask = np.ones(j5.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        max_j = j5.values[mask].max()
        max_idx = np.unravel_index(
            (j5.values * mask.astype(float)).argmax(), j5.shape
        )
        lines.append(
            f"--- Jaccard Overlap (genome-wide 5e-8) ---\n"
            f"  Max pairwise Jaccard: {max_j:.3f} "
            f"({j5.index[max_idx[0]]} vs {j5.columns[max_idx[1]]})\n"
            f"  Mean Jaccard: {j5.values[mask].mean():.3f}"
        )

    # Warnings
    warnings_list = result.get("all_warnings", [])
    if warnings_list:
        lines.append("\n--- WARNINGS ---")
        for w in warnings_list:
            lines.append(f"  [WARN] {w}")

    lines.extend([
        "",
        "--- LIMITATIONS ---",
        "  1. GWAS Catalog MAPPED_GENES uses nearest-gene heuristics.",
        "  2. Non-coding risk loci may map to wrong genes.",
        "  3. MAGMA gene-level analysis (when available) is more defensible.",
        "  4. Gene symbol harmonization may miss renamed/aliased genes.",
        "=" * 70,
    ])

    return "\n".join(lines)
