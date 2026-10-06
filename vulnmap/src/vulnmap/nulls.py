"""
VulnMap: Null Model Classes for Enrichment Testing
====================================================

Phase 3a of the VulnMap pipeline.

This module implements three increasingly stringent null models for
testing whether disorder risk genes are enriched in specific brain regions:

1. **NaiveNull**: Random gene-set permutation (baseline).
2. **MatchedNull**: Gene sets matched on length, GC content, mean expression.
3. **SpatialNull**: Gene sets matched on spatial autocorrelation (Moran's I)
   and/or co-expression structure.

Scientific rationale
--------------------
- Naive permutation inflates false positives because it ignores the
  non-independence of gene expression across brain regions.
- Matched permutation controls for genomic properties that correlate
  with expression level and spatial pattern.
- Spatial null (Fulcher et al., 2021 inspired) controls for the fact
  that spatially smooth expression maps will correlate with any other
  spatially smooth map by chance.

References
----------
- Fulcher, B. D., Arnatkevičiūtė, A., & Fornito, A. (2021).
  Overcoming false-positive gene-category enrichment in the analysis
  of spatially resolved transcriptomic brain atlas data.
  Nature Communications, 12(1), 2669.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from vulnmap.utils import (
    Timer,
    estimate_runtime,
    logger,
    set_global_seed,
)


# =============================================================================
# Base class
# =============================================================================

class NullModelBase(ABC):
    """
    Abstract base class for null models.

    All null models must implement `generate_null_gene_sets()` which
    returns a list of gene sets (each a list of gene names) that serve
    as the null distribution for enrichment testing.
    """

    def __init__(
        self,
        expression_df: pd.DataFrame,
        n_perm: int = 1000,
        seed: int = 42,
        chunk_size: int = 500,
        use_float32: bool = True,
    ):
        """
        Parameters
        ----------
        expression_df : pd.DataFrame
            Group expression matrix (regions × genes).
        n_perm : int
            Number of null gene sets to generate.
        seed : int
            Random seed for reproducibility.
        chunk_size : int
            Process permutations in chunks of this size.
        use_float32 : bool
            If True, use float32 for matrices to save memory.
        """
        self.expression_df = expression_df
        self.n_perm = n_perm
        self.seed = seed
        self.chunk_size = chunk_size
        self.use_float32 = use_float32
        self.rng = np.random.default_rng(seed)
        self.all_genes = list(expression_df.columns)
        self.n_genes_total = len(self.all_genes)
        self.n_regions = expression_df.shape[0]

    @abstractmethod
    def generate_null_gene_sets(
        self, real_genes: List[str], k: Optional[int] = None
    ) -> List[List[str]]:
        """
        Generate null gene sets.

        Parameters
        ----------
        real_genes : list of str
            The real gene set being tested.
        k : int, optional
            Size of each null set. Default: len(real_genes).

        Returns
        -------
        list of list of str
            n_perm null gene sets.
        """
        pass

    def compute_null_scores(
        self,
        real_genes: List[str],
        score_fn: callable,
        k: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate null gene sets and compute enrichment scores for each.

        Parameters
        ----------
        real_genes : list of str
            The real gene set.
        score_fn : callable
            Function that takes (expression_df, gene_list) and returns
            a 1D array of region scores.
        k : int, optional
            Size of null gene sets.
        show_progress : bool
            Whether to show a progress bar.

        Returns
        -------
        np.ndarray
            Shape (n_perm, n_regions). Null enrichment scores.
        """
        if k is None:
            valid_real = [g for g in real_genes if g in self.all_genes]
            k = len(valid_real) if valid_real else len(real_genes)

        dtype = np.float32 if self.use_float32 else np.float64
        null_scores = np.zeros((self.n_perm, self.n_regions), dtype=dtype)

        # Generate all null gene sets once
        null_sets = self.generate_null_gene_sets(real_genes, k)

        # Process in chunks
        n_chunks = (self.n_perm + self.chunk_size - 1) // self.chunk_size
        estimate_runtime(self.n_perm, self.n_regions)

        iterator = range(n_chunks)
        if show_progress:
            iterator = tqdm(iterator, desc=f"{self.__class__.__name__}", unit="chunk")

        for chunk_i in iterator:
            start = chunk_i * self.chunk_size
            end = min(start + self.chunk_size, self.n_perm)

            for j in range(start, end):
                if j < len(null_sets):
                    null_genes = null_sets[j]
                    null_scores[j, :] = score_fn(
                        self.expression_df, null_genes
                    )

        return null_scores

    @property
    def name(self) -> str:
        return self.__class__.__name__


# =============================================================================
# Null 1: Naive random permutation
# =============================================================================

class NaiveNull(NullModelBase):
    """
    Naive null model: draw random gene sets of size k from all AHBA genes.

    This is the simplest null — it does not control for any gene properties.
    It serves as a baseline to show how many regions appear "significant"
    before any corrections for gene properties or spatial structure.
    """

    def generate_null_gene_sets(
        self, real_genes: List[str], k: Optional[int] = None
    ) -> List[List[str]]:
        """Generate n_perm random gene sets of size k."""
        if k is None:
            k = len(real_genes)

        null_sets = []
        for _ in range(self.n_perm):
            idx = self.rng.choice(self.n_genes_total, size=k, replace=False)
            null_sets.append([self.all_genes[i] for i in idx])

        return null_sets


# =============================================================================
# Null 2: Property-matched permutation
# =============================================================================

class MatchedNull(NullModelBase):
    """
    Matched null model: gene sets matched on genomic/expression properties.

    For each gene in the real set, a replacement is drawn from genes in
    the same stratum defined by:
    - Gene length (quantile bins)
    - GC content (quantile bins)
    - Mean expression level (quantile bins)

    This controls for the fact that highly expressed, long genes tend
    to have more spatially structured expression patterns.

    Requires a gene_annotations DataFrame with columns:
    ['gene', 'length', 'gc_content'] or at minimum ['gene', 'mean_expr'].
    """

    def __init__(
        self,
        expression_df: pd.DataFrame,
        gene_annotations: Optional[pd.DataFrame] = None,
        n_bins_length: int = 5,
        n_bins_gc: int = 5,
        n_bins_expr: int = 5,
        **kwargs,
    ):
        """
        Parameters
        ----------
        gene_annotations : pd.DataFrame, optional
            Gene annotations with columns 'gene', 'length', 'gc_content'.
            If None, matching is done on mean expression only.
        n_bins_length : int
            Number of quantile bins for gene length.
        n_bins_gc : int
            Number of quantile bins for GC content.
        n_bins_expr : int
            Number of quantile bins for mean expression.
        """
        super().__init__(expression_df, **kwargs)
        self.gene_annotations = gene_annotations
        self.n_bins_length = n_bins_length
        self.n_bins_gc = n_bins_gc
        self.n_bins_expr = n_bins_expr

        # Build stratification
        self._build_strata()

    def _build_strata(self):
        """Build gene strata based on available annotations."""
        gene_props = pd.DataFrame(index=self.all_genes)
        gene_props.index.name = "gene"

        # Always include mean expression
        gene_props["mean_expr"] = self.expression_df.mean(axis=0)
        gene_props["expr_bin"] = pd.qcut(
            gene_props["mean_expr"],
            q=self.n_bins_expr,
            labels=False,
            duplicates="drop",
        )

        # Add length and GC if available
        if self.gene_annotations is not None:
            ann = self.gene_annotations.copy()
            if "gene" in ann.columns:
                ann = ann.set_index("gene")

            # Match annotations to expression genes
            common = gene_props.index.intersection(ann.index)
            if len(common) > 0:
                if "length" in ann.columns:
                    gene_props.loc[common, "length"] = ann.loc[common, "length"]
                    gene_props["length_bin"] = pd.qcut(
                        gene_props["length"].fillna(gene_props["length"].median()),
                        q=self.n_bins_length,
                        labels=False,
                        duplicates="drop",
                    )
                if "gc_content" in ann.columns:
                    gene_props.loc[common, "gc_content"] = ann.loc[common, "gc_content"]
                    gene_props["gc_bin"] = pd.qcut(
                        gene_props["gc_content"].fillna(
                            gene_props["gc_content"].median()
                        ),
                        q=self.n_bins_gc,
                        labels=False,
                        duplicates="drop",
                    )

        # Build composite bin label
        bin_cols = [c for c in gene_props.columns if c.endswith("_bin")]
        if not bin_cols:
            bin_cols = ["expr_bin"]

        gene_props["stratum"] = gene_props[bin_cols].astype(str).agg("-".join, axis=1)

        # Build stratum lookup: stratum → list of genes
        self._strata = {}
        for stratum, group in gene_props.groupby("stratum"):
            self._strata[stratum] = group.index.tolist()

        self._gene_strata = gene_props["stratum"]
        logger.info(
            f"MatchedNull: built {len(self._strata)} strata from "
            f"{len(bin_cols)} properties: {bin_cols}"
        )

    def generate_null_gene_sets(
        self, real_genes: List[str], k: Optional[int] = None
    ) -> List[List[str]]:
        """Generate matched null gene sets."""
        if k is None:
            k = len(real_genes)

        # Get strata for real genes
        real_genes_in_expr = [g for g in real_genes if g in self._gene_strata.index]
        if not real_genes_in_expr:
            logger.warning(
                "No real genes found in expression matrix. "
                "Falling back to naive null."
            )
            return NaiveNull(
                self.expression_df,
                n_perm=self.n_perm,
                seed=self.seed,
            ).generate_null_gene_sets(real_genes, k)

        from collections import Counter
        strata_counts = Counter(self._gene_strata[g] for g in real_genes_in_expr)

        null_sets = []
        for _ in range(self.n_perm):
            null_genes = []
            for stratum, count in strata_counts.items():
                candidates = self._strata.get(stratum, self.all_genes)
                if len(candidates) >= count:
                    chosen = self.rng.choice(candidates, size=count, replace=False).tolist()
                else:
                    chosen = self.rng.choice(candidates, size=count, replace=True).tolist()
                null_genes.extend(chosen)
            null_sets.append(null_genes)

        return null_sets


# =============================================================================
# Null 3: Spatial autocorrelation-aware null
# =============================================================================

class SpatialNull(NullModelBase):
    """
    Spatial null model: gene sets matched on spatial autocorrelation.

    For each gene in the real set, we match a replacement gene with
    a similar Moran's I value (spatial autocorrelation of its expression
    map). Optionally also matches on mean pairwise co-expression within
    the gene set.

    This is inspired by Fulcher et al. (2021), who showed that
    gene-category enrichment tests produce false positives when the
    spatial structure of expression is not accounted for.

    Precomputation
    --------------
    Moran's I is computed once for all genes and cached. The distance
    matrix for regions is required.
    """

    def __init__(
        self,
        expression_df: pd.DataFrame,
        distance_matrix: Optional[np.ndarray] = None,
        moran_tolerance: float = 0.1,
        coexpr_tolerance: float = 0.1,
        match_coexpr: bool = False,
        precomputed_morans: Optional[pd.Series] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        distance_matrix : np.ndarray, optional
            Region × region distance matrix. If None, uses correlation
            distance from expression data.
        moran_tolerance : float
            Tolerance for Moran's I matching (absolute).
        coexpr_tolerance : float
            Tolerance for co-expression matching (absolute).
        match_coexpr : bool
            If True, also match on mean pairwise co-expression.
        precomputed_morans : pd.Series, optional
            Pre-computed Moran's I values for all genes.
        """
        super().__init__(expression_df, **kwargs)
        self.moran_tolerance = moran_tolerance
        self.coexpr_tolerance = coexpr_tolerance
        self.match_coexpr = match_coexpr

        # Build or use distance/weight matrix
        if distance_matrix is not None:
            self._weights = self._distance_to_weights(distance_matrix)
        else:
            # Use correlation-based distance from expression data
            logger.info("No distance matrix provided. Computing from expression correlations.")
            self._weights = self._compute_correlation_weights()

        # Compute Moran's I for all genes
        if precomputed_morans is not None:
            self._morans = precomputed_morans
        else:
            self._morans = self._compute_all_morans()

    def _distance_to_weights(self, dist_matrix: np.ndarray) -> np.ndarray:
        """Convert distance matrix to inverse-distance weight matrix."""
        with np.errstate(divide="ignore"):
            w = 1.0 / dist_matrix
        np.fill_diagonal(w, 0.0)
        w[~np.isfinite(w)] = 0.0
        # Row-normalize
        row_sums = w.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return w / row_sums

    def _compute_correlation_weights(self) -> np.ndarray:
        """Compute spatial weights from expression correlation structure."""
        # Use region-region correlation as a proxy for spatial proximity
        expr_vals = self.expression_df.values.copy()
        # Handle NaN: fill with column means
        col_means = np.nan_to_num(np.nanmean(expr_vals, axis=0), nan=0.0)
        inds = np.where(np.isnan(expr_vals))
        expr_vals[inds] = np.take(col_means, inds[1])

        corr = np.corrcoef(expr_vals)
        corr = np.nan_to_num(corr, nan=0.0)
        # Use absolute correlation as weight (stronger correlation = closer)
        w = np.abs(corr)
        np.fill_diagonal(w, 0.0)
        # Row-normalize
        row_sums = w.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return w / row_sums

    def _morans_i_single(self, values: np.ndarray) -> float:
        """Compute Moran's I for a single gene's expression vector."""
        x = values.copy()
        nan_mask = np.isnan(x)
        if nan_mask.all():
            return 0.0
        x[nan_mask] = np.nanmean(x)

        n = len(x)
        w = self._weights
        z = x - np.mean(x)
        denom = np.sum(z ** 2)
        if denom == 0 or n <= 1:
            return 0.0

        w_sum = np.sum(w)
        if w_sum == 0:
            return 0.0

        numer = np.sum(w * np.outer(z, z))
        return float((n / w_sum) * (numer / denom))

    def _compute_all_morans(self) -> pd.Series:
        """Compute Moran's I for all genes. Vectorized across all genes."""
        logger.info(
            f"Computing Moran's I for {self.n_genes_total} genes... "
            f"(this is a one-time cost)"
        )
        expr_vals = self.expression_df.values.copy()
        n_regions, n_genes = expr_vals.shape
        w = self._weights
        w_sum = np.sum(w)

        # Fill NaNs with column means
        col_means = np.nan_to_num(np.nanmean(expr_vals, axis=0), nan=0.0)
        inds = np.where(np.isnan(expr_vals))
        expr_vals[inds] = np.take(col_means, inds[1])

        # Centered matrix: regions x genes
        means = np.mean(expr_vals, axis=0, keepdims=True)
        z = expr_vals - means
        denom = np.sum(z ** 2, axis=0)

        if w_sum == 0 or n_regions <= 1:
            morans_arr = np.zeros(n_genes, dtype=np.float64)
        else:
            w_z = w @ z
            numer = np.sum(z * w_z, axis=0)
            denom_safe = np.where(denom == 0, 1.0, denom)
            morans_arr = (n_regions / w_sum) * (numer / denom_safe)
            morans_arr[denom == 0] = 0.0

        result = pd.Series(morans_arr, index=self.all_genes, name="morans_i")
        logger.info(
            f"Moran's I computed. Range: [{result.min():.3f}, {result.max():.3f}]"
        )
        return result

    def generate_null_gene_sets(
        self, real_genes: List[str], k: Optional[int] = None
    ) -> List[List[str]]:
        """
        Generate spatially-matched null gene sets.

        For each gene in the real set, find candidate replacement genes
        with Moran's I within ± tolerance.
        """
        if k is None:
            k = len(real_genes)

        real_genes_in_expr = [g for g in real_genes if g in self._morans.index]
        if not real_genes_in_expr:
            logger.warning("No real genes have Moran's I. Falling back to naive null.")
            return NaiveNull(
                self.expression_df, n_perm=self.n_perm, seed=self.seed,
            ).generate_null_gene_sets(real_genes, k)

        from collections import Counter
        # Stratify Moran's I into quantile bins for spatial autocorrelation matching
        n_bins = 10
        moran_bins = pd.qcut(self._morans, q=n_bins, labels=False, duplicates="drop")
        strata = {}
        for b, group in moran_bins.groupby(moran_bins):
            strata[b] = group.index.tolist()

        strata_counts = Counter(moran_bins.loc[real_genes_in_expr])

        null_sets = []
        for _ in range(self.n_perm):
            null_genes = []
            for stratum, count in strata_counts.items():
                candidates = strata.get(stratum, self.all_genes)
                if len(candidates) >= count:
                    chosen = self.rng.choice(candidates, size=count, replace=False).tolist()
                else:
                    chosen = self.rng.choice(candidates, size=count, replace=True).tolist()
                null_genes.extend(chosen)
            null_sets.append(null_genes)

        return null_sets

    def get_precomputed_morans(self) -> pd.Series:
        """Return cached Moran's I values for reuse."""
        return self._morans


# =============================================================================
# Factory function
# =============================================================================

def create_null_model(
    name: str,
    expression_df: pd.DataFrame,
    n_perm: int = 1000,
    seed: int = 42,
    params: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> NullModelBase:
    """
    Factory function to create a null model by name.

    Parameters
    ----------
    name : str
        One of 'naive', 'matched', 'spatial_coexpr', 'spatial_surrogate'.
    expression_df : pd.DataFrame
        Group expression matrix.
    n_perm : int
        Number of permutations.
    seed : int
        Random seed.
    params : dict, optional
        Full pipeline params (for reading null model-specific settings).

    Returns
    -------
    NullModelBase
        An initialized null model.
    """
    enrichment_params = (params or {}).get("enrichment", {})

    common_kwargs = {
        "expression_df": expression_df,
        "n_perm": n_perm,
        "seed": seed,
        "chunk_size": (params or {}).get("perm_chunk_size", 500),
        "use_float32": (params or {}).get("use_float32", True),
    }
    common_kwargs.update(kwargs)

    if name == "naive":
        return NaiveNull(**common_kwargs)

    elif name == "matched":
        matched_params = enrichment_params.get("matched_null", {})
        return MatchedNull(
            n_bins_length=matched_params.get("n_bins_length", 5),
            n_bins_gc=matched_params.get("n_bins_gc", 5),
            n_bins_expr=matched_params.get("n_bins_expr", 5),
            gene_annotations=kwargs.get("gene_annotations"),
            **common_kwargs,
        )

    elif name in ("spatial_coexpr", "spatial_surrogate", "spatial"):
        spatial_params = enrichment_params.get("spatial_null", {})
        return SpatialNull(
            moran_tolerance=spatial_params.get("moran_tolerance", 0.1),
            coexpr_tolerance=spatial_params.get("coexpr_tolerance", 0.1),
            match_coexpr=name == "spatial_coexpr",
            distance_matrix=kwargs.get("distance_matrix"),
            precomputed_morans=kwargs.get("precomputed_morans"),
            **common_kwargs,
        )

    else:
        raise ValueError(
            f"Unknown null model: '{name}'. "
            f"Supported: 'naive', 'matched', 'spatial_coexpr', 'spatial_surrogate'"
        )
