# Data Directory: raw/

This directory is **git-ignored**. It contains raw data files that are either
auto-downloaded by the pipeline or must be supplied by the user.

## Auto-Downloaded (no action needed)

| File/Dir | Source | Downloaded By |
|----------|--------|---------------|
| `abagen/` | Allen Human Brain Atlas | `abagen.get_expression_data()` on first run (~4 GB) |
| `gwas/` | GWAS Catalog REST API | `genesets.py` (cached JSON per disorder) |
| `gene_annotations.csv` | Ensembl BioMart | `nulls.py` (gene length, GC content) |

## User-Supplied (optional but recommended)

| File | Format | Purpose |
|------|--------|---------|
| `magma/{disorder}.genes.out` | MAGMA gene-level output | Phase 2: MAGMA gene sets |
| `celltype_markers.csv` | CSV: columns `gene`, `cell_type` | Phase 5: cell-type deconvolution |
| `atrophy/parkinsons.csv` | CSV: columns `region`, `effect_size` | Phase 4: PD damage map |
| `atrophy/alzheimers.csv` | CSV: columns `region`, `effect_size` | Phase 4: AD damage map |

## MAGMA Commands (for reference)

If you have GWAS summary statistics, run MAGMA as follows:

```bash
# Step 1: Annotate SNPs to genes (requires a gene location file)
magma --annotate \
  --snp-loc {snp_loc_file} \
  --gene-loc {gene_loc_file} \
  --out {disorder}_annot

# Step 2: Gene-level analysis
magma --bfile {reference_panel} \
  --gene-annot {disorder}_annot.genes.annot \
  --pval {summary_stats_file} ncol=N \
  --out data/raw/magma/{disorder}
```

The output file `{disorder}.genes.out` is what VulnMap reads.
See de Leeuw et al. 2015 (PLOS Comp Bio) for MAGMA documentation.
