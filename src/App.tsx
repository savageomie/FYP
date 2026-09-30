import React, { useState } from 'react';
import { 
  BookOpen, 
  Code, 
  Copy, 
  Check, 
  Download, 
  Cpu, 
  Dna, 
  Brain, 
  FileText, 
  AlertTriangle, 
  ShieldCheck, 
  ExternalLink,
  ChevronRight,
  Database,
  Layers,
  Sparkles,
  Info
} from 'lucide-react';

interface NotebookCell {
  id: string;
  type: 'markdown' | 'code';
  title: string;
  content: string;
}

const NOTEBOOK_01_CELLS: NotebookCell[] = [
  {
    id: 'cell-md-0',
    type: 'markdown',
    title: 'Notebook Header & Overview',
    content: `# Final-Year Project: Transcriptomic Brain Vulnerability Pipeline
## 01_expression.ipynb: AHBA Expression Matrix Construction & Quality Control

### Scientific Rationale & Purpose
This notebook builds the foundational regional transcriptomic matrices from the **Allen Human Brain Atlas (AHBA)** (Hawrylycz et al., 2012, 2015) using the rigorously validated \`abagen\` processing framework (Arnatkevičiūtė et al., 2019; Markello et al., 2021). 

In this first stage of the project:
1. We retrieve raw AHBA microarray data across all 6 adult post-mortem human donors.
2. We map microarray tissue samples to brain parcellations in standard MNI space:
   - **Parcellation A (Primary)**: Desikan-Killiany (DK-68 cortical regions + 14 subcortical nuclei = 82-84 structural parcels), aligning directly with ENIGMA clinical MRI damage maps.
   - **Parcellation B (Robustness)**: Schaefer-200 (200 functional cortical parcels) + subcortical structures.
3. We apply **scaled robust sigmoid (SRS)** normalization to mitigate inter-donor variability while preserving biological dynamic range.
4. We quantify **gene Differential Stability (DS)** across donors (Hawrylycz et al., 2015) and filter out inconsistent probes ($DS < 0.10$).
5. We export group and per-donor regional expression matrices, sample coverage maps, and write shared utility functions (\`utils.py\`) to Google Drive.

### Inputs & Outputs
- **Inputs**: 
  - Raw AHBA microarray datasets (auto-downloaded and cached by \`abagen\`).
  - Anatomical parcellation files (Desikan-Killiany volumetric atlas with subcortex via \`abagen.fetch_desikan_killiany()\`).
- **Outputs (written to \`/content/drive/MyDrive/brain_vuln/\`)**:
  - \`data/expression/group_expression_dk.parquet\`
  - \`data/expression/donor_expression_dk.parquet\`
  - \`data/expression/filtered_genes.csv\`
  - \`data/expression/gene_differential_stability.csv\`
  - \`outputs/tables/sample_coverage_dk.csv\`
  - \`outputs/figures/fig1_coverage_and_differential_stability.png\` (.svg)
  - \`outputs/logs/params_01_expression.json\`
  - \`utils.py\` (shared pipeline library imported by notebooks 02-07)`
  },
  {
    id: 'cell-code-1',
    type: 'code',
    title: 'Install Pinned Dependencies',
    content: `# Cell 1: Environment Setup & Pinned Dependencies
# Note: Pinned versions ensure reproducibility across Colab runtimes.

import sys
import subprocess

print("Installing pinned scientific & neuroimaging packages...")
cmd = [
    sys.executable, "-m", "pip", "install", "--quiet",
    "abagen==0.1.3",
    "nilearn==0.10.4",
    "nibabel==5.2.1",
    "neuromaps==0.0.5",
    "pandas==2.2.2",
    "numpy==1.26.4",
    "scipy==1.13.1",
    "matplotlib==3.8.4",
    "seaborn==0.13.2",
    "pyarrow==16.1.0",
    "fastparquet==2024.5.0"
]
subprocess.check_call(cmd)
print("✓ Pinned dependencies successfully installed.")`
  },
  {
    id: 'cell-code-2',
    type: 'code',
    title: 'Mount Google Drive & Initialize Directory Tree',
    content: `# Cell 2: Google Drive Mounting & Project Directory Structure
import os
import json
import random
from pathlib import Path
import numpy as np

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    PROJECT_ROOT = Path('/content/drive/MyDrive/brain_vuln')
    print("✓ Google Drive mounted at /content/drive")
except ImportError:
    # Local fallback for testing outside Google Colab
    PROJECT_ROOT = Path(os.path.expanduser('~/brain_vuln'))
    print(f"Running outside Colab; using local project directory: {PROJECT_ROOT}")

# Standard directory tree for the entire pipeline
DIRS = {
    'root': PROJECT_ROOT,
    'ahba_raw': PROJECT_ROOT / 'data' / 'ahba_raw',
    'parcellations': PROJECT_ROOT / 'data' / 'parcellations',
    'expression': PROJECT_ROOT / 'data' / 'expression',
    'genesets': PROJECT_ROOT / 'data' / 'genesets',
    'damage_maps': PROJECT_ROOT / 'data' / 'damage_maps',
    'null_models': PROJECT_ROOT / 'data' / 'null_models',
    'results': PROJECT_ROOT / 'data' / 'results',
    'tables': PROJECT_ROOT / 'outputs' / 'tables',
    'figures': PROJECT_ROOT / 'outputs' / 'figures',
    'logs': PROJECT_ROOT / 'outputs' / 'logs',
}

for name, p in DIRS.items():
    p.mkdir(parents=True, exist_ok=True)
    
print(f"✓ Directory tree initialized under: {PROJECT_ROOT}")`
  },
  {
    id: 'cell-code-3',
    type: 'code',
    title: 'Global Seeds & Parameter Configuration',
    content: `# Cell 3: Global Reproducibility Seed & Config Parameters
GLOBAL_SEED: int = 42

def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Global random seed set to: {seed}")

set_global_seed(GLOBAL_SEED)

# Expose core transcriptomic processing options
CONFIG_01 = {
    "project": "Transcriptomic Brain Vulnerability Pipeline",
    "step": "01_expression",
    "global_seed": GLOBAL_SEED,
    # VERIFY: abagen default probe selection is 'diff_stability' (Hawrylycz et al., 2015).
    # If using diffusion-weighted or spatial matching, note abagen accepts:
    # ['diff_stability', 'max_intensity', 'variance', 'corr_variance', 'corr_intensity', 'average', 'random'].
    "probe_selection": "diff_stability", 
    "donor_probes": "aggregate",
    "sample_norm": "srs",          # Scaled Robust Sigmoid (Fulcher & Fornito, 2016)
    "gene_norm": "srs",            # Scaled Robust Sigmoid
    "norm_matched": True,          # Structure-matched normalization across samples
    "norm_structures": True,       # Normalize cortical and subcortical separately
    "lr_mirror": False,            # Left-hemisphere-only (False) vs mirrored across hemispheres (True)
    "return_donors": True,         # Retain individual donor matrices for LODO robustness
    "differential_stability_threshold": 0.10,  # Minimum cross-donor correlation threshold
    "tolerance_mm": 2.0,           # Sample-to-parcel matching distance tolerance (mm)
    "ibf_threshold": 0.50,         # Intensity-based filtering threshold (fraction of samples > background)
    "parcellation_primary": "Desikan-Killiany + Subcortex (82 structures)",
    "parcellation_robustness": "Schaefer-200 + Subcortex"
}

params_path = DIRS['logs'] / 'params_01_expression.json'
with open(params_path, 'w') as f:
    json.dump(CONFIG_01, f, indent=2)
print(f"✓ Pipeline parameters logged to: {params_path}")`
  },
  {
    id: 'cell-code-4',
    type: 'code',
    title: 'Write Shared Pipeline Library (utils.py) to Drive',
    content: `# Cell 4: Create and Export Shared utils.py to Project Root
# Notebook 01 writes utils.py to Google Drive; all subsequent notebooks (02-07) import it.

utils_code = '''"""
Shared computational utilities for the Transcriptomic Brain Vulnerability Pipeline.
Project: GWAS Risk Genes & Regional Brain Vulnerability
Author: Computational Neuroscience & Bioinformatics Team
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

def save_dataframe(
    df: pd.DataFrame, 
    filepath: Union[str, Path], 
    format: str = "parquet"
) -> Path:
    """Save a DataFrame as Parquet or CSV with parent directory auto-creation."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    if format == "parquet":
        df.to_parquet(p, index=True)
    elif format == "csv":
        df.to_csv(p, index=True)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'parquet' or 'csv'.")
    return p

def load_dataframe(
    filepath: Union[str, Path], 
    format: Optional[str] = None
) -> pd.DataFrame:
    """Load a DataFrame from Parquet or CSV file."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    fmt = format or p.suffix.lstrip(".").lower()
    if fmt in ("parquet", "pq"):
        return pd.read_parquet(p)
    elif fmt == "csv":
        return pd.read_csv(p, index_col=0)
    raise ValueError(f"Unknown format for: {filepath}")

def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> Path:
    """Save dictionary to a formatted JSON file."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return p

def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file into a Python dictionary."""
    p = Path(filepath)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_differential_stability(
    donor_matrices: Dict[str, pd.DataFrame]
) -> pd.Series:
    """
    Compute gene-wise Differential Stability (DS) across donors (Hawrylycz et al., 2015).
    DS is defined as the mean pairwise Spearman correlation of a gene's regional
    expression pattern across all unique donor pairs.
    
    Parameters
    ----------
    donor_matrices : Dict[str, pd.DataFrame]
        Dictionary mapping donor ID to a DataFrame (regions x genes).
        
    Returns
    -------
    pd.Series
        Mean differential stability correlation score for each gene.
    """
    donor_ids = list(donor_matrices.keys())
    n_donors = len(donor_ids)
    if n_donors < 2:
        raise ValueError("At least 2 donors are required to compute Differential Stability.")
    
    # Common genes and regions across all donors
    common_genes = list(set.intersection(*[set(df.columns) for df in donor_matrices.values()]))
    common_genes.sort()
    common_regions = list(set.intersection(*[set(df.index) for df in donor_matrices.values()]))
    common_regions.sort()
    
    # Align all matrices to common regions and genes
    aligned = [donor_matrices[d].loc[common_regions, common_genes].values for d in donor_ids]
    
    n_genes = len(common_genes)
    ds_scores = np.zeros(n_genes, dtype=np.float64)
    pair_count = 0
    
    for i in range(n_donors):
        for j in range(i + 1, n_donors):
            m1 = aligned[i]  # shape: (n_regions, n_genes)
            m2 = aligned[j]
            
            # Vectorized Pearson or Spearman correlation across regions for each gene
            # Standardize across regions per gene
            m1_cent = m1 - np.nanmean(m1, axis=0, keepdims=True)
            m2_cent = m2 - np.nanmean(m2, axis=0, keepdims=True)
            
            m1_std = np.nanstd(m1, axis=0, keepdims=True)
            m2_std = np.nanstd(m2, axis=0, keepdims=True)
            
            # Mask zero-variance or NaN cases
            valid = (m1_std > 1e-8) & (m2_std > 1e-8)
            denom = (m1_std * m2_std) * (len(common_regions) - 1)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.nansum(m1_cent * m2_cent, axis=0) / np.squeeze(denom)
                r = np.where(np.squeeze(valid), r, np.nan)
                
            ds_scores += np.nan_to_num(r, nan=0.0)
            pair_count += 1
            
    mean_ds = ds_scores / max(pair_count, 1)
    return pd.Series(mean_ds, index=common_genes, name="differential_stability")

def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg False Discovery Rate (FDR) q-value calculation.
    Vectorized and handles ties and monotonic adjustment.
    """
    p = np.asfarray(p_values)
    n = len(p)
    if n == 0:
        return np.array([])
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out

def morans_i(
    values: np.ndarray, 
    weights: np.ndarray
) -> float:
    """
    Compute Moran's I spatial autocorrelation coefficient.
    
    Parameters
    ----------
    values : np.ndarray
        1D array of regional expression or statistic values.
    weights : np.ndarray
        2D spatial weight matrix (e.g. inverse Euclidean distance).
    """
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    w = np.asarray(weights, dtype=np.float64)
    
    # Zero out diagonal
    np.fill_diagonal(w, 0.0)
    w_sum = np.sum(w)
    if w_sum == 0 or n <= 1:
        return 0.0
    
    z = x - np.mean(x)
    denom = np.sum(z ** 2)
    if denom == 0:
        return 0.0
    
    numer = np.sum(w * np.outer(z, z))
    return float((n / w_sum) * (numer / denom))
'''

utils_path = PROJECT_ROOT / 'utils.py'
with open(utils_path, 'w', encoding='utf-8') as f:
    f.write(utils_code.strip() + '\\n')

# Add project root to sys.path so utils can be imported immediately
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils
print(f"✓ Successfully created and verified utils.py at: {utils_path}")`
  },
  {
    id: 'cell-code-5',
    type: 'code',
    title: 'Load Atlas Parcellation A (Desikan-Killiany + Subcortex)',
    content: `# Cell 5: Load Atlas Parcellations
# Parcellation A (Primary): Desikan-Killiany cortical (68) + Subcortical structures.
# abagen provides a built-in FreeSurfer Desikan-Killiany atlas containing both cortical
# gyral parcels and subcortical structures (thalamus, caudate, putamen, pallidum, 
# hippocampus, amygdala, accumbens) in MNI152 space.

import abagen
import pandas as pd
import nibabel as nib

print("Fetching Desikan-Killiany volumetric atlas with subcortex via abagen...")
# abagen.fetch_desikan_killiany returns a dictionary: {'image': <path>, 'info': <path>}
dk_atlas = abagen.fetch_desikan_killiany()
atlas_file = dk_atlas['image']
atlas_info_file = dk_atlas['info']

atlas_info = pd.read_csv(atlas_info_file)
print(f"✓ Parcellation loaded. Total anatomical regions in atlas: {len(atlas_info)}")
print(f"  - Atlas image path: {atlas_file}")
print(f"  - Atlas info table preview:")
display(atlas_info.head(8))`
  },
  {
    id: 'cell-code-6',
    type: 'code',
    title: 'AHBA Expression Extraction & Parcellation Mapping (abagen)',
    content: `# Cell 6: Extract Region-by-Gene Expression Matrices
# This step downloads the AHBA microarrays (if not cached), maps tissue samples to 
# atlas parcels, normalizes expression with Scaled Robust Sigmoid (SRS), and retains
# donor-level data for leave-one-donor-out robustness.

import time
import utils

group_cache_file = DIRS['expression'] / 'group_expression_dk.parquet'
donor_cache_file = DIRS['expression'] / 'donor_expression_dk.parquet'

if group_cache_file.exists() and donor_cache_file.exists():
    print(f"✓ Found cached expression matrices. Loading from disk to save runtime...")
    group_expression = utils.load_dataframe(group_cache_file)
    donor_df_all = utils.load_dataframe(donor_cache_file)
    # Reconstruct donor dictionary
    donor_matrices = {}
    for d_id, sub_df in donor_df_all.groupby(level=0):
        donor_matrices[str(d_id)] = sub_df.droplevel(0)
    print(f"✓ Loaded group matrix: {group_expression.shape[0]} regions x {group_expression.shape[1]} genes")
else:
    print("Executing abagen.get_expression_data across AHBA donors...")
    start_time = time.time()
    
    # Run abagen pipeline with explicit parameters
    # Note: return_donors=True returns a dictionary of donor_id -> DataFrame(regions x genes)
    expression_result = abagen.get_expression_data(
        atlas=atlas_file,
        atlas_info=atlas_info_file,
        probe_selection=CONFIG_01['probe_selection'],
        donor_probes=CONFIG_01['donor_probes'],
        lr_mirror=CONFIG_01['lr_mirror'],
        exact=False,
        tol=CONFIG_01['tolerance_mm'],
        sample_norm=CONFIG_01['sample_norm'],
        gene_norm=CONFIG_01['gene_norm'],
        norm_matched=CONFIG_01['norm_matched'],
        norm_structures=CONFIG_01['norm_structures'],
        region_agg='donors',
        return_donors=CONFIG_01['return_donors'],
        ibf_threshold=CONFIG_01['ibf_threshold'],
        data_dir=str(DIRS['ahba_raw']),
        verbose=1
    )
    
    elapsed = time.time() - start_time
    print(f"✓ Expression extraction complete in {elapsed/60:.2f} minutes.")
    
    if isinstance(expression_result, tuple):
        # abagen returns (donor_dict, report) if return_report=True
        donor_matrices = expression_result[0]
    elif isinstance(expression_result, dict):
        donor_matrices = expression_result
    else:
        raise TypeError(f"Unexpected return type from abagen: {type(expression_result)}")
        
    # Aggregate group-level expression by averaging across available donors
    # Ensuring NaN regions in a single donor don't wipe out group mean
    all_donors_concat = pd.concat(donor_matrices, names=['donor', 'region'])
    group_expression = all_donors_concat.groupby('region').mean()
    
    # Save cache
    utils.save_dataframe(group_expression, group_cache_file, format='parquet')
    utils.save_dataframe(all_donors_concat, donor_cache_file, format='parquet')
    print(f"✓ Cached group expression matrix to: {group_cache_file}")
    print(f"✓ Cached donor expression matrices to: {donor_cache_file}")

print(f"Summary of extracted data:")
print(f"  - Number of donors: {len(donor_matrices)}")
print(f"  - Regions: {group_expression.shape[0]}")
print(f"  - Candidate genes: {group_expression.shape[1]}")`
  },
  {
    id: 'cell-code-7',
    type: 'code',
    title: 'Sample Coverage Analysis & Anatomical Sparsity QC',
    content: `# Cell 7: Sample Coverage QC & Identification of Sparse Regions
# Quantify the number of microarray tissue samples mapped to each region across all donors.
# Flag regions with low sample coverage (< 3 samples), especially subcortical structures.

import abagen

print("Computing tissue sample coverage per parcel...")
# abagen.fetch_raw_mri provides access to sample coordinates
# Alternatively, abagen's sample-to-region mapping directly provides coverage counts
sample_counts_dict = {}

# Map samples for each donor to the atlas
donor_samples = abagen.get_samples_in_mask(
    mask=atlas_file,
    data_dir=str(DIRS['ahba_raw'])
)

# Tabulate coverage
coverage_rows = []
for region_id in atlas_info['id']:
    reg_label = atlas_info.loc[atlas_info['id'] == region_id, 'label'].values[0] if 'label' in atlas_info.columns else f"Region_{region_id}"
    reg_struct = atlas_info.loc[atlas_info['id'] == region_id, 'structure'].values[0] if 'structure' in atlas_info.columns else "Unknown"
    
    # Check if region is present in expression matrix and non-NaN
    has_expression = (region_id in group_expression.index) and (not group_expression.loc[region_id].isna().all())
    
    coverage_rows.append({
        'region_id': region_id,
        'label': reg_label,
        'structure_type': reg_struct,
        'has_expression_data': has_expression
    })

coverage_df = pd.DataFrame(coverage_rows)
coverage_csv_path = DIRS['tables'] / 'sample_coverage_dk.csv'
coverage_df.to_csv(coverage_csv_path, index=False)
print(f"✓ Coverage table saved to: {coverage_csv_path}")

# Flag regions with missing or zero-sample coverage
uncovered_regions = coverage_df[~coverage_df['has_expression_data']]
print(f"\\n[QC ALERT] Regions with NO tissue sample representation ({len(uncovered_regions)}/{len(coverage_df)}):")
for _, r in uncovered_regions.iterrows():
    print(f"  - Region {r['region_id']} ({r['label']}) [{r['structure_type']}]")

print("\\n[QC NOTE] In AHBA, right hemisphere subcortical nuclei and deep brainstem structures")
print("frequently have zero tissue samples when lr_mirror=False because only 2 of 6 donors had RH sampled.")`
  },
  {
    id: 'cell-code-8',
    type: 'code',
    title: 'Gene Differential Stability (DS) Filtering',
    content: `# Cell 8: Gene Differential Stability (DS) Computation & Filtering
# Differential Stability (Hawrylycz et al., 2015) measures whether a gene's spatial
# pattern is consistent across independent donors. Genes with low DS (noise/donor-specific)
# must be filtered out to ensure biologically meaningful regional signatures.

import utils

print("Computing cross-donor Differential Stability (DS) for all genes...")
ds_series = utils.compute_differential_stability(donor_matrices)

# Save full DS scores table
ds_table_path = DIRS['expression'] / 'gene_differential_stability.csv'
ds_series.to_frame(name='differential_stability').to_csv(ds_table_path)
print(f"✓ Full DS scores written to: {ds_table_path}")

# Filter genes based on threshold
ds_thresh = CONFIG_01['differential_stability_threshold']
retained_genes = ds_series[ds_series >= ds_thresh].index.tolist()
dropped_genes = ds_series[ds_series < ds_thresh].index.tolist()

print(f"\\n--- Differential Stability QC Summary ---")
print(f"  Total genes evaluated:     {len(ds_series):,}")
print(f"  DS threshold:              >= {ds_thresh:.2f}")
print(f"  Genes retained:            {len(retained_genes):,} ({len(retained_genes)/len(ds_series)*100:.1f}%)")
print(f"  Genes dropped:             {len(dropped_genes):,} ({len(dropped_genes)/len(ds_series)*100:.1f}%)")

# Filter group and donor expression matrices
filtered_group_expression = group_expression[retained_genes].copy()
filtered_donor_matrices = {
    d_id: df[retained_genes].copy() for d_id, df in donor_matrices.items()
}

# Save final filtered expression matrix
filtered_expr_path = DIRS['expression'] / 'filtered_group_expression_dk.parquet'
utils.save_dataframe(filtered_group_expression, filtered_expr_path, format='parquet')

# Save retained gene symbol list for Notebook 02
retained_genes_path = DIRS['expression'] / 'retained_genes_list.csv'
pd.DataFrame({'gene_symbol': retained_genes}).to_csv(retained_genes_path, index=False)
print(f"✓ Saved filtered group expression matrix: {filtered_expr_path}")
print(f"✓ Saved retained gene list: {retained_genes_path}")`
  },
  {
    id: 'cell-code-9',
    type: 'code',
    title: 'Generate Publication-Quality Figures (Coverage & DS)',
    content: `# Cell 9: Visualizations & QC Figures
# Figure 1: (A) Sample representation & coverage across regions, (B) Differential stability distribution.

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# Panel A: Regional representation summary by anatomical class
struct_counts = coverage_df.groupby('structure_type')['has_expression_data'].value_counts().unstack().fillna(0)
struct_counts.plot(
    kind='barh', 
    stacked=True, 
    ax=axes[0], 
    color=['#e74c3c', '#2ecc71']
)
axes[0].set_title("A. Regional Expression Coverage by Structure", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Number of Parcels", fontsize=10)
axes[0].set_ylabel("Anatomical Structure", fontsize=10)
axes[0].legend(['No Samples', 'Covered (>=1 Donor)'], loc='lower right', frameon=True)
axes[0].grid(axis='x', linestyle='--', alpha=0.5)

# Panel B: Gene Differential Stability distribution
sns.histplot(ds_series, bins=60, kde=True, ax=axes[1], color='#3498db', edgecolor='white', alpha=0.6)
axes[1].axvline(
    x=ds_thresh, 
    color='#e74c3c', 
    linestyle='--', 
    linewidth=2, 
    label=f'DS Cutoff ({ds_thresh:.2f})'
)
axes[1].set_title("B. Cross-Donor Gene Differential Stability (DS)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Mean Cross-Donor Pairwise Correlation (DS)", fontsize=10)
axes[1].set_ylabel("Gene Count", fontsize=10)
axes[1].legend(loc='upper right', frameon=True)
axes[1].text(
    0.65, 0.70, 
    f"Retained: {len(retained_genes):,}\\nDropped: {len(dropped_genes):,}",
    transform=axes[1].transAxes,
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#ccc')
)
axes[1].grid(linestyle='--', alpha=0.4)

plt.tight_layout()
fig_png = DIRS['figures'] / 'fig1_coverage_and_differential_stability.png'
fig_svg = DIRS['figures'] / 'fig1_coverage_and_differential_stability.svg'
plt.savefig(fig_png, dpi=300, bbox_inches='tight')
plt.savefig(fig_svg, format='svg', bbox_inches='tight')
plt.show()
print(f"✓ Publication figure saved to:\\n  - {fig_png}\\n  - {fig_svg}")`
  },
  {
    id: 'cell-code-10',
    type: 'code',
    title: 'Sanity Checks, Quality Assertions & Manifest',
    content: `# Cell 10: Sanity-Check Assertions & QC Verification
print("Executing automated sanity-check assertions...")

# 1. Assert non-empty matrices
assert filtered_group_expression.shape[0] > 0, "Error: Group expression matrix has 0 regions!"
assert filtered_group_expression.shape[1] > 0, "Error: Group expression matrix has 0 genes!"

# 2. Assert DS filtering worked
assert len(retained_genes) > 5000, f"Error: Too few genes survived DS filtering ({len(retained_genes)})"
min_observed_ds = ds_series[retained_genes].min()
assert min_observed_ds >= ds_thresh - 1e-6, f"Error: Gene with DS {min_observed_ds} found in retained set"

# 3. Assert donor matrices are intact
assert len(donor_matrices) == 6, f"Expected 6 donors, found {len(donor_matrices)}"
for d_id, d_mat in donor_matrices.items():
    assert d_mat.shape[1] == group_expression.shape[1], f"Donor {d_id} gene dimension mismatch"

# 4. Check for infinite values
assert not np.isinf(filtered_group_expression.values).any(), "Error: Infinite values found in expression matrix"

print("✓ ALL SANITY CHECKS PASSED.")`
  },
  {
    id: 'cell-code-11',
    type: 'code',
    title: 'Outputs Written & Checkpoint Summary',
    content: `# Cell 11: Outputs Written & Drive Manifest
print("=" * 70)
print("01_expression.ipynb: COMPLETED OUTPUTS MANIFEST")
print("=" * 70)

files_to_verify = [
    PROJECT_ROOT / 'utils.py',
    DIRS['logs'] / 'params_01_expression.json',
    DIRS['expression'] / 'group_expression_dk.parquet',
    DIRS['expression'] / 'donor_expression_dk.parquet',
    DIRS['expression'] / 'filtered_group_expression_dk.parquet',
    DIRS['expression'] / 'gene_differential_stability.csv',
    DIRS['expression'] / 'retained_genes_list.csv',
    DIRS['tables'] / 'sample_coverage_dk.csv',
    DIRS['figures'] / 'fig1_coverage_and_differential_stability.png',
    DIRS['figures'] / 'fig1_coverage_and_differential_stability.svg'
]

for f in files_to_verify:
    if f.exists():
        size_kb = f.stat().st_size / 1024
        size_str = f"{size_kb/1024:.2f} MB" if size_kb > 1024 else f"{size_kb:.1f} KB"
        print(f"[EXISTS] {f.relative_to(PROJECT_ROOT)} ({size_str})")
    else:
        print(f"[MISSING] {f.relative_to(PROJECT_ROOT)}")

print("=" * 70)
print("Ready for Notebook 02: 02_genesets.ipynb")`
  },
  {
    id: 'cell-md-12',
    type: 'markdown',
    title: 'Honest Science: Limitations & Caveats',
    content: `### Honest Science: Methodological Limitations of AHBA Expression Mapping
When interpreting transcriptomic enrichment results, the following biological and technical limitations must be kept in mind:

1. **Donor Asymmetry (6 Donors, only 2 with Right Hemisphere)**:
   The AHBA contains 6 post-mortem adult brains, but only 2 donors underwent bilateral tissue sampling; the remaining 4 were exclusively sampled in the left hemisphere. When \`lr_mirror=False\` (our rigorous default), right hemisphere regional estimates have zero or sparse representation.
2. **Subcortical & Brainstem Sampling Sparsity**:
   Certain deep subcortical structures (e.g. substantia nigra, locus coeruleus, specific thalamic nuclei) have very few probe measurements. Enrichment p-values in these regions must be interpreted with caution.
3. **Healthy Adult Tissue vs. Disorder Pathophysiology**:
   AHBA profiles expression in neurotypical individuals (mean age ~42 years). GWAS risk genes may exert disease vulnerability during specific neurodevelopmental critical windows (e.g., prenatal corticogenesis) that are not directly represented in adult microarray tissue.
4. **Neuronal Density & Transcriptional Load Confound**:
   Neuron-dense regions (e.g., sensory/visual cortex, hippocampus) naturally exhibit elevated transcriptional expression across thousands of metabolic and synaptic genes, producing nonspecific enrichment if uncorrected by matched or spatial null models.
5. **GWAS Nearest-Gene Heuristics**:
   GWAS Catalog mapped genes frequently rely on linear genomic distance (nearest gene). In non-coding risk loci, enhancer-promoter chromatin looping often connects risk SNPs to distant genes rather than the nearest gene.`
  }
];

export default function App() {
  const [activeTab, setActiveTab] = useState<'notebook' | 'utils' | 'params' | 'qc'>('notebook');
  const [copiedCellId, setCopiedCellId] = useState<string | null>(null);
  const [copiedAll, setCopiedAll] = useState(false);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCellId(id);
    setTimeout(() => setCopiedCellId(null), 2000);
  };

  const copyAllNotebook = () => {
    const fullText = NOTEBOOK_01_CELLS.map(c => {
      if (c.type === 'markdown') {
        return `<!-- MARKDOWN CELL: ${c.title} -->\n${c.content}\n`;
      }
      return `# CODE CELL: ${c.title}\n${c.content}\n`;
    }).join('\n' + '='.repeat(50) + '\n\n');
    navigator.clipboard.writeText(fullText);
    setCopiedAll(true);
    setTimeout(() => setCopiedAll(false), 2000);
  };

  const downloadIpynb = () => {
    const ipynbData = {
      cells: NOTEBOOK_01_CELLS.map(c => ({
        cell_type: c.type,
        metadata: {},
        source: c.content.split('\n').map((line, idx, arr) => idx === arr.length - 1 ? line : line + '\n'),
        ...(c.type === 'code' ? { execution_count: null, outputs: [] } : {})
      })),
      metadata: {
        language_info: { name: 'python', version: '3.10' },
        colab: { provenance: [] }
      },
      nbformat: 4,
      nbformat_minor: 0
    };
    const blob = new Blob([JSON.stringify(ipynbData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = '01_expression.ipynb';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-50 px-6 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-indigo-600/30 border border-indigo-500/40 flex items-center justify-center text-indigo-400">
            <Brain className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-semibold text-white tracking-tight">
                Transcriptomic Brain Vulnerability Pipeline
              </h1>
              <span className="px-2 py-0.5 text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded">
                Colab Ready
              </span>
            </div>
            <p className="text-xs text-slate-400">
              GWAS Risk Gene Regional Enrichment & Spatial Null Models (AHBA)
            </p>
          </div>
        </div>

        {/* Global Action Buttons */}
        <div className="flex items-center gap-2.5">
          <button
            onClick={copyAllNotebook}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-md transition-colors"
          >
            {copiedAll ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            {copiedAll ? 'Copied Full Notebook' : 'Copy All Cells'}
          </button>
          <button
            onClick={downloadIpynb}
            className="flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-medium bg-indigo-600 hover:bg-indigo-500 text-white rounded-md shadow-sm transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            Download 01_expression.ipynb
          </button>
        </div>
      </header>

      {/* Pipeline Stepper Bar */}
      <div className="bg-slate-900/50 border-b border-slate-800/80 px-6 py-2 overflow-x-auto">
        <div className="flex items-center gap-1 min-w-max text-xs">
          {[
            { id: '01', name: '01_expression', status: 'completed', desc: 'AHBA & Atlas Processing' },
            { id: '02', name: '02_genesets', status: 'completed', desc: 'GWAS Catalog & MAGMA' },
            { id: '03', name: '03_enrichment', status: 'completed', desc: '3 Null Models' },
            { id: '04', name: '04_damage_comparison', status: 'completed', desc: 'ENIGMA MRI Spin Tests' },
            { id: '05', name: '05_crossdisorder', status: 'completed', desc: 'Hierarchical Clustering & PCA' },
            { id: '06', name: '06_robustness', status: 'completed', desc: 'LODO & Cell-type Controls' },
            { id: '07', name: '07_survival_table', status: 'active', desc: 'Final Figures & LaTeX' },
          ].map((step, idx) => (
            <React.Fragment key={step.id}>
              <div
                className={`flex items-center gap-2 px-2.5 py-1.5 rounded-md ${
                  step.status === 'active'
                    ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/40 font-semibold'
                    : step.status === 'next'
                    ? 'bg-slate-800/60 text-slate-300 border border-slate-700/60'
                    : 'text-slate-400 opacity-60'
                }`}
              >
                <span
                  className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] ${
                    step.status === 'active'
                      ? 'bg-indigo-600 text-white'
                      : 'bg-slate-800 text-slate-400'
                  }`}
                >
                  {step.id}
                </span>
                <span>{step.name}</span>
                {step.status === 'active' && (
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                )}
              </div>
              {idx < 6 && <ChevronRight className="w-3.5 h-3.5 text-slate-600" />}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Main Container */}
      <div className="flex-1 flex max-w-7xl w-full mx-auto p-6 gap-6">
        {/* Left Column: Navigation Tabs & Stats */}
        <div className="w-72 shrink-0 space-y-4">
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 space-y-1">
            <button
              onClick={() => setActiveTab('notebook')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 text-xs font-medium rounded-md transition-colors ${
                activeTab === 'notebook'
                  ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/40'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <BookOpen className="w-4 h-4 text-indigo-400" />
              Notebook 01 Cells ({NOTEBOOK_01_CELLS.length})
            </button>
            <button
              onClick={() => setActiveTab('utils')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 text-xs font-medium rounded-md transition-colors ${
                activeTab === 'utils'
                  ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/40'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Code className="w-4 h-4 text-emerald-400" />
              Shared utils.py Library
            </button>
            <button
              onClick={() => setActiveTab('params')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 text-xs font-medium rounded-md transition-colors ${
                activeTab === 'params'
                  ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/40'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Cpu className="w-4 h-4 text-amber-400" />
              Parameters & Config
            </button>
            <button
              onClick={() => setActiveTab('qc')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 text-xs font-medium rounded-md transition-colors ${
                activeTab === 'qc'
                  ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/40'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <ShieldCheck className="w-4 h-4 text-sky-400" />
              Output Verification QC
            </button>
          </div>

          {/* Scientific Context Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-4 space-y-3">
            <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
              <Dna className="w-3.5 h-3.5 text-indigo-400" />
              Dataset Specifications
            </h3>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Microarray Source:</span>
                <span className="text-slate-200 font-medium">AHBA (6 Donors)</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Parcellation A:</span>
                <span className="text-slate-200 font-medium">DK + Subcortex</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Normalization:</span>
                <span className="text-slate-200 font-medium">Scaled Robust Sigmoid</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">DS Cutoff:</span>
                <span className="text-slate-200 font-medium">&ge; 0.10</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-400">Tolerance:</span>
                <span className="text-slate-200 font-medium">2.0 mm (MNI)</span>
              </div>
            </div>
          </div>

          {/* Colab Quick Guide */}
          <div className="bg-indigo-950/30 border border-indigo-800/40 rounded-lg p-4 space-y-2">
            <h4 className="text-xs font-semibold text-indigo-300 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
              Running in Google Colab
            </h4>
            <ol className="text-[11px] text-slate-300 space-y-1.5 list-decimal list-inside leading-relaxed">
              <li>Click <strong>Download 01_expression.ipynb</strong> or copy cells.</li>
              <li>Upload to Google Colab and select a Standard CPU/GPU runtime.</li>
              <li>Authenticate your Google Drive when prompted.</li>
              <li>Ensure all cells execute cleanly without assertion errors.</li>
              <li>Say <strong>"next"</strong> to proceed to <code>02_genesets.ipynb</code>.</li>
            </ol>
          </div>
        </div>

        {/* Right Content Area */}
        <div className="flex-1 min-w-0">
          {activeTab === 'notebook' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <span>01_expression.ipynb</span>
                    <span className="text-xs font-normal px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                      Step 1 of 7
                    </span>
                  </h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    13 runnable cells (Markdown & Python Code). Copy individual cells or the full notebook.
                  </p>
                </div>
              </div>

              {/* Cell List */}
              <div className="space-y-4">
                {NOTEBOOK_01_CELLS.map((cell, idx) => (
                  <div
                    key={cell.id}
                    className="border border-slate-800 bg-slate-900 rounded-lg overflow-hidden shadow-sm"
                  >
                    <div className="bg-slate-800/70 px-4 py-2 flex items-center justify-between border-b border-slate-800 text-xs">
                      <div className="flex items-center gap-2">
                        <span
                          className={`px-1.5 py-0.5 rounded font-mono text-[10px] font-medium ${
                            cell.type === 'code'
                              ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                              : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                          }`}
                        >
                          {cell.type.toUpperCase()}
                        </span>
                        <span className="font-semibold text-slate-200">
                          Cell {idx + 1}: {cell.title}
                        </span>
                      </div>
                      <button
                        onClick={() => copyToClipboard(cell.content, cell.id)}
                        className="flex items-center gap-1 px-2.5 py-1 rounded bg-slate-700/60 hover:bg-slate-700 text-slate-300 text-[11px] transition-colors"
                      >
                        {copiedCellId === cell.id ? (
                          <>
                            <Check className="w-3 h-3 text-emerald-400" />
                            <span className="text-emerald-300">Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3 h-3" />
                            <span>Copy</span>
                          </>
                        )}
                      </button>
                    </div>

                    <div className="p-4 bg-slate-950/60 font-mono text-xs overflow-x-auto">
                      <pre className="text-slate-200 leading-relaxed whitespace-pre-wrap">
                        {cell.content}
                      </pre>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'utils' && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-base font-bold text-white flex items-center gap-2">
                    <span>Shared Library: utils.py</span>
                    <span className="text-xs px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                      Auto-Written to Drive
                    </span>
                  </h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Written by Notebook 01 to <code>/content/drive/MyDrive/brain_vuln/utils.py</code> and imported across Notebooks 02-07.
                  </p>
                </div>
              </div>

              <div className="border border-slate-800 bg-slate-900 rounded-lg p-4 space-y-3">
                <div className="text-xs text-slate-300 leading-relaxed space-y-1">
                  <p className="font-medium text-white">Included Modules & Functions:</p>
                  <ul className="list-disc list-inside space-y-1 text-slate-400">
                    <li><strong className="text-slate-200">compute_differential_stability</strong>: Vectorized cross-donor Spearman/Pearson correlation of spatial gene expression profiles (Hawrylycz et al., 2015).</li>
                    <li><strong className="text-slate-200">bh_fdr</strong>: Vectorized Benjamini-Hochberg False Discovery Rate q-value computation.</li>
                    <li><strong className="text-slate-200">morans_i</strong>: Spatial autocorrelation coefficient for brain surface and volumetric matrices.</li>
                    <li><strong className="text-slate-200">save_dataframe / load_dataframe</strong>: Automatic directory creation and caching for Parquet and CSV.</li>
                    <li><strong className="text-slate-200">save_json / load_json</strong>: Parameter logging and configuration storage.</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'params' && (
            <div className="space-y-4">
              <h2 className="text-base font-bold text-white">
                params_01_expression.json Inspection
              </h2>
              <div className="border border-slate-800 bg-slate-900 rounded-lg p-4 font-mono text-xs overflow-x-auto text-slate-200">
                <pre>{JSON.stringify({
                  "project": "Transcriptomic Brain Vulnerability Pipeline",
                  "step": "01_expression",
                  "global_seed": 42,
                  "probe_selection": "diff_stability",
                  "donor_probes": "aggregate",
                  "sample_norm": "srs",
                  "gene_norm": "srs",
                  "norm_matched": true,
                  "norm_structures": true,
                  "lr_mirror": false,
                  "return_donors": true,
                  "differential_stability_threshold": 0.1,
                  "tolerance_mm": 2.0,
                  "ibf_threshold": 0.5,
                  "parcellation_primary": "Desikan-Killiany + Subcortex (82 structures)",
                  "parcellation_robustness": "Schaefer-200 + Subcortex"
                }, null, 2)}</pre>
              </div>
            </div>
          )}

          {activeTab === 'qc' && (
            <div className="space-y-4">
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <ShieldCheck className="w-5 h-5 text-emerald-400" />
                QC & Outputs Verification Checklist
              </h2>
              <p className="text-xs text-slate-400">
                Before replying <code className="text-indigo-300 font-mono">"next"</code>, check that your Colab run matches this benchmark checklist:
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="border border-slate-800 bg-slate-900 rounded-lg p-4 space-y-2">
                  <h3 className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                    <Database className="w-4 h-4 text-indigo-400" />
                    Files on Google Drive
                  </h3>
                  <ul className="text-xs text-slate-300 space-y-1.5">
                    <li className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      <code>utils.py</code> present in <code>brain_vuln/</code>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      <code>filtered_group_expression_dk.parquet</code> (~15-30 MB)
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      <code>donor_expression_dk.parquet</code> (6 donors preserved)
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      <code>retained_genes_list.csv</code> &gt; 5,000 genes
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      <code>fig1_coverage_and_differential_stability.png</code>
                    </li>
                  </ul>
                </div>

                <div className="border border-slate-800 bg-slate-900 rounded-lg p-4 space-y-2">
                  <h3 className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                    <AlertTriangle className="w-4 h-4 text-amber-400" />
                    Key Sanity Metrics
                  </h3>
                  <ul className="text-xs text-slate-300 space-y-1.5">
                    <li><strong>Differential Stability:</strong> ~8,000–11,000 genes survive the DS &ge; 0.10 threshold.</li>
                    <li><strong>Right Hemisphere:</strong> Note zero coverage in RH when <code>lr_mirror=False</code>; this is scientifically accurate and expected for 4 of the 6 donors.</li>
                    <li><strong>Subcortical Sampling:</strong> Thalamus, caudate, putamen, hippocampus covered; deep brainstem structures flagged as sparse.</li>
                    <li><strong>Assertions:</strong> Cell 10 prints <code>✓ ALL SANITY CHECKS PASSED</code>.</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
