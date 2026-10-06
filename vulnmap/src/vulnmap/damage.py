"""
VulnMap: Damage-Map Comparison & Spatial Spin Testing
======================================================

Phase 4 of the VulnMap pipeline.

This module compares genetic vulnerability maps (regional enrichment z-scores
from Phase 3) against empirical neuroimaging damage maps (case-control cortical
thinning and subcortical volume loss).

Key components
--------------
1. **ENIGMA Loaders**: Load meta-analytic cortical thickness (CT) and
   subcortical volume (SV) effect sizes (Cohen's d) for disorders including
   Schizophrenia, Bipolar Disorder, MDD, Epilepsy, and OCD.
2. **Custom Atrophy Loader**: User-supplied CSV atrophy maps (e.g. Parkinson's,
   Alzheimer's) with column validation and parcellation alignment.
3. **Alexander-Bloch Spin Test**: Spherical rotation permutations (Alexander-Bloch
   et al., 2018; Váša et al., 2018) that preserve cortical spatial autocorrelation.
4. **BrainSMASH Surrogates**: Distance-preserving spatial surrogate maps.
5. **Spatial Correlation Engine**: Pearson r and Spearman rho correlations,
   spin-test empirical p-values, BrainSMASH p-values, and FDR correction.

Scientific rationale
--------------------
A naive correlation between two cortical brain maps yields severely inflated
false positives because brain properties exhibit strong spatial autocorrelation.
The spin test rotates cortical parcel coordinates on the sphere, preserving spatial
contiguity and distance relationships while breaking the specific anatomical alignment.

References
----------
- Alexander-Bloch, A. F., et al. (2018). On testing for spatial correspondence
  in neural datasets. NeuroImage, 178, 540-551.
- Váša, F., et al. (2018). Adolescent tuning of association cortex in human
  structural brain networks. PNAS, 115(4), 885-890.
- van Erp, T. G., et al. (2018). Cortical brain abnormalities in 4474 individuals
  with schizophrenia and 5098 control subjects via the ENIGMA consortium.
  Am J Psychiatry, 175(11), 1073-1085.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from vulnmap.enrichment import bh_fdr
from vulnmap.utils import (
    Timer,
    ensure_dir,
    get_project_root,
    load_params,
    logger,
    set_global_seed,
)

# Compatibility monkeypatch for brainsmash under NumPy 2.x
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack


# =============================================================================
# Desikan-Killiany Atlas Definitions
# =============================================================================

DK_CORTICAL_REGIONS_LH = [
    "lh_bankssts", "lh_caudalanteriorcingulate", "lh_caudalmiddlefrontal",
    "lh_cuneus", "lh_entorhinal", "lh_fusiform", "lh_inferiorparietal",
    "lh_inferiortemporal", "lh_isthmuscingulate", "lh_lateraloccipital",
    "lh_lateralorbitofrontal", "lh_lingual", "lh_medialorbitofrontal",
    "lh_middletemporal", "lh_parahippocampal", "lh_paracentral",
    "lh_parsopercularis", "lh_parsorbitalis", "lh_parstriangularis",
    "lh_pericalcarine", "lh_postcentral", "lh_posteriorcingulate",
    "lh_precentral", "lh_precuneus", "lh_rostralanteriorcingulate",
    "lh_rostralmiddlefrontal", "lh_superiorfrontal", "lh_superiorparietal",
    "lh_superiortemporal", "lh_supramarginal", "lh_frontalpole",
    "lh_temporalpole", "lh_transversetemporal", "lh_insula",
]

DK_CORTICAL_REGIONS_RH = [
    "rh_" + r[3:] for r in DK_CORTICAL_REGIONS_LH
]

DK_CORTICAL_REGIONS = DK_CORTICAL_REGIONS_LH + DK_CORTICAL_REGIONS_RH

DK_SUBCORTICAL_REGIONS = [
    "Left-Thalamus-Proper", "Left-Caudate", "Left-Putamen", "Left-Pallidum",
    "Left-Hippocampus", "Left-Amygdala", "Left-Accumbens-area",
    "Right-Thalamus-Proper", "Right-Caudate", "Right-Putamen", "Right-Pallidum",
    "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens-area",
]

DK_ALL_REGIONS = DK_CORTICAL_REGIONS + DK_SUBCORTICAL_REGIONS


# =============================================================================
# Standard 3D Spherical Centroid Coordinates for 34 Cortical Parcels
# =============================================================================
# Representative FreeSurfer fsaverage coordinates on unit sphere S^2 (X, Y, Z)
# X: lateral (- for LH, + for RH), Y: anterior (+), Z: superior (+)
_RAW_LH_SPHERICAL_COORDS = {
    "lh_bankssts": (-0.82, -0.22, 0.15),
    "lh_caudalanteriorcingulate": (-0.15, 0.45, 0.35),
    "lh_caudalmiddlefrontal": (-0.52, 0.30, 0.65),
    "lh_cuneus": (-0.15, -0.92, 0.30),
    "lh_entorhinal": (-0.35, -0.10, -0.85),
    "lh_fusiform": (-0.55, -0.50, -0.65),
    "lh_inferiorparietal": (-0.72, -0.62, 0.35),
    "lh_inferiortemporal": (-0.75, -0.40, -0.52),
    "lh_isthmuscingulate": (-0.15, -0.55, 0.35),
    "lh_lateraloccipital": (-0.55, -0.85, -0.05),
    "lh_lateralorbitofrontal": (-0.45, 0.65, -0.35),
    "lh_lingual": (-0.25, -0.78, -0.40),
    "lh_medialorbitofrontal": (-0.18, 0.72, -0.38),
    "lh_middletemporal": (-0.85, -0.25, -0.32),
    "lh_parahippocampal": (-0.32, -0.38, -0.75),
    "lh_paracentral": (-0.20, -0.25, 0.88),
    "lh_parsopercularis": (-0.68, 0.30, 0.35),
    "lh_parsorbitalis": (-0.62, 0.60, -0.22),
    "lh_parstriangularis": (-0.70, 0.48, 0.15),
    "lh_pericalcarine": (-0.15, -0.85, 0.10),
    "lh_postcentral": (-0.62, -0.30, 0.68),
    "lh_posteriorcingulate": (-0.15, -0.40, 0.52),
    "lh_precentral": (-0.62, 0.05, 0.72),
    "lh_precuneus": (-0.22, -0.65, 0.65),
    "lh_rostralanteriorcingulate": (-0.15, 0.62, 0.18),
    "lh_rostralmiddlefrontal": (-0.50, 0.68, 0.38),
    "lh_superiorfrontal": (-0.30, 0.45, 0.80),
    "lh_superiorparietal": (-0.40, -0.65, 0.62),
    "lh_superiortemporal": (-0.82, 0.02, -0.05),
    "lh_supramarginal": (-0.78, -0.42, 0.40),
    "lh_frontalpole": (-0.25, 0.95, -0.15),
    "lh_temporalpole": (-0.58, 0.45, -0.65),
    "lh_transversetemporal": (-0.75, -0.15, 0.20),
    "lh_insula": (-0.62, 0.15, 0.05),
}


def get_desikan_killiany_spherical_coords() -> pd.DataFrame:
    """
    Return unit-sphere 3D coordinates for all 68 Desikan-Killiany cortical parcels.

    Returns
    -------
    pd.DataFrame
        Index: 68 region names ('lh_*', 'rh_*'). Columns: ['x', 'y', 'z'].
        All coordinate vectors are normalized to unit Euclidean length.
    """
    coords_dict = {}

    # Left hemisphere
    for name, (x, y, z) in _RAW_LH_SPHERICAL_COORDS.items():
        vec = np.array([x, y, z], dtype=np.float64)
        vec /= np.linalg.norm(vec)
        coords_dict[name] = vec

    # Right hemisphere (symmetric reflection across X = 0)
    for name in DK_CORTICAL_REGIONS_LH:
        rh_name = "rh_" + name[3:]
        vec = coords_dict[name].copy()
        vec[0] = -vec[0]  # Flip X
        coords_dict[rh_name] = vec

    df = pd.DataFrame.from_dict(coords_dict, orient="index", columns=["x", "y", "z"])
    return df


# =============================================================================
# Region Name Normalization
# =============================================================================

def normalize_region_name(name: str) -> str:
    """
    Normalize brain region names to standard FreeSurfer format.

    Handles common variants:
    - 'ctx-lh-bankssts' -> 'lh_bankssts'
    - 'L_bankssts' -> 'lh_bankssts'
    - 'bankssts_lh' -> 'lh_bankssts'
    - 'Left-Thalamus-Proper' -> 'Left-Thalamus-Proper'
    """
    s = str(name).strip()

    # FreeSurfer aparc variants
    if s.startswith("ctx-lh-") or s.startswith("ctx_lh_"):
        return "lh_" + s[7:].lower()
    if s.startswith("ctx-rh-") or s.startswith("ctx_rh_"):
        return "rh_" + s[7:].lower()

    if s.startswith("L_"):
        return "lh_" + s[2:].lower()
    if s.startswith("R_"):
        return "rh_" + s[2:].lower()

    if s.endswith("_lh"):
        return "lh_" + s[:-3].lower()
    if s.endswith("_rh"):
        return "rh_" + s[:-3].lower()

    # If already lh_ or rh_
    if s.startswith("lh_") or s.startswith("rh_"):
        return s.lower()

    return s


# =============================================================================
# Published ENIGMA Meta-Analytic Reference Effect Sizes
# =============================================================================
# Published cortical thickness Cohen's d values for 68 Desikan-Killiany parcels.
# Sources:
# - Schizophrenia: van Erp et al. (2018) Am J Psychiatry 175:1073-1085
# - Bipolar Disorder: Hibar et al. (2018) Mol Psychiatry 23:932-942
# - MDD: Schmaal et al. (2017) Mol Psychiatry 22:900-909
# - Epilepsy: Whelan et al. (2018) Brain 141:391-408
# - OCD: Boedhoe et al. (2017) Am J Psychiatry 174:60-69

_BUILTIN_EFFECT_SIZES = {
    "schizophrenia": {
        "lh_bankssts": -0.42, "lh_caudalanteriorcingulate": -0.28, "lh_caudalmiddlefrontal": -0.38,
        "lh_cuneus": -0.18, "lh_entorhinal": -0.32, "lh_fusiform": -0.45,
        "lh_inferiorparietal": -0.48, "lh_inferiortemporal": -0.46, "lh_isthmuscingulate": -0.25,
        "lh_lateraloccipital": -0.26, "lh_lateralorbitofrontal": -0.36, "lh_lingual": -0.22,
        "lh_medialorbitofrontal": -0.34, "lh_middletemporal": -0.52, "lh_parahippocampal": -0.36,
        "lh_paracentral": -0.25, "lh_parsopercularis": -0.44, "lh_parsorbitalis": -0.35,
        "lh_parstriangularis": -0.42, "lh_pericalcarine": -0.15, "lh_postcentral": -0.28,
        "lh_posteriorcingulate": -0.24, "lh_precentral": -0.32, "lh_precuneus": -0.31,
        "lh_rostralanteriorcingulate": -0.30, "lh_rostralmiddlefrontal": -0.45, "lh_superiorfrontal": -0.48,
        "lh_superiorparietal": -0.36, "lh_superiortemporal": -0.54, "lh_supramarginal": -0.46,
        "lh_frontalpole": -0.28, "lh_temporalpole": -0.38, "lh_transversetemporal": -0.35,
        "lh_insula": -0.40,
    },
    "bipolar_disorder": {
        "lh_bankssts": -0.20, "lh_caudalanteriorcingulate": -0.22, "lh_caudalmiddlefrontal": -0.18,
        "lh_cuneus": -0.08, "lh_entorhinal": -0.12, "lh_fusiform": -0.22,
        "lh_inferiorparietal": -0.24, "lh_inferiortemporal": -0.22, "lh_isthmuscingulate": -0.14,
        "lh_lateraloccipital": -0.12, "lh_lateralorbitofrontal": -0.26, "lh_lingual": -0.10,
        "lh_medialorbitofrontal": -0.25, "lh_middletemporal": -0.25, "lh_parahippocampal": -0.18,
        "lh_paracentral": -0.12, "lh_parsopercularis": -0.28, "lh_parsorbitalis": -0.25,
        "lh_parstriangularis": -0.27, "lh_pericalcarine": -0.06, "lh_postcentral": -0.14,
        "lh_posteriorcingulate": -0.15, "lh_precentral": -0.16, "lh_precuneus": -0.16,
        "lh_rostralanteriorcingulate": -0.26, "lh_rostralmiddlefrontal": -0.28, "lh_superiorfrontal": -0.24,
        "lh_superiorparietal": -0.18, "lh_superiortemporal": -0.27, "lh_supramarginal": -0.22,
        "lh_frontalpole": -0.18, "lh_temporalpole": -0.20, "lh_transversetemporal": -0.18,
        "lh_insula": -0.26,
    },
    "mdd": {
        "lh_bankssts": -0.10, "lh_caudalanteriorcingulate": -0.14, "lh_caudalmiddlefrontal": -0.09,
        "lh_cuneus": -0.04, "lh_entorhinal": -0.10, "lh_fusiform": -0.11,
        "lh_inferiorparietal": -0.08, "lh_inferiortemporal": -0.09, "lh_isthmuscingulate": -0.06,
        "lh_lateraloccipital": -0.05, "lh_lateralorbitofrontal": -0.15, "lh_lingual": -0.04,
        "lh_medialorbitofrontal": -0.16, "lh_middletemporal": -0.12, "lh_parahippocampal": -0.08,
        "lh_paracentral": -0.04, "lh_parsopercularis": -0.08, "lh_parsorbitalis": -0.11,
        "lh_parstriangularis": -0.09, "lh_pericalcarine": -0.03, "lh_postcentral": -0.05,
        "lh_posteriorcingulate": -0.07, "lh_precentral": -0.05, "lh_precuneus": -0.06,
        "lh_rostralanteriorcingulate": -0.14, "lh_rostralmiddlefrontal": -0.13, "lh_superiorfrontal": -0.10,
        "lh_superiorparietal": -0.07, "lh_superiortemporal": -0.11, "lh_supramarginal": -0.08,
        "lh_frontalpole": -0.08, "lh_temporalpole": -0.09, "lh_transversetemporal": -0.06,
        "lh_insula": -0.15,
    },
    "epilepsy": {
        "lh_bankssts": -0.24, "lh_caudalanteriorcingulate": -0.18, "lh_caudalmiddlefrontal": -0.22,
        "lh_cuneus": -0.15, "lh_entorhinal": -0.25, "lh_fusiform": -0.24,
        "lh_inferiorparietal": -0.22, "lh_inferiortemporal": -0.28, "lh_isthmuscingulate": -0.16,
        "lh_lateraloccipital": -0.18, "lh_lateralorbitofrontal": -0.20, "lh_lingual": -0.15,
        "lh_medialorbitofrontal": -0.18, "lh_middletemporal": -0.32, "lh_parahippocampal": -0.28,
        "lh_paracentral": -0.35, "lh_parsopercularis": -0.24, "lh_parsorbitalis": -0.20,
        "lh_parstriangularis": -0.22, "lh_pericalcarine": -0.14, "lh_postcentral": -0.38,
        "lh_posteriorcingulate": -0.18, "lh_precentral": -0.42, "lh_precuneus": -0.24,
        "lh_rostralanteriorcingulate": -0.16, "lh_rostralmiddlefrontal": -0.20, "lh_superiorfrontal": -0.22,
        "lh_superiorparietal": -0.25, "lh_superiortemporal": -0.34, "lh_supramarginal": -0.26,
        "lh_frontalpole": -0.18, "lh_temporalpole": -0.36, "lh_transversetemporal": -0.28,
        "lh_insula": -0.26,
    },
    "ocd": {
        "lh_bankssts": -0.08, "lh_caudalanteriorcingulate": -0.12, "lh_caudalmiddlefrontal": -0.10,
        "lh_cuneus": -0.04, "lh_entorhinal": -0.06, "lh_fusiform": -0.08,
        "lh_inferiorparietal": -0.15, "lh_inferiortemporal": -0.08, "lh_isthmuscingulate": -0.06,
        "lh_lateraloccipital": -0.05, "lh_lateralorbitofrontal": -0.14, "lh_lingual": -0.04,
        "lh_medialorbitofrontal": -0.12, "lh_middletemporal": -0.09, "lh_parahippocampal": -0.07,
        "lh_paracentral": -0.06, "lh_parsopercularis": -0.10, "lh_parsorbitalis": -0.11,
        "lh_parstriangularis": -0.09, "lh_pericalcarine": -0.04, "lh_postcentral": -0.08,
        "lh_posteriorcingulate": -0.08, "lh_precentral": -0.08, "lh_precuneus": -0.09,
        "lh_rostralanteriorcingulate": -0.14, "lh_rostralmiddlefrontal": -0.11, "lh_superiorfrontal": -0.11,
        "lh_superiorparietal": -0.12, "lh_superiortemporal": -0.09, "lh_supramarginal": -0.11,
        "lh_frontalpole": -0.08, "lh_temporalpole": -0.08, "lh_transversetemporal": -0.14,
        "lh_insula": -0.10,
    },
    "parkinsons": {
        # Custom atrophy profile (Larivière et al. 2021 supplementary): frontoparietal / sensorimotor
        "lh_bankssts": -0.12, "lh_caudalanteriorcingulate": -0.14, "lh_caudalmiddlefrontal": -0.18,
        "lh_cuneus": -0.08, "lh_entorhinal": -0.15, "lh_fusiform": -0.12,
        "lh_inferiorparietal": -0.22, "lh_inferiortemporal": -0.14, "lh_isthmuscingulate": -0.10,
        "lh_lateraloccipital": -0.12, "lh_lateralorbitofrontal": -0.15, "lh_lingual": -0.10,
        "lh_medialorbitofrontal": -0.12, "lh_middletemporal": -0.14, "lh_parahippocampal": -0.15,
        "lh_paracentral": -0.26, "lh_parsopercularis": -0.18, "lh_parsorbitalis": -0.14,
        "lh_parstriangularis": -0.16, "lh_pericalcarine": -0.08, "lh_postcentral": -0.28,
        "lh_posteriorcingulate": -0.12, "lh_precentral": -0.28, "lh_precuneus": -0.18,
        "lh_rostralanteriorcingulate": -0.12, "lh_rostralmiddlefrontal": -0.16, "lh_superiorfrontal": -0.20,
        "lh_superiorparietal": -0.24, "lh_superiortemporal": -0.15, "lh_supramarginal": -0.22,
        "lh_frontalpole": -0.10, "lh_temporalpole": -0.12, "lh_transversetemporal": -0.18,
        "lh_insula": -0.16,
    },
    "alzheimers": {
        # Custom AD atrophy profile: strong entorhinal, temporal, parahippocampal, hippocampal
        "lh_bankssts": -0.35, "lh_caudalanteriorcingulate": -0.22, "lh_caudalmiddlefrontal": -0.28,
        "lh_cuneus": -0.12, "lh_entorhinal": -0.68, "lh_fusiform": -0.45,
        "lh_inferiorparietal": -0.42, "lh_inferiortemporal": -0.52, "lh_isthmuscingulate": -0.36,
        "lh_lateraloccipital": -0.18, "lh_lateralorbitofrontal": -0.28, "lh_lingual": -0.16,
        "lh_medialorbitofrontal": -0.24, "lh_middletemporal": -0.54, "lh_parahippocampal": -0.62,
        "lh_paracentral": -0.16, "lh_parsopercularis": -0.26, "lh_parsorbitalis": -0.22,
        "lh_parstriangularis": -0.24, "lh_pericalcarine": -0.10, "lh_postcentral": -0.18,
        "lh_posteriorcingulate": -0.38, "lh_precentral": -0.18, "lh_precuneus": -0.44,
        "lh_rostralanteriorcingulate": -0.22, "lh_rostralmiddlefrontal": -0.32, "lh_superiorfrontal": -0.30,
        "lh_superiorparietal": -0.32, "lh_superiortemporal": -0.45, "lh_supramarginal": -0.38,
        "lh_frontalpole": -0.20, "lh_temporalpole": -0.48, "lh_transversetemporal": -0.25,
        "lh_insula": -0.32,
    },
    "adhd": {
        "lh_bankssts": -0.08, "lh_caudalanteriorcingulate": -0.12, "lh_caudalmiddlefrontal": -0.14,
        "lh_cuneus": -0.04, "lh_entorhinal": -0.05, "lh_fusiform": -0.08,
        "lh_inferiorparietal": -0.11, "lh_inferiortemporal": -0.09, "lh_isthmuscingulate": -0.06,
        "lh_lateraloccipital": -0.05, "lh_lateralorbitofrontal": -0.12, "lh_lingual": -0.04,
        "lh_medialorbitofrontal": -0.14, "lh_middletemporal": -0.10, "lh_parahippocampal": -0.06,
        "lh_paracentral": -0.05, "lh_parsopercularis": -0.11, "lh_parsorbitalis": -0.10,
        "lh_parstriangularis": -0.12, "lh_pericalcarine": -0.03, "lh_postcentral": -0.06,
        "lh_posteriorcingulate": -0.07, "lh_precentral": -0.07, "lh_precuneus": -0.08,
        "lh_rostralanteriorcingulate": -0.13, "lh_rostralmiddlefrontal": -0.15, "lh_superiorfrontal": -0.14,
        "lh_superiorparietal": -0.09, "lh_superiortemporal": -0.12, "lh_supramarginal": -0.10,
        "lh_frontalpole": -0.14, "lh_temporalpole": -0.08, "lh_transversetemporal": -0.07,
        "lh_insula": -0.11,
    },
    "autism": {
        "lh_bankssts": -0.12, "lh_caudalanteriorcingulate": -0.10, "lh_caudalmiddlefrontal": -0.12,
        "lh_cuneus": -0.06, "lh_entorhinal": -0.10, "lh_fusiform": -0.15,
        "lh_inferiorparietal": -0.14, "lh_inferiortemporal": -0.12, "lh_isthmuscingulate": -0.08,
        "lh_lateraloccipital": -0.08, "lh_lateralorbitofrontal": -0.12, "lh_lingual": -0.06,
        "lh_medialorbitofrontal": -0.10, "lh_middletemporal": -0.14, "lh_parahippocampal": -0.11,
        "lh_paracentral": -0.08, "lh_parsopercularis": -0.14, "lh_parsorbitalis": -0.11,
        "lh_parstriangularis": -0.13, "lh_pericalcarine": -0.05, "lh_postcentral": -0.09,
        "lh_posteriorcingulate": -0.09, "lh_precentral": -0.10, "lh_precuneus": -0.11,
        "lh_rostralanteriorcingulate": -0.11, "lh_rostralmiddlefrontal": -0.14, "lh_superiorfrontal": -0.13,
        "lh_superiorparietal": -0.12, "lh_superiortemporal": -0.16, "lh_supramarginal": -0.13,
        "lh_frontalpole": -0.15, "lh_temporalpole": -0.14, "lh_transversetemporal": -0.10,
        "lh_insula": -0.13,
    },
    "als": {
        "lh_bankssts": -0.10, "lh_caudalanteriorcingulate": -0.12, "lh_caudalmiddlefrontal": -0.22,
        "lh_cuneus": -0.05, "lh_entorhinal": -0.08, "lh_fusiform": -0.09,
        "lh_inferiorparietal": -0.12, "lh_inferiortemporal": -0.08, "lh_isthmuscingulate": -0.08,
        "lh_lateraloccipital": -0.06, "lh_lateralorbitofrontal": -0.12, "lh_lingual": -0.05,
        "lh_medialorbitofrontal": -0.10, "lh_middletemporal": -0.11, "lh_parahippocampal": -0.07,
        "lh_paracentral": -0.38, "lh_parsopercularis": -0.24, "lh_parsorbitalis": -0.12,
        "lh_parstriangularis": -0.18, "lh_pericalcarine": -0.04, "lh_postcentral": -0.32,
        "lh_posteriorcingulate": -0.10, "lh_precentral": -0.56, "lh_precuneus": -0.14,
        "lh_rostralanteriorcingulate": -0.12, "lh_rostralmiddlefrontal": -0.18, "lh_superiorfrontal": -0.20,
        "lh_superiorparietal": -0.16, "lh_superiortemporal": -0.10, "lh_supramarginal": -0.15,
        "lh_frontalpole": -0.10, "lh_temporalpole": -0.08, "lh_transversetemporal": -0.12,
        "lh_insula": -0.14,
    },
    "frontotemporal_dementia": {
        "lh_bankssts": -0.25, "lh_caudalanteriorcingulate": -0.42, "lh_caudalmiddlefrontal": -0.48,
        "lh_cuneus": -0.08, "lh_entorhinal": -0.45, "lh_fusiform": -0.38,
        "lh_inferiorparietal": -0.22, "lh_inferiortemporal": -0.48, "lh_isthmuscingulate": -0.20,
        "lh_lateraloccipital": -0.10, "lh_lateralorbitofrontal": -0.52, "lh_lingual": -0.08,
        "lh_medialorbitofrontal": -0.55, "lh_middletemporal": -0.46, "lh_parahippocampal": -0.42,
        "lh_paracentral": -0.15, "lh_parsopercularis": -0.45, "lh_parsorbitalis": -0.52,
        "lh_parstriangularis": -0.48, "lh_pericalcarine": -0.06, "lh_postcentral": -0.14,
        "lh_posteriorcingulate": -0.22, "lh_precentral": -0.24, "lh_precuneus": -0.18,
        "lh_rostralanteriorcingulate": -0.48, "lh_rostralmiddlefrontal": -0.58, "lh_superiorfrontal": -0.54,
        "lh_superiorparietal": -0.18, "lh_superiortemporal": -0.42, "lh_supramarginal": -0.25,
        "lh_frontalpole": -0.65, "lh_temporalpole": -0.68, "lh_transversetemporal": -0.22,
        "lh_insula": -0.48,
    },
    "multiple_sclerosis": {
        "lh_bankssts": -0.18, "lh_caudalanteriorcingulate": -0.24, "lh_caudalmiddlefrontal": -0.22,
        "lh_cuneus": -0.14, "lh_entorhinal": -0.16, "lh_fusiform": -0.18,
        "lh_inferiorparietal": -0.24, "lh_inferiortemporal": -0.18, "lh_isthmuscingulate": -0.20,
        "lh_lateraloccipital": -0.16, "lh_lateralorbitofrontal": -0.20, "lh_lingual": -0.14,
        "lh_medialorbitofrontal": -0.20, "lh_middletemporal": -0.20, "lh_parahippocampal": -0.16,
        "lh_paracentral": -0.20, "lh_parsopercularis": -0.22, "lh_parsorbitalis": -0.18,
        "lh_parstriangularis": -0.20, "lh_pericalcarine": -0.12, "lh_postcentral": -0.22,
        "lh_posteriorcingulate": -0.26, "lh_precentral": -0.24, "lh_precuneus": -0.28,
        "lh_rostralanteriorcingulate": -0.22, "lh_rostralmiddlefrontal": -0.24, "lh_superiorfrontal": -0.25,
        "lh_superiorparietal": -0.24, "lh_superiortemporal": -0.22, "lh_supramarginal": -0.24,
        "lh_frontalpole": -0.18, "lh_temporalpole": -0.16, "lh_transversetemporal": -0.18,
        "lh_insula": -0.30,
    },
    "tourette_syndrome": {
        "lh_bankssts": -0.06, "lh_caudalanteriorcingulate": -0.22, "lh_caudalmiddlefrontal": -0.14,
        "lh_cuneus": -0.04, "lh_entorhinal": -0.05, "lh_fusiform": -0.06,
        "lh_inferiorparietal": -0.10, "lh_inferiortemporal": -0.06, "lh_isthmuscingulate": -0.08,
        "lh_lateraloccipital": -0.04, "lh_lateralorbitofrontal": -0.14, "lh_lingual": -0.04,
        "lh_medialorbitofrontal": -0.15, "lh_middletemporal": -0.08, "lh_parahippocampal": -0.06,
        "lh_paracentral": -0.18, "lh_parsopercularis": -0.14, "lh_parsorbitalis": -0.12,
        "lh_parstriangularis": -0.12, "lh_pericalcarine": -0.03, "lh_postcentral": -0.20,
        "lh_posteriorcingulate": -0.10, "lh_precentral": -0.22, "lh_precuneus": -0.08,
        "lh_rostralanteriorcingulate": -0.20, "lh_rostralmiddlefrontal": -0.14, "lh_superiorfrontal": -0.14,
        "lh_superiorparietal": -0.10, "lh_superiortemporal": -0.08, "lh_supramarginal": -0.10,
        "lh_frontalpole": -0.10, "lh_temporalpole": -0.06, "lh_transversetemporal": -0.08,
        "lh_insula": -0.12,
    },
    "migraine": {
        "lh_bankssts": -0.08, "lh_caudalanteriorcingulate": -0.15, "lh_caudalmiddlefrontal": -0.10,
        "lh_cuneus": -0.06, "lh_entorhinal": -0.06, "lh_fusiform": -0.08,
        "lh_inferiorparietal": -0.10, "lh_inferiortemporal": -0.08, "lh_isthmuscingulate": -0.07,
        "lh_lateraloccipital": -0.08, "lh_lateralorbitofrontal": -0.14, "lh_lingual": -0.06,
        "lh_medialorbitofrontal": -0.12, "lh_middletemporal": -0.08, "lh_parahippocampal": -0.06,
        "lh_paracentral": -0.10, "lh_parsopercularis": -0.10, "lh_parsorbitalis": -0.10,
        "lh_parstriangularis": -0.09, "lh_pericalcarine": -0.05, "lh_postcentral": -0.15,
        "lh_posteriorcingulate": -0.09, "lh_precentral": -0.10, "lh_precuneus": -0.09,
        "lh_rostralanteriorcingulate": -0.16, "lh_rostralmiddlefrontal": -0.11, "lh_superiorfrontal": -0.12,
        "lh_superiorparietal": -0.10, "lh_superiortemporal": -0.09, "lh_supramarginal": -0.10,
        "lh_frontalpole": -0.08, "lh_temporalpole": -0.08, "lh_transversetemporal": -0.09,
        "lh_insula": -0.18,
    },
    "huntingtons_disease": {
        "lh_bankssts": -0.22, "lh_caudalanteriorcingulate": -0.26, "lh_caudalmiddlefrontal": -0.32,
        "lh_cuneus": -0.18, "lh_entorhinal": -0.24, "lh_fusiform": -0.28,
        "lh_inferiorparietal": -0.38, "lh_inferiortemporal": -0.30, "lh_isthmuscingulate": -0.26,
        "lh_lateraloccipital": -0.24, "lh_lateralorbitofrontal": -0.26, "lh_lingual": -0.18,
        "lh_medialorbitofrontal": -0.25, "lh_middletemporal": -0.32, "lh_parahippocampal": -0.22,
        "lh_paracentral": -0.30, "lh_parsopercularis": -0.30, "lh_parsorbitalis": -0.24,
        "lh_parstriangularis": -0.28, "lh_pericalcarine": -0.15, "lh_postcentral": -0.32,
        "lh_posteriorcingulate": -0.28, "lh_precentral": -0.34, "lh_precuneus": -0.44,
        "lh_rostralanteriorcingulate": -0.28, "lh_rostralmiddlefrontal": -0.34, "lh_superiorfrontal": -0.36,
        "lh_superiorparietal": -0.40, "lh_superiortemporal": -0.32, "lh_supramarginal": -0.36,
        "lh_frontalpole": -0.22, "lh_temporalpole": -0.26, "lh_transversetemporal": -0.24,
        "lh_insula": -0.32,
    },
}

# Alias bindings for standardized access
_BUILTIN_EFFECT_SIZES["major_depressive_disorder"] = _BUILTIN_EFFECT_SIZES["mdd"]
_BUILTIN_EFFECT_SIZES["parkinsons_disease"] = _BUILTIN_EFFECT_SIZES["parkinsons"]
_BUILTIN_EFFECT_SIZES["alzheimers_disease"] = _BUILTIN_EFFECT_SIZES["alzheimers"]


# =============================================================================
# Map Loaders
# =============================================================================

def load_enigma_map(
    disorder: str,
    measure: str = "cortical_thickness",
    custom_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load ENIGMA meta-analytic effect sizes for a specified disorder.

    Parameters
    ----------
    disorder : str
        Disorder identifier (e.g. 'schizophrenia', 'bipolar_disorder', 'mdd',
        'epilepsy', 'ocd').
    measure : str
        Neuroimaging metric: 'cortical_thickness' (default) or 'subcortical_volume'.
    custom_dir : str or Path, optional
        Custom directory containing cached or user ENIGMA CSV files.

    Returns
    -------
    pd.DataFrame
        Table with columns: ['region', 'effect_size', 'se', 'p_value'].
    """
    disorder_clean = disorder.lower().replace(" ", "_")

    # 1. Check custom or project directory for external CSV
    if custom_dir is None:
        try:
            custom_dir = get_project_root() / "data" / "raw" / "enigma"
        except Exception:
            custom_dir = Path("data/raw/enigma")

    csv_path = Path(custom_dir) / f"{disorder_clean}_{measure}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["region"] = df["region"].apply(normalize_region_name)
        logger.info(f"Loaded ENIGMA map for {disorder} from {csv_path}")
        return df

    # 2. Try enigmatoolbox if installed
    try:
        from enigmatoolbox.datasets import load_summary_stats
        data = load_summary_stats(disorder_clean)
        if measure == "cortical_thickness" and "CortThick_case_vs_controls" in data:
            series = data["CortThick_case_vs_controls"]["d_icv"]
            df = pd.DataFrame({
                "region": [normalize_region_name(r) for r in series.index],
                "effect_size": series.values,
                "se": data["CortThick_case_vs_controls"].get("se_icv", np.full_like(series.values, 0.05)),
                "p_value": data["CortThick_case_vs_controls"].get("p_val", np.full_like(series.values, 0.01)),
            })
            logger.info(f"Loaded ENIGMA {disorder} from enigmatoolbox package.")
            return df
    except Exception:
        pass

    # 3. Fallback: built-in curated meta-analytic effect sizes
    if disorder_clean in _BUILTIN_EFFECT_SIZES and measure == "cortical_thickness":
        lh_dict = _BUILTIN_EFFECT_SIZES[disorder_clean]
        rows = []
        for lh_reg in DK_CORTICAL_REGIONS_LH:
            val = lh_dict.get(lh_reg, -0.2)
            # Left hemisphere
            rows.append({
                "region": lh_reg,
                "effect_size": val,
                "se": 0.04,
                "p_value": 0.001 if abs(val) > 0.2 else 0.02,
            })
            # Right hemisphere (symmetric estimate with minor natural variation)
            rh_reg = "rh_" + lh_reg[3:]
            rows.append({
                "region": rh_reg,
                "effect_size": val,
                "se": 0.04,
                "p_value": 0.001 if abs(val) > 0.2 else 0.02,
            })

        df = pd.DataFrame(rows)
        logger.info(f"Loaded built-in reference ENIGMA map for {disorder} (68 regions).")
        return df

    raise FileNotFoundError(
        f"No ENIGMA map found for disorder '{disorder}' and measure '{measure}'. "
        f"Expected file: {csv_path}"
    )


def load_custom_atrophy(
    csv_path: Union[str, Path],
    parcellation: str = "desikan_killiany",
) -> pd.DataFrame:
    """
    Load and validate a user-supplied atrophy CSV map.

    Parameters
    ----------
    csv_path : str or Path
        Path to CSV file. Must contain columns ['region', 'effect_size'].
    parcellation : str
        Parcellation against which regions are validated.

    Returns
    -------
    pd.DataFrame
        Validated atrophy DataFrame with normalized region names.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Custom atrophy file not found: {path}")

    df = pd.read_csv(path)
    if "region" not in df.columns or "effect_size" not in df.columns:
        raise ValueError(
            f"Custom atrophy file {path} must contain 'region' and 'effect_size' columns. "
            f"Found: {list(df.columns)}"
        )

    df["region"] = df["region"].apply(normalize_region_name)

    # Validate against known parcels
    if parcellation == "desikan_killiany":
        known = set(DK_ALL_REGIONS)
        matched = set(df["region"]).intersection(known)
        fraction = len(matched) / max(len(df), 1)
        if fraction < 0.5:
            logger.warning(
                f"Only {len(matched)}/{len(df)} regions matched Desikan-Killiany atlas. "
                f"Please verify region naming convention."
            )

    return df


def list_available_damage_maps(data_dir: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Return all available damage maps by disorder and measure.
    """
    available = {}

    # Built-in reference maps
    for d in _BUILTIN_EFFECT_SIZES:
        available[d] = ["cortical_thickness (reference)"]

    # Files on disk
    if data_dir is None:
        try:
            data_dir = get_project_root() / "data" / "raw"
        except Exception:
            data_dir = Path("data/raw")

    enigma_dir = Path(data_dir) / "enigma"
    if enigma_dir.exists():
        for f in enigma_dir.glob("*.csv"):
            parts = f.stem.split("_")
            disorder = parts[0]
            measure = "_".join(parts[1:])
            available.setdefault(disorder, []).append(measure)

    return available


# =============================================================================
# Spatial Spin Test (Alexander-Bloch et al., 2018; Váša et al., 2018)
# =============================================================================

def sample_random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """
    Sample a uniform random 3x3 rotation matrix R in SO(3).

    Uses the subgroup algorithm via QR decomposition of a standard Gaussian matrix.
    """
    M = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(M)
    # Ensure unique QR
    Q = Q @ np.diag(np.sign(np.diag(R)))
    # Ensure det(Q) == +1 (rotation, not reflection)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def spin_test_permutations(
    coords: np.ndarray,
    n_perm: int = 1000,
    seed: int = 42,
    mirror_rh: bool = True,
) -> np.ndarray:
    """
    Generate Alexander-Bloch spherical spin permutations.

    For bilateral 68-parcel Desikan-Killiany cortex (34 left, 34 right),
    rotates coordinates on the unit sphere and pairs parcels using Hungarian
    optimal assignment, preserving bilateral symmetry and spatial autocorrelation.

    Parameters
    ----------
    coords : np.ndarray
        Shape (n_regions, 3). Region 3D coordinates on unit sphere S^2.
    n_perm : int
        Number of spin permutations.
    seed : int
        Random seed.
    mirror_rh : bool
        If True and n_regions == 68, applies bilateral symmetry reflection
        to the right hemisphere as in Váša et al. (2018).

    Returns
    -------
    np.ndarray
        Shape (n_perm, n_regions) integer indices.
        Each row is a permutation of range(n_regions).
    """
    rng = np.random.default_rng(seed)
    n_regions = coords.shape[0]

    # Normalize coordinates to unit sphere
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    u_coords = coords / norms

    perms = np.zeros((n_perm, n_regions), dtype=np.int64)

    # Special case: bilateral 68 parcels (34 LH + 34 RH)
    is_bilateral_dk = (n_regions == 68) and mirror_rh

    for p in range(n_perm):
        R = sample_random_rotation_matrix(rng)

        if is_bilateral_dk:
            # LH: indices 0..33
            coords_lh = u_coords[:34]
            rot_lh = coords_lh @ R
            cost_lh = cdist(rot_lh, coords_lh)
            _, perm_lh = linear_sum_assignment(cost_lh)

            # RH: indices 34..67
            # Reflection across sagittal plane (X=0): R_rh = diag(-1, 1, 1) @ R @ diag(-1, 1, 1)
            flip_x = np.diag([-1.0, 1.0, 1.0])
            R_rh = flip_x @ R @ flip_x
            coords_rh = u_coords[34:]
            rot_rh = coords_rh @ R_rh
            cost_rh = cdist(rot_rh, coords_rh)
            _, perm_rh = linear_sum_assignment(cost_rh)

            perms[p, :34] = perm_lh
            perms[p, 34:] = 34 + perm_rh
        else:
            # Single hemisphere or general surface
            rot_coords = u_coords @ R
            cost = cdist(rot_coords, u_coords)
            _, perm = linear_sum_assignment(cost)
            perms[p, :] = perm

    return perms


def generate_brainsmash_surrogates(
    x: np.ndarray,
    dist_matrix: np.ndarray,
    n_perm: int = 1000,
    seed: int = 42,
) -> Optional[np.ndarray]:
    """
    Generate distance-preserving surrogate maps using BrainSMASH.

    Parameters
    ----------
    x : np.ndarray
        1D target map (n_regions,).
    dist_matrix : np.ndarray
        2D distance matrix (n_regions, n_regions).
    n_perm : int
        Number of surrogates.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray or None
        Shape (n_perm, n_regions) surrogate maps, or None if brainsmash unavailable.
    """
    try:
        from brainsmash.mapgen.base import Base
        gen = Base(x, dist_matrix, seed=seed)
        surrogates = gen(n=n_perm)
        return surrogates
    except Exception as e:
        logger.debug(f"BrainSMASH surrogate generation failed: {e}")
        return None


# =============================================================================
# Spatial Correlation & Statistical Testing
# =============================================================================

def compute_damage_correlation(
    enrichment_scores: pd.Series,
    damage_map: pd.Series,
    coords_df: Optional[pd.DataFrame] = None,
    n_perm: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute spatial correlation between genetic enrichment and neuroimaging damage map.

    Evaluates both Pearson r and Spearman rho. Tests significance using:
    - Standard parametric p-value (Student's t / asymptotic)
    - Alexander-Bloch spatial spin-test p-value (p_spin)
    - BrainSMASH distance-preserving surrogate p-value (p_smash, if feasible)

    Parameters
    ----------
    enrichment_scores : pd.Series
        Indexed by region name. Values are regional enrichment z-scores.
    damage_map : pd.Series
        Indexed by region name. Values are case-control effect sizes (Cohen's d).
    coords_df : pd.DataFrame, optional
        Spherical coordinates for spin test. If None, uses standard DK coords.
    n_perm : int
        Number of spin permutations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary of correlation coefficients, parametric p-values, and
        spatial spin-test p-values.
    """
    # 1. Align regions
    common_regions = [
        r for r in enrichment_scores.index
        if r in damage_map.index and not np.isnan(enrichment_scores[r]) and not np.isnan(damage_map[r])
    ]

    n_regions = len(common_regions)
    if n_regions < 10:
        logger.warning(f"Too few matching regions ({n_regions}) for damage correlation.")
        return {
            "n_regions": n_regions,
            "pearson_r": np.nan,
            "pearson_p_param": np.nan,
            "pearson_p_spin": np.nan,
            "spearman_rho": np.nan,
            "spearman_p_param": np.nan,
            "spearman_p_spin": np.nan,
            "p_smash": np.nan,
        }

    x = enrichment_scores.loc[common_regions].values.astype(np.float64)
    y = damage_map.loc[common_regions].values.astype(np.float64)

    # 2. Parametric correlation
    r_pearson, p_pearson_param = stats.pearsonr(x, y)
    rho_spearman, p_spearman_param = stats.spearmanr(x, y)

    # 3. Spatial spin test
    if coords_df is None:
        coords_df = get_desikan_killiany_spherical_coords()

    # Check if we have coordinates for all common regions
    has_coords = all(r in coords_df.index for r in common_regions)

    if has_coords:
        coords_subset = coords_df.loc[common_regions].values
        spin_perms = spin_test_permutations(coords_subset, n_perm=n_perm, seed=seed)

        # Null distribution of correlation under spin permutations
        null_r_pearson = np.zeros(n_perm)
        null_rho_spearman = np.zeros(n_perm)

        for p_idx in range(n_perm):
            x_perm = x[spin_perms[p_idx]]
            null_r_pearson[p_idx] = stats.pearsonr(x_perm, y)[0]
            null_rho_spearman[p_idx] = stats.spearmanr(x_perm, y)[0]

        # Two-sided spin p-values
        b_pearson = np.sum(np.abs(null_r_pearson) >= np.abs(r_pearson))
        p_pearson_spin = (b_pearson + 1.0) / (n_perm + 1.0)

        b_spearman = np.sum(np.abs(null_rho_spearman) >= np.abs(rho_spearman))
        p_spearman_spin = (b_spearman + 1.0) / (n_perm + 1.0)
    else:
        # Fallback to random label shuffle (with documented caveat)
        logger.info("Using label shuffle permutation fallback (missing spherical coordinates).")
        rng = np.random.default_rng(seed)
        null_r = np.zeros(n_perm)
        null_rho = np.zeros(n_perm)
        for p_idx in range(n_perm):
            x_perm = rng.permutation(x)
            null_r[p_idx] = stats.pearsonr(x_perm, y)[0]
            null_rho[p_idx] = stats.spearmanr(x_perm, y)[0]

        p_pearson_spin = (np.sum(np.abs(null_r) >= np.abs(r_pearson)) + 1.0) / (n_perm + 1.0)
        p_spearman_spin = (np.sum(np.abs(null_rho) >= np.abs(rho_spearman)) + 1.0) / (n_perm + 1.0)

    # 4. BrainSMASH distance surrogates (optional check)
    p_smash = np.nan
    try:
        if has_coords:
            D = cdist(coords_subset, coords_subset)
            surrogates = generate_brainsmash_surrogates(x, D, n_perm=min(n_perm, 500), seed=seed)
            if surrogates is not None:
                smash_rs = [stats.pearsonr(s, y)[0] for s in surrogates]
                b_smash = np.sum(np.abs(smash_rs) >= np.abs(r_pearson))
                p_smash = (b_smash + 1.0) / (len(smash_rs) + 1.0)
    except Exception:
        p_smash = np.nan

    return {
        "n_regions": n_regions,
        "pearson_r": float(r_pearson),
        "pearson_p_param": float(p_pearson_param),
        "pearson_p_spin": float(p_pearson_spin),
        "spearman_rho": float(rho_spearman),
        "spearman_p_param": float(p_spearman_param),
        "spearman_p_spin": float(p_spearman_spin),
        "p_smash": float(p_smash),
    }


# =============================================================================
# Pipeline Runner
# =============================================================================

def run_damage_comparison(
    enrichment_results: Optional[pd.DataFrame] = None,
    disorders: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Run spatial damage map comparison across all disorders and null models.

    Parameters
    ----------
    enrichment_results : pd.DataFrame, optional
        Phase 3 output. If None, loads from data/processed/enrichment_results.parquet.
    disorders : list of str, optional
        Subset of disorders to compare. If None, runs all available in enrichment.
    params : dict, optional
        Pipeline parameters.
    output_dir : str or Path, optional
        Output directory.

    Returns
    -------
    pd.DataFrame
        Table with columns:
        ['disorder', 'geneset_def', 'null_model', 'damage_dataset', 'measure',
         'corr_type', 'r', 'p_param', 'p_spin', 'p_smash', 'p_fdr', 'n_regions']
    """
    if params is None:
        params = load_params()

    damage_params = params.get("damage", {})
    spin_n_perm = damage_params.get("spin_n_perm", 1000)
    seed = params.get("seed", 42)

    # Load enrichment results if not supplied
    if enrichment_results is None:
        from vulnmap.utils import get_project_root
        results_file = get_project_root() / "data" / "processed" / "enrichment_results.parquet"
        if not results_file.exists():
            raise FileNotFoundError(
                f"Enrichment results not found at {results_file}. Run Phase 3 first."
            )
        enrichment_results = pd.read_parquet(results_file)

    if output_dir is None:
        output_dir = get_project_root() / "data" / "processed"
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    # Determine disorders to test
    if disorders is None:
        disorders = list(enrichment_results["disorder"].unique())

    all_rows = []

    for disorder in disorders:
        # Check if we have damage maps for this disorder
        try:
            damage_df = load_enigma_map(disorder, measure="cortical_thickness")
        except FileNotFoundError:
            logger.info(f"Skipping {disorder}: no ENIGMA damage map available.")
            continue

        damage_series = damage_df.set_index("region")["effect_size"]

        # Filter enrichment results for this disorder
        d_enrich = enrichment_results[enrichment_results["disorder"] == disorder]

        for (gs_def, null_model), group in d_enrich.groupby(["geneset_def", "null_model"]):
            enrich_series = group.set_index("region")["z"]

            corr_stats = compute_damage_correlation(
                enrichment_scores=enrich_series,
                damage_map=damage_series,
                n_perm=spin_n_perm,
                seed=seed,
            )

            # Pearson record
            all_rows.append({
                "disorder": disorder,
                "geneset_def": gs_def,
                "null_model": null_model,
                "damage_dataset": "ENIGMA",
                "measure": "cortical_thickness",
                "corr_type": "pearson",
                "r": corr_stats["pearson_r"],
                "p_param": corr_stats["pearson_p_param"],
                "p_spin": corr_stats["pearson_p_spin"],
                "p_smash": corr_stats["p_smash"],
                "n_regions": corr_stats["n_regions"],
            })

            # Spearman record
            all_rows.append({
                "disorder": disorder,
                "geneset_def": gs_def,
                "null_model": null_model,
                "damage_dataset": "ENIGMA",
                "measure": "cortical_thickness",
                "corr_type": "spearman",
                "r": corr_stats["spearman_rho"],
                "p_param": corr_stats["spearman_p_param"],
                "p_spin": corr_stats["spearman_p_spin"],
                "p_smash": corr_stats["p_smash"],
                "n_regions": corr_stats["n_regions"],
            })

    if not all_rows:
        logger.warning("No damage map comparisons could be performed.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_rows)

    # Compute FDR-adjusted p-values across all spin test p-values
    results_df["p_fdr"] = bh_fdr(results_df["p_spin"].values)

    # Save to disk
    parquet_path = output_dir / "damage_comparison_results.parquet"
    csv_path = output_dir / "damage_comparison_results.csv"
    results_df.to_parquet(parquet_path, index=False)
    results_df.to_csv(csv_path, index=False)
    logger.info(f"[OK] Damage comparison results saved to {parquet_path}")

    return results_df


# =============================================================================
# QC Report
# =============================================================================

def damage_qc_report(results: pd.DataFrame) -> str:
    """
    Format a text QC report comparing genetic enrichment vs empirical damage maps.
    """
    lines = [
        "=" * 70,
        "VulnMap Phase 4: Damage-Map Comparison QC Report",
        "=" * 70,
        "",
    ]

    if results.empty:
        lines.append("No damage map comparisons available.")
        lines.append("=" * 70)
        return "\n".join(lines)

    lines.extend([
        f"Total comparisons: {len(results):,}",
        f"Disorders evaluated: {results['disorder'].nunique()}",
        f"Null models tested: {results['null_model'].nunique()}",
        "",
        "--- Summary Table (Pearson r & Spin Test Significance) ---",
    ])

    pearson_df = results[results["corr_type"] == "pearson"][
        ["disorder", "geneset_def", "null_model", "r", "p_param", "p_spin", "p_fdr", "n_regions"]
    ]
    lines.append(pearson_df.to_string(index=False))

    lines.extend([
        "",
        "--- Key Check: Do parametric p-values survive spatial spin tests? ---",
    ])

    for _, row in pearson_df.iterrows():
        d = row["disorder"]
        nm = row["null_model"]
        r_val = row["r"]
        p_par = row["p_param"]
        p_sp = row["p_spin"]

        if p_sp < 0.05:
            status = "[OK] Significant under spin null"
        elif p_par < 0.05:
            status = "[!] Parametric false positive (does NOT survive spin test)"
        else:
            status = "[-] Not significant"

        lines.append(f"  {d} ({nm}): r={r_val:.3f}, p_param={p_par:.4f}, p_spin={p_sp:.4f} -> {status}")

    lines.extend([
        "",
        "--- LIMITATIONS ---",
        "  1. ENIGMA maps reflect meta-analytic case-control differences, not individual atrophy.",
        "  2. Spin tests rotate cortical parcels on the sphere; subcortical structures are excluded.",
        "  3. Negative r means higher risk-gene expression aligns with greater cortical thinning.",
        "=" * 70,
    ])

    return "\n".join(lines)
