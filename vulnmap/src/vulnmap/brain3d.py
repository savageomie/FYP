"""
VulnMap: High-Fidelity 3D Anatomical Brain Topography Visualizer
================================================================

Provides:
1. High-definition, Gouraud-shaded 3D realistic cortical surface reconstruction
   using FreeSurfer fsaverage5 surface meshes (inflated and anatomical pial).
2. Accurate vertex-level mapping of all 68 Desikan-Killiany cortical parcels
   and 14 stereotaxic subcortical nuclei in MNI space.
3. Dual-brain side-by-side comparison (transcriptomic vulnerability vs MRI gray matter loss).
4. Legacy centroid-based glass brain representation as an alternative view.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import nibabel.freesurfer as fs
import nilearn.datasets as n_ds
import nilearn.surface as n_surf

from vulnmap.damage import (
    DK_CORTICAL_REGIONS_LH,
    DK_CORTICAL_REGIONS_RH,
    _RAW_LH_SPHERICAL_COORDS,
)
from vulnmap.utils import get_project_root, logger

# Standard 3D Centroid Coordinates for 14 Subcortical Nuclei (MNI coordinates in mm)
MNI_SUBCORTICAL_COORDS = {
    "Left-Thalamus-Proper": (-12.0, -18.0, 8.0),
    "Right-Thalamus-Proper": (12.0, -18.0, 8.0),
    "Left-Caudate": (-13.0, 10.0, 10.0),
    "Right-Caudate": (13.0, 10.0, 10.0),
    "Left-Putamen": (-25.0, 1.0, 1.0),
    "Right-Putamen": (25.0, 1.0, 1.0),
    "Left-Pallidum": (-19.0, -4.0, -1.0),
    "Right-Pallidum": (19.0, -4.0, -1.0),
    "Left-Hippocampus": (-25.0, -20.0, -15.0),
    "Right-Hippocampus": (25.0, -20.0, -15.0),
    "Left-Amygdala": (-23.0, -6.0, -18.0),
    "Right-Amygdala": (23.0, -6.0, -18.0),
    "Left-Accumbens-area": (-9.0, 12.0, -7.0),
    "Right-Accumbens-area": (9.0, 12.0, -7.0),
}

# Lobe anatomical classification
LOBE_MAP = {
    "caudalanteriorcingulate": "Cingulate",
    "isthmuscingulate": "Cingulate",
    "posteriorcingulate": "Cingulate",
    "rostralanteriorcingulate": "Cingulate",
    "caudalmiddlefrontal": "Frontal",
    "lateralorbitofrontal": "Frontal",
    "medialorbitofrontal": "Frontal",
    "parsopercularis": "Frontal",
    "parsorbitalis": "Frontal",
    "parstriangularis": "Frontal",
    "precentral": "Frontal",
    "rostralmiddlefrontal": "Frontal",
    "superiorfrontal": "Frontal",
    "frontalpole": "Frontal",
    "paracentral": "Frontal",
    "cuneus": "Occipital",
    "lateraloccipital": "Occipital",
    "lingual": "Occipital",
    "pericalcarine": "Occipital",
    "inferiorparietal": "Parietal",
    "postcentral": "Parietal",
    "precuneus": "Parietal",
    "superiorparietal": "Parietal",
    "supramarginal": "Parietal",
    "bankssts": "Temporal",
    "entorhinal": "Temporal",
    "fusiform": "Temporal",
    "inferiortemporal": "Temporal",
    "middletemporal": "Temporal",
    "parahippocampal": "Temporal",
    "superiortemporal": "Temporal",
    "temporalpole": "Temporal",
    "transversetemporal": "Temporal",
    "insula": "Insula",
}


def _ensure_aparc_annotations() -> Tuple[Path, Path]:
    """Ensure Desikan-Killiany surface annotation files for fsaverage5 exist locally."""
    atlas_dir = get_project_root() / "data" / "raw" / "atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)

    lh_path = atlas_dir / "lh.aparc.annot"
    rh_path = atlas_dir / "rh.aparc.annot"

    base_url = (
        "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/"
        "brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label/"
    )

    if not lh_path.exists():
        urllib.request.urlretrieve(base_url + "lh.aparc.annot", str(lh_path))
    if not rh_path.exists():
        urllib.request.urlretrieve(base_url + "rh.aparc.annot", str(rh_path))

    return lh_path, rh_path


@lru_cache(maxsize=4)
def load_fsaverage5_surface(surface_type: str = "inflated") -> Dict[str, Any]:
    """
    Fetch and load fsaverage5 surface mesh (vertices and triangular faces).

    Parameters
    ----------
    surface_type : str
        'inflated' (opens sulci for parcel visibility) or 'pial' (anatomical folded brain).
    """
    fsaverage = n_ds.fetch_surf_fsaverage("fsaverage5")

    if surface_type == "pial":
        coords_l, faces_l = n_surf.load_surf_mesh(fsaverage["pial_left"])
        coords_r, faces_r = n_surf.load_surf_mesh(fsaverage["pial_right"])
    else:
        # Default: inflated surface
        coords_l, faces_l = n_surf.load_surf_mesh(fsaverage["infl_left"])
        coords_r, faces_r = n_surf.load_surf_mesh(fsaverage["infl_right"])
        # Offset hemispheres slightly so they sit side-by-side realistically
        coords_l = coords_l.copy()
        coords_r = coords_r.copy()
        coords_l[:, 0] -= 24.0
        coords_r[:, 0] += 24.0

    return {
        "coords_l": coords_l,
        "faces_l": faces_l,
        "coords_r": coords_r,
        "faces_r": faces_r,
    }


@lru_cache(maxsize=1)
def load_desikan_annotations() -> Dict[str, Any]:
    """Load Desikan-Killiany parcel annotations for LH and RH vertices."""
    lh_path, rh_path = _ensure_aparc_annotations()

    labels_l, _, names_l = fs.read_annot(str(lh_path))
    labels_r, _, names_r = fs.read_annot(str(rh_path))

    return {
        "labels_l": labels_l,
        "names_l": [n.decode("utf-8") for n in names_l],
        "labels_r": labels_r,
        "names_r": [n.decode("utf-8") for n in names_r],
    }


def get_all_82_coordinates() -> pd.DataFrame:
    """Return 3D coordinates for all 82 Desikan-Killiany parcels (glass-brain mode)."""
    coords = {}
    for name, (x, y, z) in _RAW_LH_SPHERICAL_COORDS.items():
        v = np.array([x, y, z], dtype=np.float64)
        coords[name] = v / np.linalg.norm(v)

    for name in DK_CORTICAL_REGIONS_LH:
        rh_name = "rh_" + name[3:]
        v = coords[name].copy()
        v[0] = -v[0]
        coords[rh_name] = v

    for name, (x, y, z) in MNI_SUBCORTICAL_COORDS.items():
        # Scale to normalized unit range for glass brain
        coords[name] = np.array([x / 60.0, y / 60.0, z / 60.0], dtype=np.float64)

    return pd.DataFrame.from_dict(coords, orient="index", columns=["x", "y", "z"])


def create_realistic_3d_brain(
    data_series: Union[pd.Series, Dict[str, float]],
    title: str = "Realistic 3D Anatomical Brain Topography",
    val_name: str = "Vulnerability z-score",
    surface_type: str = "inflated",
    hemisphere: str = "both",
    colorscale: str = "RdBu_r",
    midpoint: float = 0.0,
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
    show_subcortex: bool = True,
    height: int = 580,
) -> go.Figure:
    """
    Render a high-definition, Gouraud-shaded 3D realistic cortical surface mesh
    with parcel-level vulnerability mapping and interactive WebGL controls.

    Parameters
    ----------
    data_series : pd.Series or dict
        Mapping from region names (e.g. 'lh_bankssts', 'Left-Thalamus-Proper') to values.
    title : str
        Figure title.
    val_name : str
        Metric name for hover tooltip and colorbar.
    surface_type : str
        'inflated' (recommended) or 'pial'.
    hemisphere : str
        'both', 'left', or 'right'.
    colorscale : str
        Plotly colorscale name (e.g. 'RdBu_r', 'YlOrRd_r', 'Viridis').
    midpoint : float
        Midpoint of color scale.
    cmin, cmax : float, optional
        Color limits. If None, derived from data extremes.
    show_subcortex : bool
        Whether to render 3D subcortical nuclei spheres inside the brain cavity.
    height : int
        Figure height in pixels.
    """
    if isinstance(data_series, pd.Series):
        score_dict = data_series.to_dict()
    else:
        score_dict = dict(data_series)

    # Determine color limits
    vals_all = [float(v) for v in score_dict.values() if not np.isnan(v)]
    if not vals_all:
        vals_all = [0.0]
    if cmin is None:
        cmin = -max(2.5, float(np.percentile(np.abs(vals_all), 95)))
    if cmax is None:
        cmax = max(2.5, float(np.percentile(np.abs(vals_all), 95)))

    # Load surface mesh & parcellation
    surf = load_fsaverage5_surface(surface_type=surface_type)
    annot = load_desikan_annotations()

    fig = go.Figure()

    # Build Left Hemisphere Mesh
    if hemisphere in ("both", "left"):
        coords_l = surf["coords_l"]
        faces_l = surf["faces_l"]
        labels_l = annot["labels_l"]
        names_l = annot["names_l"]

        v_intensity_l = np.zeros(len(coords_l), dtype=np.float64)
        v_hover_l = []

        for i, l_idx in enumerate(labels_l):
            pname = names_l[l_idx]
            if pname in ("unknown", "corpuscallosum"):
                v_intensity_l[i] = 0.0
                v_hover_l.append(f"<b>Medial Wall</b> ({pname})")
            else:
                full_name = f"lh_{pname}"
                val = float(score_dict.get(full_name, 0.0))
                v_intensity_l[i] = val
                clean_name = pname.replace("_", " ").title()
                lobe = LOBE_MAP.get(pname, "Cortical")
                v_hover_l.append(
                    f"<b>{clean_name}</b> (Left)<br>"
                    f"Lobe: {lobe}<br>"
                    f"{val_name}: {val:+.2f}"
                )

        fig.add_trace(
            go.Mesh3d(
                x=coords_l[:, 0],
                y=coords_l[:, 1],
                z=coords_l[:, 2],
                i=faces_l[:, 0],
                j=faces_l[:, 1],
                k=faces_l[:, 2],
                intensity=v_intensity_l,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                text=v_hover_l,
                hoverinfo="text",
                name="Left Hemisphere",
                lighting=dict(
                    ambient=0.45,
                    diffuse=0.75,
                    roughness=0.55,
                    specular=0.20,
                    fresnel=0.1,
                ),
                flatshading=False,
                showscale=True,
                colorbar=dict(
                    title=dict(text=val_name, side="right"),
                    thickness=14,
                    len=0.7,
                    x=1.02,
                ),
            )
        )

    # Build Right Hemisphere Mesh
    if hemisphere in ("both", "right"):
        coords_r = surf["coords_r"]
        faces_r = surf["faces_r"]
        labels_r = annot["labels_r"]
        names_r = annot["names_r"]

        v_intensity_r = np.zeros(len(coords_r), dtype=np.float64)
        v_hover_r = []

        for i, l_idx in enumerate(labels_r):
            pname = names_r[l_idx]
            if pname in ("unknown", "corpuscallosum"):
                v_intensity_r[i] = 0.0
                v_hover_r.append(f"<b>Medial Wall</b> ({pname})")
            else:
                full_name = f"rh_{pname}"
                val = float(score_dict.get(full_name, 0.0))
                v_intensity_r[i] = val
                clean_name = pname.replace("_", " ").title()
                lobe = LOBE_MAP.get(pname, "Cortical")
                v_hover_r.append(
                    f"<b>{clean_name}</b> (Right)<br>"
                    f"Lobe: {lobe}<br>"
                    f"{val_name}: {val:+.2f}"
                )

        fig.add_trace(
            go.Mesh3d(
                x=coords_r[:, 0],
                y=coords_r[:, 1],
                z=coords_r[:, 2],
                i=faces_r[:, 0],
                j=faces_r[:, 1],
                k=faces_r[:, 2],
                intensity=v_intensity_r,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                text=v_hover_r,
                hoverinfo="text",
                name="Right Hemisphere",
                lighting=dict(
                    ambient=0.45,
                    diffuse=0.75,
                    roughness=0.55,
                    specular=0.20,
                    fresnel=0.1,
                ),
                flatshading=False,
                showscale=False,
            )
        )

    # Build Subcortical Nuclei Spheres
    if show_subcortex and hemisphere == "both":
        sub_x, sub_y, sub_z, sub_c, sub_text, sub_sizes = [], [], [], [], [], []
        for name, (x, y, z) in MNI_SUBCORTICAL_COORDS.items():
            val = float(score_dict.get(name, 0.0))
            clean_name = name.replace("-", " ").title()
            sub_x.append(x)
            sub_y.append(y)
            sub_z.append(z)
            sub_c.append(val)
            sub_sizes.append(int(np.clip(10 + 3 * abs(val), 8, 22)))
            sub_text.append(
                f"<b>{clean_name}</b><br>"
                f"Lobe: Subcortex<br>"
                f"{val_name}: {val:+.2f}<br>"
                f"MNI: ({x:.0f}, {y:.0f}, {z:.0f})"
            )

        fig.add_trace(
            go.Scatter3d(
                x=sub_x,
                y=sub_y,
                z=sub_z,
                mode="markers",
                marker=dict(
                    size=sub_sizes,
                    color=sub_c,
                    colorscale=colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    line=dict(width=1.5, color="#111827"),
                    opacity=0.98,
                ),
                text=sub_text,
                hoverinfo="text",
                name="Subcortical Nuclei",
                showlegend=False,
            )
        )

    # Scene camera & lighting
    fig.update_layout(
        title=dict(text=f"🧠 {title}", x=0.5, font=dict(size=16, color="#1e293b")),
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title="", visible=False),
            yaxis=dict(showgrid=False, showticklabels=False, title="", visible=False),
            zaxis=dict(showgrid=False, showticklabels=False, title="", visible=False),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(
                eye=dict(x=1.35, y=-1.50, z=0.90),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, b=10, t=45),
        height=height,
    )

    return fig


def _create_centroid_glass_brain(
    data_series: pd.Series,
    title: str = "Regional Brain Vulnerability Topography",
    val_name: str = "Vulnerability z-score",
    colorscale: str = "RdBu_r",
    midpoint: float = 0.0,
    height: int = 580,
) -> go.Figure:
    """Render legacy glass-brain parcel centroid map in Plotly."""
    coords_df = get_all_82_coordinates()
    common_regions = [r for r in coords_df.index if r in data_series.index]

    if not common_regions:
        common_regions = list(coords_df.index)
        vals = np.zeros(len(common_regions))
    else:
        vals = data_series.loc[common_regions].values

    plot_df = coords_df.loc[common_regions].copy()
    plot_df["val"] = vals

    hover_texts = []
    marker_colors = []
    marker_sizes = []

    for reg, row in plot_df.iterrows():
        v = row["val"]
        clean_name = (
            str(reg)
            .replace("lh_", "Left ")
            .replace("rh_", "Right ")
            .replace("-", " ")
            .replace("_", " ")
            .title()
        )
        base_name = reg[3:] if str(reg).startswith(("lh_", "rh_")) else reg
        lobe = LOBE_MAP.get(base_name, "Subcortex")

        hover_texts.append(
            f"<b>{clean_name}</b><br>"
            f"Lobe: {lobe}<br>"
            f"{val_name}: {v:+.2f}"
        )
        marker_colors.append(v)
        marker_sizes.append(int(np.clip(10 + 4 * abs(v), 8, 26)))

    fig = go.Figure()

    # Glass outline
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 16)
    gx = (1.02 * np.outer(np.cos(u), np.sin(v))).flatten()
    gy = (1.25 * np.outer(np.sin(u), np.sin(v))).flatten()
    gz = (0.95 * np.outer(np.ones(np.size(u)), np.cos(v))).flatten()

    fig.add_trace(
        go.Mesh3d(
            x=gx,
            y=gy,
            z=gz,
            alphahull=0,
            opacity=0.06,
            color="#90cdf4",
            hoverinfo="skip",
            name="Cortical Shell",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=plot_df["x"],
            y=plot_df["y"],
            z=plot_df["z"],
            mode="markers",
            marker=dict(
                size=marker_sizes,
                color=marker_colors,
                colorscale=colorscale,
                cmid=midpoint,
                colorbar=dict(
                    title=dict(text=val_name, side="right"),
                    thickness=15,
                    len=0.75,
                ),
                line=dict(width=1.5, color="#1a202c"),
                opacity=0.95,
            ),
            text=hover_texts,
            hoverinfo="text",
            name="Brain Parcels",
        )
    )

    fig.update_layout(
        title=dict(text=f"🧠 {title}", x=0.5, font=dict(size=16, color="#1a365d")),
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title="", visible=False),
            yaxis=dict(showgrid=False, showticklabels=False, title="", visible=False),
            zaxis=dict(showgrid=False, showticklabels=False, title="", visible=False),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(
                eye=dict(x=1.35, y=-1.55, z=0.95),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, b=10, t=45),
        height=height,
    )
    return fig


def create_3d_brain_plot(
    data_series: pd.Series,
    title: str = "Regional Brain Vulnerability Topography",
    val_name: str = "Vulnerability z-score",
    colorscale: str = "RdBu_r",
    midpoint: float = 0.0,
    render_mode: str = "realistic",
    surface_type: str = "inflated",
    hemisphere: str = "both",
    height: int = 580,
) -> go.Figure:
    """
    Main entrypoint: render an interactive 3D brain map in Plotly.

    Parameters
    ----------
    render_mode : str
        'realistic' (default, Gouraud-shaded cortical surface mesh) or
        'glass_centroids' (spherical parcel centroids in transparent glass brain).
    """
    if render_mode == "realistic":
        return create_realistic_3d_brain(
            data_series=data_series,
            title=title,
            val_name=val_name,
            surface_type=surface_type,
            hemisphere=hemisphere,
            colorscale=colorscale,
            midpoint=midpoint,
            height=height,
        )
    return _create_centroid_glass_brain(
        data_series=data_series,
        title=title,
        val_name=val_name,
        colorscale=colorscale,
        midpoint=midpoint,
        height=height,
    )


def create_dual_3d_comparison(
    vuln_series: pd.Series,
    damage_series: pd.Series,
    disorder_label: str = "Disorder",
    render_mode: str = "realistic",
    surface_type: str = "inflated",
) -> Tuple[go.Figure, go.Figure, float]:
    """
    Generate side-by-side 3D realistic brain figures comparing transcriptomic vulnerability
    against in vivo MRI cortical thinning, along with Pearson correlation.
    """
    common = vuln_series.index.intersection(damage_series.index)
    v_clean = vuln_series.loc[common]
    d_clean = damage_series.loc[common]

    r_val = float(np.corrcoef(v_clean.values, d_clean.values)[0, 1]) if len(common) > 2 else 0.0

    fig_vuln = create_3d_brain_plot(
        v_clean,
        title=f"{disorder_label}: Risk-Gene Vulnerability (AHBA)",
        val_name="Enrichment z-score",
        colorscale="RdBu_r",
        midpoint=0.0,
        render_mode=render_mode,
        surface_type=surface_type,
        height=480,
    )

    fig_dam = create_3d_brain_plot(
        d_clean,
        title=f"{disorder_label}: MRI Gray Matter Atrophy (ENIGMA)",
        val_name="Cohen's d (Thinning)",
        colorscale="YlOrRd_r",
        midpoint=-0.25,
        render_mode=render_mode,
        surface_type=surface_type,
        height=480,
    )

    return fig_vuln, fig_dam, r_val
