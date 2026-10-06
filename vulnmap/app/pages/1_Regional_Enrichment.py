"""
Page 1: Regional Brain Enrichment & Null Model Shrinkage (Hypothesis 1)
======================================================================
"""

import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from vulnmap.app_service import load_app_datasets, get_available_disorders, get_available_null_models
from vulnmap.brain3d import create_3d_brain_plot

st.set_page_config(page_title="Regional Enrichment | VulnMap", page_icon="📊", layout="wide")

st.title("📊 Regional Brain Enrichment & Null Model Shrinkage")
st.markdown(
    "Test **Hypothesis 1**: *Does regional brain vulnerability follow risk-gene expression, "
    "and does the observed significance survive spatial autocorrelation control?*"
)

data = load_app_datasets()
enrich_df = data["enrichment"]

if enrich_df.empty:
    st.error("Enrichment data not found. Please run `make enrichment` first.")
    st.stop()

# Sidebar Filters
st.sidebar.header("Enrichment Parameters")
disorders = get_available_disorders(enrich_df)
selected_disorder = st.sidebar.selectbox("Select Disorder", disorders, index=0)

disorder_sub = enrich_df[enrich_df["disorder"] == selected_disorder]

available_defs = sorted(list(disorder_sub["geneset_def"].unique()))
selected_def = st.sidebar.selectbox("GWAS Significance Threshold", available_defs, index=0)

available_nulls = get_available_null_models(disorder_sub)
selected_null = st.sidebar.selectbox("Null Model Audit", available_nulls, index=0)

fdr_alpha = st.sidebar.slider("FDR Significance Threshold (α)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

# Filter data
filtered_df = disorder_sub[
    (disorder_sub["geneset_def"] == selected_def) &
    (disorder_sub["null_model"] == selected_null)
].copy()

if filtered_df.empty:
    st.warning("No data found for the selected parameter combination.")
    st.stop()

# Recompute significance based on dynamic slider
p_col = "p_fdr" if "p_fdr" in filtered_df.columns else "p_emp"
filtered_df["is_sig"] = filtered_df[p_col] < fdr_alpha

# Clean region names for presentation
filtered_df["region_clean"] = (
    filtered_df["region"]
    .str.replace("lh_", "L-")
    .str.replace("rh_", "R-")
    .str.replace("_", " ")
    .str.title()
)

# Metric Summary Cards
m1, m2, m3, m4 = st.columns(4)
sig_count = int(filtered_df["is_sig"].sum())
total_count = len(filtered_df)
sig_pct = (sig_count / total_count * 100.0) if total_count > 0 else 0.0

top_vuln = filtered_df.sort_values("z", ascending=False).iloc[0]
top_resil = filtered_df.sort_values("z", ascending=True).iloc[0]

with m1:
    st.metric("Total Brain Parcels", f"{total_count}")
with m2:
    st.metric(f"Significant Regions (FDR < {fdr_alpha:.2f})", f"{sig_count} ({sig_pct:.1f}%)")
with m3:
    st.metric("Top Vulnerable Region", f"{top_vuln['region_clean']}", f"z = +{top_vuln['z']:.2f}")
with m4:
    st.metric("Top Resilient Region", f"{top_resil['region_clean']}", f"z = {top_resil['z']:.2f}")

st.write("")

# Main Visualizations Tabs
tab_3d, tab_profile, tab_shrinkage, tab_table = st.tabs([
    "🧠 Interactive 3D Brain Topography",
    "📊 Regional Vulnerability Profile",
    "📉 Null Model Shrinkage Effect",
    "📑 Data Table & Export",
])

with tab_3d:
    st.subheader(f"3D Anatomical Brain Topography: {selected_disorder.replace('_', ' ').title()}")
    st.caption("Rotate, pan, and zoom in 3D space. Red indicates high genetic vulnerability; blue indicates relative resilience.")

    # 3D Display Controls
    c_b1, c_b2, c_b3 = st.columns(3)
    with c_b1:
        render_mode = st.selectbox(
            "Brain Representation",
            ["realistic", "glass_centroids"],
            format_func=lambda x: "🧠 Realistic Cortical Surface (Mesh3D)" if x == "realistic" else "🌐 Glass Brain Parcels (Centroids)",
            key="p1_render_mode",
        )
    with c_b2:
        surf_type = st.selectbox(
            "Cortical Surface Folding",
            ["inflated", "pial"],
            format_func=lambda x: "Inflated (Expanded Sulci - Best for Parcels)" if x == "inflated" else "Pial (Anatomical Gyral Folds)",
            key="p1_surf_type",
            disabled=(render_mode != "realistic"),
        )
    with c_b3:
        hemi = st.selectbox(
            "Hemisphere View",
            ["both", "left", "right"],
            format_func=lambda x: "Both Hemispheres" if x == "both" else ("Left Hemisphere Only" if x == "left" else "Right Hemisphere Only"),
            key="p1_hemi",
            disabled=(render_mode != "realistic"),
        )

    score_series = filtered_df.set_index("region")["z"]
    fig_3d = create_3d_brain_plot(
        score_series,
        title=f"{selected_disorder.replace('_', ' ').title()} Vulnerability Topography",
        val_name="Enrichment z-score",
        colorscale="RdBu_r",
        midpoint=0.0,
        render_mode=render_mode,
        surface_type=surf_type,
        hemisphere=hemi,
        height=620,
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with tab_profile:
    st.subheader(f"Regional Enrichment z-score Profile: {selected_disorder.replace('_', ' ').title()}")
    st.caption(f"Gene Set: {selected_def} | Null Model: {selected_null}")

    # Top & Bottom Extremes
    sorted_df = filtered_df.sort_values("z", ascending=False).reset_index(drop=True)
    n_display = st.slider("Number of Extreme Regions to Display", min_value=10, max_value=len(sorted_df), value=min(30, len(sorted_df)), step=2)

    half = n_display // 2
    extremes_df = pd.concat([sorted_df.head(half), sorted_df.tail(half)]).drop_duplicates().sort_values("z", ascending=True)

    # Plotly horizontal bar chart
    fig = px.bar(
        extremes_df,
        x="z",
        y="region_clean",
        orientation="h",
        color="z",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        text=extremes_df["z"].apply(lambda v: f"{v:+.2f}"),
        labels={"z": "Spatial Enrichment z-score", "region_clean": "Brain Region"},
        height=max(450, len(extremes_df) * 22),
    )
    fig.add_vline(x=0, line_width=1, line_dash="solid", line_color="#333333")
    fig.add_vline(x=1.96, line_width=1, line_dash="dash", line_color="#c0392b", annotation_text="z = +1.96 (p < 0.05)")
    fig.add_vline(x=-1.96, line_width=1, line_dash="dash", line_color="#2980b9", annotation_text="z = -1.96")

    fig.update_layout(
        margin=dict(l=10, r=20, t=30, b=40),
        xaxis_title="Enrichment z-score",
        yaxis_title="",
        coloraxis_colorbar=dict(title="z-score"),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_shrinkage:
    st.subheader("The Null Model Shrinkage Phenomenon")
    st.markdown(
        """
        **Methodological Insight**: Naive null models (randomly shuffling gene labels across the genome)
        assume genes are spatially independent. In real brain tissue, co-expression and spatial autocorrelation
        induce spurious regional correlations. Notice how the number of "significant" regions shrinks dramatically
        when comparing **Naive** to **Matched** and **Spatial Co-expression** null models!
        """
    )

    # Compute shrinkage across null models for current disorder
    shrinkage_df = (
        disorder_sub[disorder_sub["geneset_def"] == selected_def]
        .groupby("null_model")[p_col]
        .apply(lambda s: (s < fdr_alpha).sum())
        .reset_index(name="n_significant")
    )

    fig_shrink = px.bar(
        shrinkage_df,
        x="null_model",
        y="n_significant",
        color="null_model",
        color_discrete_sequence=["#e74c3c", "#f39c12", "#2980b9", "#27ae60"],
        text="n_significant",
        labels={"null_model": "Null Model Specification", "n_significant": "Significant Regions (FDR < 0.05)"},
        title=f"Significant Parcel Shrinkage: {selected_disorder.replace('_', ' ').title()}",
    )
    fig_shrink.update_traces(textposition="outside")
    fig_shrink.update_layout(showlegend=False, yaxis_title="Significant Region Count", height=400)
    st.plotly_chart(fig_shrink, use_container_width=True)

with tab_table:
    st.subheader("Regional Vulnerability Statistics")
    display_cols = ["region_clean", "z", "score", "null_mean", "null_sd", p_col, "is_sig", "n_genes"]
    table_view = filtered_df[display_cols].sort_values("z", ascending=False).rename(
        columns={
            "region_clean": "Region",
            "z": "Enrichment z-score",
            "score": "Mean Raw Score",
            "null_mean": "Null Mean",
            "null_sd": "Null SD",
            p_col: "FDR Adjusted p-value",
            "is_sig": "Significant (FDR < α)",
            "n_genes": "Gene Count",
        }
    )
    st.dataframe(table_view, use_container_width=True, height=450)

    csv_data = table_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Regional Enrichment CSV",
        data=csv_data,
        file_name=f"vulnmap_{selected_disorder}_{selected_def}_{selected_null}.csv",
        mime="text/csv",
    )
