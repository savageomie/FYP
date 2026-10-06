"""
Page 4: Custom Risk-Gene Set Live Analysis
=========================================
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

from vulnmap.app_service import run_user_gene_enrichment, harmonize_user_genes, get_expression_matrix
from vulnmap.brain3d import create_3d_brain_plot

st.set_page_config(page_title="Custom Gene Analysis | VulnMap", page_icon="⚡", layout="wide")

st.title("⚡ Live Custom Gene-Set Vulnerability Analysis")
st.markdown(
    "Input your candidate risk genes to map their spatial brain vulnerability topography on-the-fly "
    "using AHBA microarray expression and permutation null models."
)

# Preset Gene Sets
PRESETS = {
    "Synaptic & Neurotransmission": "DRD2, COMT, GRIN2A, CACNA1C, DISC1, SNAP25, SYP, SYN1, BDNF, HTR2A",
    "Neuroinflammatory & Glial": "AIF1, CD68, GFAP, APOE, TREM2, CX3CR1, ITGAM, PECAM1, VWF",
    "Neurodegenerative Proteopathy": "APP, MAPT, SNCA, LRRK2, TARDBP, FUS, SOD1, PARK7, PINK1",
}

# Sidebar configuration
st.sidebar.header("Analysis Settings")
null_choice = st.sidebar.selectbox("Null Model", ["naive", "matched"], index=0, help="Naive = uniform permutation; Matched = stratified by gene length & GC")
n_perm = st.sidebar.slider("Permutation Draws", min_value=100, max_value=1000, value=500, step=100, help="Reduced permutations ensure sub-second response")
fdr_alpha = st.sidebar.slider("FDR Significance Threshold (α)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
seed = st.sidebar.number_input("Reproducibility Seed", value=42, step=1)

# Preset Buttons Row
st.subheader("1. Select a Preset or Enter Custom Gene Symbols")
p_cols = st.columns(3)
selected_preset = None

if p_cols[0].button("🧠 Synaptic & Neurotransmission", use_container_width=True):
    selected_preset = PRESETS["Synaptic & Neurotransmission"]
if p_cols[1].button("🛡️ Neuroinflammatory & Glial", use_container_width=True):
    selected_preset = PRESETS["Neuroinflammatory & Glial"]
if p_cols[2].button("🧬 Neurodegenerative Proteopathy", use_container_width=True):
    selected_preset = PRESETS["Neurodegenerative Proteopathy"]

default_text = selected_preset if selected_preset else "DRD2, COMT, GRIN2A, CACNA1C, DISC1, BDNF, SNAP25, SYP"

gene_input = st.text_area(
    "Enter gene symbols (comma- or newline-separated):",
    value=default_text,
    height=110,
    help="Enter standard HGNC gene symbols.",
)

# Parse genes
raw_tokens = [t.strip().upper() for t in gene_input.replace("\n", ",").split(",") if t.strip()]

# Harmonization Preview
expr_df = get_expression_matrix()
harm_info = harmonize_user_genes(raw_tokens, list(expr_df.columns))

h_col1, h_col2, h_col3 = st.columns(3)
with h_col1:
    st.metric("Input Genes", f"{harm_info['n_input']}")
with h_col2:
    st.metric("Harmonized in AHBA", f"{harm_info['n_valid']}")
with h_col3:
    st.metric("Mapping Rate", f"{harm_info['mapping_rate']:.1f}%")

if harm_info["unrecognized_genes"]:
    st.warning(f"Unrecognized / filtered genes: {', '.join(harm_info['unrecognized_genes'])}")

# Run Button
if st.button("🚀 Run Brain Vulnerability Analysis", type="primary", use_container_width=True):
    if harm_info["n_valid"] < 3:
        st.error("Please provide at least 3 valid gene symbols found in the AHBA atlas.")
    else:
        with st.spinner(f"Computing regional scores across 82 Desikan-Killiany parcels ({n_perm} permutations)..."):
            try:
                res = run_user_gene_enrichment(
                    gene_symbols=harm_info["valid_genes"],
                    null_model_name=null_choice,
                    n_perm=n_perm,
                    seed=int(seed),
                    fdr_alpha=fdr_alpha,
                    expression_df=expr_df,
                )
                st.session_state["custom_res"] = res
                st.success("Analysis complete!")
            except Exception as e:
                st.error(f"Error computing enrichment: {e}")

# Results Display
if "custom_res" in st.session_state:
    res = st.session_state["custom_res"]
    res_df = res["results_df"].copy()
    res_df["region_clean"] = (
        res_df["region"]
        .astype(str)
        .str.replace("lh_", "L-")
        .str.replace("rh_", "R-")
        .str.replace("_", " ")
        .str.title()
    )

    st.markdown("---")
    st.subheader("2. Regional Brain Vulnerability Results")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Significant Parcels", f"{res['n_significant_fdr']} / 82")
    with c2:
        st.metric("Max Vulnerability z", f"+{res['max_z']:.2f}")
    with c3:
        st.metric("Mean Regional z", f"{res['mean_z']:+.2f}")
    with c4:
        st.metric("Min Vulnerability z", f"{res['min_z']:.2f}")

    # Matcher banner
    st.info(
        "💡 **Step 4 Complete: Brain Topography Mapped!** "
        "Want to see which of the **15 clinical disorders** (e.g. Alzheimer's, Parkinson's, Schizophrenia) "
        "your gene set matches? Navigate to **Page 3: Cross-Disorder & Cell Types → 🎯 15-Disease Signature Matcher** "
        "and select *'Import Custom Gene-Set Result'*!"
    )

    tab_3d, tab_bars, tab_table = st.tabs([
        "🧠 3D Brain Topography",
        "📊 Regional Extremes Chart",
        "📋 Detailed Table & Export",
    ])

    with tab_3d:
        st.subheader("3D Anatomical Brain Visualization")
        st.markdown(
            "Explore the spatial topography of your candidate gene set in full 3D. "
            "**Red cortex** indicates heightened vulnerability ($z > 0$), while **Blue cortex** indicates relative resilience ($z < 0$). "
            "Drag to rotate, scroll to zoom, and hover over any region to inspect exact statistics."
        )

        c_g1, c_g2, c_g3 = st.columns(3)
        with c_g1:
            cg_render = st.selectbox(
                "Brain Representation",
                ["realistic", "glass_centroids"],
                format_func=lambda x: "🧠 Realistic Cortical Surface (Mesh3D)" if x == "realistic" else "🌐 Glass Brain Parcels (Centroids)",
                key="p4_render_mode",
            )
        with c_g2:
            cg_surf = st.selectbox(
                "Cortical Surface Folding",
                ["inflated", "pial"],
                format_func=lambda x: "Inflated (Expanded Sulci - Best for Parcels)" if x == "inflated" else "Pial (Anatomical Gyral Folds)",
                key="p4_surf_type",
                disabled=(cg_render != "realistic"),
            )
        with c_g3:
            cg_hemi = st.selectbox(
                "Hemisphere View",
                ["both", "left", "right"],
                format_func=lambda x: "Both Hemispheres" if x == "both" else ("Left Hemisphere Only" if x == "left" else "Right Hemisphere Only"),
                key="p4_hemi",
                disabled=(cg_render != "realistic"),
            )

        z_series = res_df.set_index("region")["z"]
        fig_3d = create_3d_brain_plot(
            z_series,
            title=f"Custom Gene Set Brain Vulnerability (N={len(harm_info['valid_genes'])} Genes)",
            val_name="Vulnerability z-score",
            render_mode=cg_render,
            surface_type=cg_surf,
            hemisphere=cg_hemi,
            height=620,
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab_bars:
        # Top & Bottom Extremes Chart
        sorted_res = res_df.sort_values("z", ascending=False).reset_index(drop=True)
        extremes = pd.concat([sorted_res.head(10), sorted_res.tail(10)]).drop_duplicates().sort_values("z", ascending=True)

        fig_res = px.bar(
            extremes,
            x="z",
            y="region_clean",
            orientation="h",
            color="z",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            text=extremes["z"].apply(lambda v: f"{v:+.2f}"),
            labels={"z": "Vulnerability z-score", "region_clean": "Brain Parcel"},
            title="Top Vulnerable vs Resilient Brain Regions for Custom Gene Set",
            height=520,
        )
        fig_res.add_vline(x=0, line_width=1, line_color="#333333")
        fig_res.add_vline(x=1.96, line_dash="dash", line_color="#c0392b", annotation_text="z = +1.96 (p < 0.05)")
        fig_res.add_vline(x=-1.96, line_dash="dash", line_color="#2980b9", annotation_text="z = -1.96")

        st.plotly_chart(fig_res, use_container_width=True)

    with tab_table:
        # Detailed Table & Export
        st.subheader("Regional Vulnerability Scores Table")
        table_cols = ["region_clean", "z", "score", "null_mean", "null_sd", "p_fdr", "significant_fdr"]
        out_table = res_df[table_cols].sort_values("z", ascending=False).rename(
            columns={
                "region_clean": "Region",
                "z": "Vulnerability z-score",
                "score": "Observed Mean Score",
                "null_mean": "Null Mean",
                "null_sd": "Null SD",
                "p_fdr": "FDR p-value",
                "significant_fdr": "Significant (FDR < α)",
            }
        )
        st.dataframe(out_table, use_container_width=True, height=350)

        csv_out = out_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Custom Vulnerability Scores CSV",
            data=csv_out,
            file_name="custom_gene_vulnerability_scores.csv",
            mime="text/csv",
        )

