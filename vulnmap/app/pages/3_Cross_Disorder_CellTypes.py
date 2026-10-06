"""
Page 3: Cross-Disorder Structure & Cell-Type Deconvolution (Hypothesis 3)
========================================================================
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
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

from vulnmap.app_service import load_app_datasets
from vulnmap.brain3d import LOBE_MAP

st.set_page_config(page_title="Cross-Disorder & Cell Types | VulnMap", page_icon="🧬", layout="wide")

st.title("🧬 Cross-Disorder Architecture & Cell-Type Deconvolution")
st.markdown(
    "Test **Hypothesis 3**: *Do psychiatric and neurodegenerative disorders share distinct spatial "
    "vulnerability topography, and how much of this regional vulnerability is explained by cell-type composition?*"
)

data = load_app_datasets()
sim_df = data["crossdisorder_sim"]
mat_df = data["crossdisorder_mat"]
celltypes_df = data["celltypes"]

tab_sim, tab_dendro, tab_pca, tab_cells, tab_matcher = st.tabs([
    "Pairwise Similarity Matrix",
    "Hierarchical Clustering",
    "Principal Component Analysis (PCA)",
    "Cell-Type Deconvolution",
    "🎯 15-Disease Signature Matcher",
])


with tab_sim:
    st.subheader("Cross-Disorder Vulnerability Similarity Matrix")
    st.markdown("Pairwise Pearson correlation ($r$) of regional vulnerability profiles across disorders.")

    if not sim_df.empty:
        clean_labels = [c.replace("_", " ").title() for c in sim_df.columns]

        fig_heat = px.imshow(
            sim_df.values,
            x=clean_labels,
            y=clean_labels,
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            text_auto=".2f",
            labels=dict(color="Pearson r"),
            height=500,
        )
        fig_heat.update_layout(margin=dict(l=10, r=10, t=30, b=30))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Cross-disorder similarity matrix not found. Please run `make crossdis`.")

with tab_dendro:
    st.subheader("Cross-Disorder Hierarchical Clustering")
    st.markdown("Agglomerative hierarchical clustering using correlation distance ($1 - r$) and average linkage.")

    if not sim_df.empty and len(sim_df) >= 2:
        dist_mat = 1.0 - sim_df.values
        np.fill_diagonal(dist_mat, 0.0)
        dist_mat = np.clip(dist_mat, 0.0, 2.0)

        if len(sim_df) == 2:
            condensed = np.array([dist_mat[0, 1]])
        else:
            condensed = squareform(dist_mat, checks=False)

        Z = hierarchy.linkage(condensed, method="average")

        fig_dend, ax = plt.subplots(figsize=(8, 4), dpi=150)
        hierarchy.dendrogram(
            Z,
            labels=[c.replace("_", " ").title() for c in sim_df.columns],
            orientation="top",
            color_threshold=0.5,
            above_threshold_color="#2c3e50",
            ax=ax,
        )
        ax.set_ylabel("Correlation Distance (1 - r)", fontsize=9)
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        fig_dend.subplots_adjust(bottom=0.25)
        st.pyplot(fig_dend)
        plt.close(fig_dend)
    else:
        st.info("Insufficient disorders for dendrogram visualization.")

with tab_pca:
    st.subheader("Principal Component Analysis of Vulnerability Maps")
    st.markdown("Decomposition identifying the primary orthogonal axes of regional human brain vulnerability.")

    if not mat_df.empty and len(mat_df) >= 2 and mat_df.shape[1] >= 2:
        n_comp = min(len(mat_df), mat_df.shape[1], 2)
        pca = PCA(n_components=n_comp, random_state=42)
        scores = pca.fit_transform(mat_df.values)
        var_exp = pca.explained_variance_ratio_ * 100.0

        pca_plot_df = pd.DataFrame({
            "disorder": [d.replace("_", " ").title() for d in mat_df.index],
            "PC1": scores[:, 0],
            "PC2": scores[:, 1] if n_comp > 1 else [0.0] * len(mat_df),
        })

        fig_pca = px.scatter(
            pca_plot_df,
            x="PC1",
            y="PC2",
            text="disorder",
            labels={
                "PC1": f"Principal Component 1 ({var_exp[0]:.1f}% var explained)",
                "PC2": f"Principal Component 2 ({var_exp[1]:.1f}% var explained)" if n_comp > 1 else "PC2",
            },
            height=480,
        )
        fig_pca.update_traces(
            marker=dict(size=14, color="#2980b9", line=dict(width=1.5, color="#1a365d")),
            textposition="top center",
            textfont=dict(size=11, family="sans-serif"),
        )
        fig_pca.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        fig_pca.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

        st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.info("Cross-disorder matrix not available for PCA.")

with tab_cells:
    st.subheader("Cell-Type Deconvolution & Compositional Mediation")
    st.markdown(
        "Quantifies how much of each disorder's regional vulnerability profile is driven by "
        "underlying cellular composition (Astrocytes, Microglia, Excitatory, Inhibitory, Endothelial)."
    )

    if not celltypes_df.empty:
        spatial_ct = celltypes_df[celltypes_df["null_model"] == "spatial_coexpr"].copy()
        if spatial_ct.empty:
            spatial_ct = celltypes_df.copy()

        spatial_ct["r2_pct"] = spatial_ct["r_squared"] * 100.0
        spatial_ct["disorder_clean"] = spatial_ct["disorder"].str.replace("_", " ").str.title()

        fig_ct = px.bar(
            spatial_ct,
            x="disorder_clean",
            y="r2_pct",
            color="geneset_def" if "geneset_def" in spatial_ct.columns else None,
            barmode="group",
            text=spatial_ct["r2_pct"].apply(lambda v: f"{v:.1f}%"),
            labels={"disorder_clean": "Disorder", "r2_pct": "Variance Explained (R² %)"},
            title="Cell-Type Composition Variance Explained (R²)",
            height=420,
        )
        fig_ct.update_traces(textposition="outside")
        st.plotly_chart(fig_ct, use_container_width=True)

        st.subheader("Cell-Type Regression Summary Table")
        st.dataframe(spatial_ct, use_container_width=True)
    else:
        st.info("Cell-type regression data not found. Please run `make celltypes`.")

with tab_matcher:
    st.subheader("🎯 15-Disease Signature Fingerprint & Matcher")
    st.markdown(
        "Every disorder leaves a distinctive anatomical fingerprint across the human brain. "
        "Compare multi-lobe vulnerability radar signatures and query **any custom vulnerability profile or clinical phenotype** "
        "against all 15 reference disorders to identify which genetic signature it matches."
    )

    if not mat_df.empty:
        # Helper functions
        def get_region_lobe(region_name: str) -> str:
            if region_name.startswith("lh_") or region_name.startswith("rh_"):
                key = region_name[3:].lower()
                return LOBE_MAP.get(key, "Other Cortical")
            return "Subcortex"

        def get_lobe_profile(series_82: pd.Series) -> pd.Series:
            lobes = [get_region_lobe(r) for r in series_82.index]
            temp = pd.DataFrame({"score": series_82.values, "lobe": lobes})
            grouped = temp.groupby("lobe")["score"].mean()
            order = ["Frontal", "Temporal", "Parietal", "Occipital", "Cingulate", "Insula", "Subcortex"]
            return grouped.reindex([l for l in order if l in grouped.index]).fillna(0.0)

        disorders_list = list(mat_df.index)

        # -------------------------------------------------------------
        # Part 1: Multi-Lobe Anatomical Radar Fingerprints
        # -------------------------------------------------------------
        st.markdown("### 1. Multi-Lobe Anatomical Radar Fingerprints")
        st.markdown(
            "Select disorders to superimpose their 7-compartment anatomical profiles "
            "(Frontal, Temporal, Parietal, Occipital, Cingulate, Insula, Subcortex)."
        )

        default_select = [d for d in ["schizophrenia", "alzheimers_disease", "parkinsons_disease"] if d in disorders_list]
        selected_disorders = st.multiselect(
            "Select disorders to visualize in the Radar Fingerprint:",
            options=disorders_list,
            default=default_select,
            format_func=lambda x: x.replace("_", " ").title(),
        )

        if selected_disorders:
            fig_radar = go.Figure()
            radar_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22"]

            lobe_order = ["Frontal", "Temporal", "Parietal", "Occipital", "Cingulate", "Insula", "Subcortex"]

            for i, dis in enumerate(selected_disorders):
                l_prof = get_lobe_profile(mat_df.loc[dis])
                cats = [c for c in lobe_order if c in l_prof.index]
                vals = [l_prof[c] for c in cats]
                # close loop for radar
                cats_closed = cats + [cats[0]]
                vals_closed = vals + [vals[0]]

                c = radar_colors[i % len(radar_colors)]
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=vals_closed,
                        theta=cats_closed,
                        fill="toself",
                        fillcolor=f"rgba({int(c[1:3], 16)}, {int(c[3:5], 16)}, {int(c[5:7], 16)}, 0.15)",
                        line=dict(color=c, width=2),
                        name=dis.replace("_", " ").title(),
                    )
                )

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[-1.5, 1.5]),
                ),
                height=480,
                margin=dict(l=40, r=40, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")

        # -------------------------------------------------------------
        # Part 2: Interactive Disease Signature Matcher
        # -------------------------------------------------------------
        st.markdown("### 2. Interactive Disease Signature Matcher")
        st.markdown(
            "**'Which disorder is your signature?'** - Feed in a clinical pattern, custom gene profile, or interactive lobe weights. "
            "VulnMap will calculate full-brain concordance across all 15 reference disorders and rank matches."
        )

        input_mode = st.radio(
            "Select Query Profile Source:",
            [
                "🏥 Clinical Neuropathology Phenotype Presets",
                "⚡ Import Custom Gene-Set Result (from Page 4)",
                "🎛️ Interactive Anatomical Lobe Tuning",
            ],
            horizontal=True,
        )

        query_vector = None
        query_title = "Query Signature"

        if input_mode == "🏥 Clinical Neuropathology Phenotype Presets":
            preset_options = {
                "Frontotemporal Neurodegeneration (FTD Pattern)": {
                    "Frontal": 1.4, "Temporal": 1.1, "Insula": 0.8, "Cingulate": 0.5, "Parietal": -0.4, "Occipital": -0.8, "Subcortex": -0.2
                },
                "Mesial Temporal / Hippocampal Sclerosis (Epilepsy/AD Pattern)": {
                    "Temporal": 1.5, "Subcortex": 1.2, "Insula": 0.4, "Cingulate": 0.2, "Parietal": 0.1, "Frontal": -0.3, "Occipital": -0.7
                },
                "Basal Ganglia Striatal Degeneration (Parkinson / Huntington Pattern)": {
                    "Subcortex": 1.8, "Frontal": 0.3, "Insula": 0.2, "Cingulate": 0.1, "Temporal": -0.2, "Parietal": -0.3, "Occipital": -0.5
                },
                "Frontoparietal Cognitive Dysregulation (Psychosis Pattern)": {
                    "Frontal": 1.1, "Parietal": 0.9, "Cingulate": 0.7, "Temporal": 0.4, "Insula": 0.3, "Occipital": -0.5, "Subcortex": 0.2
                },
                "Diffuse Cortical Thinning & Senescence Pattern": {
                    "Frontal": 0.8, "Temporal": 0.8, "Parietal": 0.7, "Occipital": 0.5, "Cingulate": 0.6, "Insula": 0.6, "Subcortex": 0.0
                },
            }
            selected_preset_name = st.selectbox("Choose a Clinical Phenotype Preset:", list(preset_options.keys()))
            preset_weights = preset_options[selected_preset_name]
            query_title = selected_preset_name

            # Broadcast lobe weights to 82 regions
            region_weights = [preset_weights.get(get_region_lobe(c), 0.0) for c in mat_df.columns]
            query_vector = pd.Series(region_weights, index=mat_df.columns)

        elif input_mode == "⚡ Import Custom Gene-Set Result (from Page 4)":
            if "custom_res" in st.session_state and "results_df" in st.session_state["custom_res"]:
                custom_df = st.session_state["custom_res"]["results_df"]
                score_dict = dict(zip(custom_df["region"], custom_df["z"]))
                query_vector = pd.Series([score_dict.get(c, 0.0) for c in mat_df.columns], index=mat_df.columns)
                query_title = f"Custom Gene Set ({len(st.session_state['custom_res'].get('valid_genes', []))} genes)"
                st.success(f"Loaded live vulnerability profile for: {query_title}")
            else:
                st.warning("No custom gene set run found in session. Please run an analysis on **Page 4 (Custom Gene List)** first, or use a clinical preset above.")

        else: # Interactive Lobe Tuning
            st.markdown("Adjust relative vulnerability intensity (-2.0 to +2.0) across the brain lobes:")
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            with t_col1:
                w_fro = st.slider("Frontal", -2.0, 2.0, 1.0, 0.1)
                w_tem = st.slider("Temporal", -2.0, 2.0, 0.5, 0.1)
            with t_col2:
                w_par = st.slider("Parietal", -2.0, 2.0, -0.2, 0.1)
                w_occ = st.slider("Occipital", -2.0, 2.0, -0.8, 0.1)
            with t_col3:
                w_cin = st.slider("Cingulate", -2.0, 2.0, 0.4, 0.1)
                w_ins = st.slider("Insula", -2.0, 2.0, 0.3, 0.1)
            with t_col4:
                w_sub = st.slider("Subcortex", -2.0, 2.0, 0.8, 0.1)

            tuner_weights = {
                "Frontal": w_fro, "Temporal": w_tem, "Parietal": w_par,
                "Occipital": w_occ, "Cingulate": w_cin, "Insula": w_ins, "Subcortex": w_sub
            }
            query_title = "User-Tuned Lobe Profile"
            region_weights = [tuner_weights.get(get_region_lobe(c), 0.0) for c in mat_df.columns]
            query_vector = pd.Series(region_weights, index=mat_df.columns)

        # Match calculation and leaderboard
        if query_vector is not None:
            match_results = []
            q_norm = np.linalg.norm(query_vector.values)

            for dis in disorders_list:
                ref_vec = mat_df.loc[dis].values
                ref_norm = np.linalg.norm(ref_vec)

                # Pearson r
                if np.std(query_vector.values) > 1e-6 and np.std(ref_vec) > 1e-6:
                    r_val = float(np.corrcoef(query_vector.values, ref_vec)[0, 1])
                else:
                    r_val = 0.0

                # Cosine similarity
                if q_norm > 1e-6 and ref_norm > 1e-6:
                    cos_sim = float(np.dot(query_vector.values, ref_vec) / (q_norm * ref_norm))
                else:
                    cos_sim = 0.0

                # Match Score percentage
                match_pct = max(0.0, min(100.0, (cos_sim + 1.0) / 2.0 * 100.0))

                # Category badge
                if match_pct >= 75.0:
                    badge = "🟩 High Concordance"
                elif match_pct >= 55.0:
                    badge = "🟨 Moderate Concordance"
                else:
                    badge = "⬜ Divergent Profile"

                match_results.append({
                    "Disorder": dis.replace("_", " ").title(),
                    "disorder_raw": dis,
                    "Match Score (%)": match_pct,
                    "Pearson r": r_val,
                    "Cosine Similarity": cos_sim,
                    "Alignment Badge": badge,
                })

            match_df = pd.DataFrame(match_results).sort_values("Match Score (%)", ascending=False).reset_index(drop=True)
            top1 = match_df.iloc[0]
            top2 = match_df.iloc[1] if len(match_df) > 1 else None

            # Hero Summary Callout
            st.success(
                f"**🎯 Primary Match:** **{top1['Disorder']}** ({top1['Match Score (%)']:.1f}% match, "
                f"Pearson r = {top1['Pearson r']:+.3f}). The regional vulnerability pattern closely aligns with "
                f"the transcriptomic architecture of this disorder."
            )

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("🏆 Top Matching Disorder", top1["Disorder"])
            with m_col2:
                st.metric("Concordance Score", f"{top1['Match Score (%)']:.1f}%")
            with m_col3:
                st.metric("Spatial Correlation (r)", f"{top1['Pearson r']:+.3f}")

            # Dual Radar: Query vs Top 1 vs Top 2
            st.subheader("Radar Fingerprint: Query Profile vs Top Reference Matches")
            fig_compare_radar = go.Figure()

            # Query lobe profile
            q_lobe = get_lobe_profile(query_vector)
            lobe_order = ["Frontal", "Temporal", "Parietal", "Occipital", "Cingulate", "Insula", "Subcortex"]
            cats = [c for c in lobe_order if c in q_lobe.index]
            cats_closed = cats + [cats[0]]

            # Query trace
            q_vals = [q_lobe[c] for c in cats]
            fig_compare_radar.add_trace(
                go.Scatterpolar(
                    r=q_vals + [q_vals[0]],
                    theta=cats_closed,
                    fill="toself",
                    fillcolor="rgba(41, 128, 185, 0.2)",
                    line=dict(color="#2980b9", width=3, dash="solid"),
                    name=f"Query: {query_title}",
                )
            )

            # Top 1 trace
            t1_dis = top1["disorder_raw"]
            t1_lobe = get_lobe_profile(mat_df.loc[t1_dis])
            t1_vals = [t1_lobe[c] for c in cats]
            fig_compare_radar.add_trace(
                go.Scatterpolar(
                    r=t1_vals + [t1_vals[0]],
                    theta=cats_closed,
                    fill="none",
                    line=dict(color="#e74c3c", width=2.5),
                    name=f"#1 Match: {top1['Disorder']}",
                )
            )

            # Top 2 trace
            if top2 is not None:
                t2_dis = top2["disorder_raw"]
                t2_lobe = get_lobe_profile(mat_df.loc[t2_dis])
                t2_vals = [t2_lobe[c] for c in cats]
                fig_compare_radar.add_trace(
                    go.Scatterpolar(
                        r=t2_vals + [t2_vals[0]],
                        theta=cats_closed,
                        fill="none",
                        line=dict(color="#27ae60", width=2, dash="dot"),
                        name=f"#2 Match: {top2['Disorder']}",
                    )
                )

            fig_compare_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                height=450,
                margin=dict(l=30, r=30, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_compare_radar, use_container_width=True)

            # Ranked Leaderboard Chart & Table
            st.subheader("15-Disease Signature Alignment Leaderboard")
            fig_lead = px.bar(
                match_df.sort_values("Match Score (%)", ascending=True),
                x="Match Score (%)",
                y="Disorder",
                orientation="h",
                color="Match Score (%)",
                color_continuous_scale="Blues",
                text=match_df.sort_values("Match Score (%)", ascending=True)["Match Score (%)"].apply(lambda v: f"{v:.1f}%"),
                labels={"Disorder": "Candidate Disorder", "Match Score (%)": "Signature Match (%)"},
                height=480,
            )
            fig_lead.update_traces(textposition="outside")
            st.plotly_chart(fig_lead, use_container_width=True)

            display_cols = ["Disorder", "Match Score (%)", "Pearson r", "Cosine Similarity", "Alignment Badge"]
            st.dataframe(
                match_df[display_cols].style.format({
                    "Match Score (%)": "{:.1f}%",
                    "Pearson r": "{:+.3f}",
                    "Cosine Similarity": "{:.3f}",
                }),
                use_container_width=True,
            )
    else:
        st.info("Cross-disorder matrix not found. Please run `make crossdis`.")

