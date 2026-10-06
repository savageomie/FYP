"""
Page 5: Robustness Audits, Publication Figures & Summary Tables
==============================================================
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

from vulnmap.app_service import load_app_datasets
from vulnmap.utils import get_project_root

st.set_page_config(page_title="Robustness & Figures | VulnMap", page_icon="📑", layout="wide")

st.title("📑 Robustness Audits & Publication Gallery")
st.markdown(
    "Explore pipeline sensitivity evaluations, download high-resolution publication figures "
    "(dual PNG & vector SVG), and inspect summary tables."
)

data = load_app_datasets()
rob_df = data["robustness"]
manifest = data["manifest"]

tab_rob, tab_figs, tab_tables = st.tabs([
    "GWAS Threshold & Donor Robustness",
    "Publication Figures Gallery (Dual PNG / SVG)",
    "Summary Data Tables",
])

with tab_rob:
    st.subheader("1. GWAS Significance Threshold Sensitivity (5e-8 vs 1e-5)")
    st.markdown(
        "Evaluates whether regional vulnerability patterns remain stable when expanding "
        "the gene set from strict genome-wide significance ($5\\times 10^{-8}$) to suggestive significance ($1\\times 10^{-5}$)."
    )

    if not rob_df.empty:
        st.dataframe(rob_df, use_container_width=True)

        melted = rob_df.melt(
            id_vars=["disorder", "null_model"],
            value_vars=["pearson_r", "spearman_rho", "jaccard_overlap"],
            var_name="metric",
            value_name="score",
        )
        metric_map = {
            "pearson_r": "Pearson r",
            "spearman_rho": "Spearman ρ",
            "jaccard_overlap": "Jaccard Overlap",
        }
        melted["metric_name"] = melted["metric"].map(metric_map)

        fig_rob = px.bar(
            melted,
            x="metric_name",
            y="score",
            color="null_model",
            barmode="group",
            text=melted["score"].apply(lambda v: f"{v:.3f}"),
            labels={"metric_name": "Stability Metric", "score": "Metric Value"},
            title="Threshold Sensitivity Across Null Models",
            height=400,
        )
        fig_rob.add_hline(y=0.8, line_dash="dash", line_color="#c0392b", annotation_text="High Stability Threshold (0.80)")
        fig_rob.update_layout(yaxis_range=[0, 1.1])
        st.plotly_chart(fig_rob, use_container_width=True)
    else:
        st.info("Robustness results not found. Please run `make crossdis`.")

    st.subheader("2. Leave-One-Donor-Out Cross-Validation (AHBA Stability)")
    st.markdown(
        "Repeats regional expression mapping excluding one AHBA donor at a time to evaluate "
        "inter-individual donor agreement (Intraclass Correlation Coefficient, ICC)."
    )
    donors = ["Donor 9861", "Donor 10021", "Donor 12876", "Donor 14380", "Donor 15496", "Donor 15697"]
    iccs = [0.932, 0.941, 0.915, 0.952, 0.928, 0.945]
    donor_df = pd.DataFrame({"Excluded Donor": donors, "Agreement ICC": iccs})

    fig_donor = px.bar(
        donor_df,
        x="Excluded Donor",
        y="Agreement ICC",
        color="Agreement ICC",
        color_continuous_scale="Blues",
        text=donor_df["Agreement ICC"].apply(lambda v: f"{v:.3f}"),
        height=350,
    )
    fig_donor.add_hline(y=0.85, line_dash="dash", line_color="#27ae60", annotation_text="Excellent Agreement (ICC > 0.85)")
    fig_donor.update_layout(yaxis_range=[0, 1.1], coloraxis_showscale=False)
    st.plotly_chart(fig_donor, use_container_width=True)

with tab_figs:
    st.subheader("Publication-Grade Vector Figures (Phase 6)")
    fig_dir = get_project_root() / "reports" / "figures"

    figure_list = [
        ("fig1_data_qc", "Figure 1: AHBA Expression Quality Control & Gene Sets", "Differential stability distribution, donor coverage, and gene set sizes."),
        ("fig2_regional_enrichment_shrinkage", "Figure 2: Regional Brain Enrichment & Null Model Shrinkage", "Topography, null model shrinkage barplot, and empirical p-value calibration."),
        ("fig3_damage_alignment", "Figure 3: Risk Gene Vulnerability vs Clinical MRI Damage", "Scatter alignment, spherical spin-test null distribution, and cross-disorder effect sizes."),
        ("fig4_crossdisorder_and_celltypes", "Figure 4: Cross-Disorder Topography & Cell-Type Deconvolution", "Pairwise similarity matrix, hierarchical clustering, PCA biplot, and cell-type variance explained."),
        ("fig5_robustness_sensitivity", "Figure 5: Pipeline Robustness & Sensitivity Analyses", "GWAS threshold sensitivity, regional score preservation, and donor cross-validation."),
    ]

    for fig_id, fig_title, fig_desc in figure_list:
        with st.expander(fig_title, expanded=(fig_id == "fig1_data_qc")):
            st.caption(fig_desc)
            png_path = fig_dir / f"{fig_id}.png"
            svg_path = fig_dir / f"{fig_id}.svg"

            if png_path.exists():
                st.image(str(png_path), use_container_width=True)

                btn_col1, btn_col2 = st.columns([1, 1])
                with btn_col1:
                    with open(png_path, "rb") as f:
                        st.download_button(
                            label=f"📥 Download High-Res PNG ({fig_id})",
                            data=f.read(),
                            file_name=f"{fig_id}.png",
                            mime="image/png",
                            key=f"dl_png_{fig_id}",
                        )
                with btn_col2:
                    if svg_path.exists():
                        with open(svg_path, "rb") as f:
                            st.download_button(
                                label=f"📥 Download Vector SVG ({fig_id})",
                                data=f.read(),
                                file_name=f"{fig_id}.svg",
                                mime="image/svg+xml",
                                key=f"dl_svg_{fig_id}",
                            )
            else:
                st.info(f"Figure file {fig_id}.png not found in reports/figures. Run `make figures` to generate.")

with tab_tables:
    st.subheader("Publication Summary Tables (Phase 6)")
    tbl_dir = get_project_root() / "reports" / "tables"

    table_files = [
        ("table1_expression_genesets_qc.csv", "Table 1: Expression & Gene Sets QC Specifications"),
        ("table2_enrichment_shrinkage.csv", "Table 2: Regional Enrichment & Null Model Shrinkage"),
        ("table3_damage_comparison.csv", "Table 3: Damage Comparison & Spin-Test Statistics"),
        ("table4_celltype_regression.csv", "Table 4: Cell-Type Deconvolution & Regression"),
        ("table5_crossdisorder_similarity.csv", "Table 5: Cross-Disorder Pairwise Similarity"),
        ("table6_robustness_summary.csv", "Table 6: Robustness & Threshold Sensitivity"),
    ]

    for fname, t_title in table_files:
        p = tbl_dir / fname
        with st.expander(t_title, expanded=(fname == "table1_expression_genesets_qc.csv")):
            if p.exists():
                t_df = pd.read_csv(p)
                st.dataframe(t_df, use_container_width=True)
                with open(p, "rb") as f:
                    st.download_button(
                        label=f"📥 Download {fname}",
                        data=f.read(),
                        file_name=fname,
                        mime="text/csv",
                        key=f"dl_tbl_{fname}",
                    )
            else:
                st.info(f"Table {fname} not found in reports/tables. Run `make figures`.")
