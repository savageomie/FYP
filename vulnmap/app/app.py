"""
VulnMap: Regional Brain Vulnerability & Risk-Gene Expression Dashboard
======================================================================

Main entry point for the Streamlit interactive neuroimaging & transcriptomics
application (Phase 7).
"""

import sys
from pathlib import Path

# Add project root and src to path for direct invocation
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import streamlit as st
import pandas as pd
import numpy as np

from vulnmap.app_service import load_app_datasets, get_available_disorders


st.set_page_config(
    page_title="VulnMap | Regional Brain Vulnerability",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .kpi-num {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2b6cb0;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .phase-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #ebf8ff;
        color: #2b6cb0;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load pipeline data
data = load_app_datasets()
enrich_df = data["enrichment"]
damage_df = data["damage"]
disorders = data["disorders"]

# Header
st.markdown('<div class="main-title">🧠 VulnMap Explorer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Mapping Regional Human Brain Vulnerability from Risk-Gene Expression & Spatially Aware Null Models</div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# Key Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-num">82</div>
            <div class="kpi-label">Brain Regions (DK)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    n_disorders = len(disorders) if disorders else 15
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-num">{n_disorders}</div>
            <div class="kpi-label">Disorders Analyzed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-num">4</div>
            <div class="kpi-label">Null Models Audited</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-num">10,000</div>
            <div class="kpi-label">Permutations / Test</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-num">100%</div>
            <div class="kpi-label">Reproducible (Seed=42)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# Scientific Overview & Hypotheses
tab1, tab2, tab3 = st.tabs(["Scientific Hypotheses", "Pipeline Architecture", "Data & Coverage"])

with tab1:
    st.subheader("Research Objectives & Tested Hypotheses")
    h_col1, h_col2, h_col3 = st.columns(3)

    with h_col1:
        st.markdown("### 🔬 Hypothesis 1")
        st.markdown("**Regional Brain Vulnerability Follows Risk-Gene Expression**")
        st.write(
            "Do disorder-associated risk genes show statistically elevated expression in specific brain "
            "regions? We evaluate whether regional enrichment survives stringent spatial autocorrelation "
            "control (**Null Model Shrinkage**)."
        )
        st.info("Explore in: **1_Regional_Enrichment**")

    with h_col2:
        st.markdown("### 🧲 Hypothesis 2")
        st.markdown("**Transcriptomic Risk Aligns with MRI Cortical Damage**")
        st.write(
            "Does high genetic risk expression in a brain region predict in vivo structural atrophy "
            "or cortical thinning observed in patient populations (ENIGMA)? Significance is tested "
            "using Alexander-Bloch spherical spin permutations."
        )
        st.info("Explore in: **2_Neuroimaging_Damage**")

    with h_col3:
        st.markdown("### 🧬 Hypothesis 3")
        st.markdown("**Shared Vulnerability Architecture Across Disorders**")
        st.write(
            "Do psychiatric disorders share spatial vulnerability patterns distinct from neurodegenerative "
            "conditions? How much of this vulnerability is explained by underlying cell-type composition?"
        )
        st.info("Explore in: **3_Cross_Disorder_CellTypes**")

with tab2:
    st.subheader("Seven-Phase End-to-End Pipeline")
    p_cols = st.columns(4)

    with p_cols[0]:
        st.markdown('<span class="phase-badge">Phase 1</span>', unsafe_allow_html=True)
        st.markdown("**AHBA Expression**")
        st.caption("6 adult human post-mortem donors, scaled robust sigmoid normalization, differential stability filtering (DS >= 0.10).")

        st.markdown('<span class="phase-badge">Phase 2</span>', unsafe_allow_html=True)
        st.markdown("**Polygenic Risk Sets**")
        st.caption("GWAS Catalog & MAGMA integration, 5e-8 and 1e-5 thresholds, gene symbol harmonization.")

    with p_cols[1]:
        st.markdown('<span class="phase-badge">Phase 3</span>', unsafe_allow_html=True)
        st.markdown("**Enrichment & Shrinkage**")
        st.caption("Regional z-scoring against Naive, Matched, and Spatial Co-expression null models with BH-FDR correction.")

        st.markdown('<span class="phase-badge">Phase 4</span>', unsafe_allow_html=True)
        st.markdown("**Damage Map Alignment**")
        st.caption("ENIGMA clinical MRI cortical thickness correlation and Alexander-Bloch spherical spin testing (10,000 perms).")

    with p_cols[2]:
        st.markdown('<span class="phase-badge">Phase 5</span>', unsafe_allow_html=True)
        st.markdown("**Cross-Disorder & Cell Types**")
        st.caption("Cell-type deconvolution (R^2), hierarchical clustering, PCA decomposition, category H3 permutation test.")

        st.markdown('<span class="phase-badge">Phase 6</span>', unsafe_allow_html=True)
        st.markdown("**Publication Figures**")
        st.caption("Publication-ready multi-panel vector figures (Figures 1-5) and summary tables (Tables 1-6).")

    with p_cols[3]:
        st.markdown('<span class="phase-badge">Phase 7</span>', unsafe_allow_html=True)
        st.markdown("**Interactive Web App**")
        st.caption("Live exploratory dashboards, interactive Plotly visualizations, custom user gene set testing.")

with tab3:
    st.subheader("Data Sources & Clinical Harmonization")
    st.markdown(
        """
        - **Allen Human Brain Atlas (AHBA)**: 6 post-mortem donors mapped to the Desikan-Killiany 82-region atlas (68 cortical parcels + 14 subcortical nuclei).
        - **GWAS Catalog & PGC**: Harmonized risk gene sets for Schizophrenia, Bipolar Disorder, Major Depressive Disorder, Autism Spectrum Disorder, ADHD, Parkinson's Disease, and Alzheimer's Disease.
        - **ENIGMA Consortium**: Case-control structural MRI cortical thickness effect sizes (Cohen's d).
        """
    )
    if not enrich_df.empty:
        st.write(f"Precomputed enrichment database contains **{len(enrich_df):,}** regional data points across **{len(enrich_df['disorder'].unique())}** disorders.")

st.markdown("---")

# Quick Navigation Links
st.subheader("Quick Navigation")
nav1, nav2, nav3, nav4, nav5 = st.columns(5)

with nav1:
    if st.button("📊 Regional Enrichment", use_container_width=True):
        st.switch_page("pages/1_Regional_Enrichment.py")
with nav2:
    if st.button("🧲 Damage Alignment", use_container_width=True):
        st.switch_page("pages/2_Neuroimaging_Damage.py")
with nav3:
    if st.button("🧬 Cross-Disorder", use_container_width=True):
        st.switch_page("pages/3_Cross_Disorder_CellTypes.py")
with nav4:
    if st.button("⚡ Test Custom Genes", use_container_width=True):
        st.switch_page("pages/4_Custom_Gene_List.py")
with nav5:
    if st.button("📑 Robustness & Figures", use_container_width=True):
        st.switch_page("pages/5_Robustness_Reports.py")
