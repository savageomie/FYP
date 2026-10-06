# Comparison with Fulcher et al. (2021) Methodology

## Conceptual Alignment
Fulcher et al. (2021, *Nature Communications*) demonstrated that >80% of published imaging-transcriptomics gene category enrichment associations are likely false positives driven by brain-wide spatial autocorrelation.

## VulnMap Enhancements
1. **Multi-Null Hierarchy**: Direct side-by-side execution of Naive, Matched, and Spatial nulls on the identical dataset, explicitly outputting the **Shrinkage Metric**.
2. **Clinical Validation**: Incorporates in vivo patient structural MRI damage maps (ENIGMA) with Alexander-Bloch spherical spin tests to evaluate whether surviving transcriptomic regions reflect actual clinical atrophy.
3. **Interactive Accessibility**: Full Streamlit dashboard enabling real-time experimentation with custom candidate gene sets.
