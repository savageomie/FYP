# Thesis Outline: Transcriptomic Brain Vulnerability Pipeline

**Project Title:** Mapping Regional Human Brain Vulnerability from Risk-Gene Expression & Spatially Aware Null Models  
**Framework:** VulnMap  
**Date Generated:** 2026-10-05 22:22:00  

---

## Chapter 1: Introduction & Theoretical Foundation
- **1.1 The Spatial Vulnerability Paradigm**: Explores why systemic polygenic risk translates into stereotypic, region-specific neuropathology in psychiatric and neurodegenerative disorders.
- **1.2 Imaging-Transcriptomics**: Overview of the Allen Human Brain Atlas (AHBA) and the ENIGMA neuroimaging consortium as bridges between molecular genetics and macroscale structural pathology.
- **1.3 The Spatial Autocorrelation Problem**: Explains how brain tissue spatial autocorrelation creates severe false-positive inflation in naive statistical testing, motivating the VulnMap pipeline.
- **1.4 Research Questions & Hypotheses (H1–H4)**:
  - *H1 (Regional Enrichment)*: Regional brain vulnerability tracks disorder-associated risk-gene expression.
  - *H2 (Damage Alignment)*: Transcriptomic vulnerability predicts empirical clinical structural MRI cortical thinning (ENIGMA).
  - *H3 (Cross-Disorder Topography)*: Psychiatric disorders share a coherent vulnerability architecture distinct from neurodegenerative disorders.
  - *H4 (Cellular Composition)*: Regional vulnerability patterns are partially mediated by underlying cell-type distributions.

---

## Chapter 2: Materials & Methodology
- **2.1 AHBA Microarray Processing**: Detailed description of 6 post-mortem adult donors, scaled robust sigmoid (SRS) normalization, and probe selection by differential stability (DS >= 0.1).
- **2.2 Parcellation Architecture**: Structural Desikan-Killiany parcellation (68 cortical parcels + 14 subcortical nuclei = 82 anatomical units) aligned directly with ENIGMA clinical damage maps.
- **2.3 Polygenic Risk Gene Sets**: GWAS Catalog extraction, MAGMA gene-level scoring, and threshold definitions (genome-wide $5\times 10^{-8}$ vs suggestive $1\times 10^{-5}$).
- **2.4 Spatial Null Models & The Shrinkage Framework**:
  - *Naive Null*: Uniform permutation across the genome.
  - *Matched Null*: Stratification by gene length, GC content, and mean brain expression.
  - *Spatial Co-expression Null*: Sampling preserving empirical inter-regional spatial autocorrelation (Moran's I matching).
- **2.5 Alexander-Bloch Spherical Spin Tests**: Geodesic rotation on the sphere to evaluate cortical damage alignment while preserving spatial topology.
- **2.6 Cellular Deconvolution & Cross-Disorder Structure**: Cell-type regression against canonical marker profiles, hierarchical clustering (average linkage), and PCA decomposition.

---

## Chapter 3: Results
- **3.1 AHBA Data Quality Control & Differential Stability**: Retention of high-stability genes, donor coverage uniformity, and GWAS polygenic overlap.
- **3.2 Hypothesis 1 Results — Null Model Shrinkage Audit**: Quantifying the collapse of spurious significance from Naive to Spatial Co-expression nulls across disorders.
- **3.3 Hypothesis 2 Results — Neuroimaging Damage Map Alignment**: Empirical correlation between transcriptomic vulnerability and ENIGMA cortical thinning; spin-test statistical evaluation.
- **3.4 Hypothesis 3 Results — Cross-Disorder Shared Architectures**: Pairwise correlation matrices, clustering of psychiatric vs neurodegenerative disorders, and dominant principal components of brain vulnerability.
- **3.5 Cellular Deconvolution**: Variance explained ($R^2$) by cell classes (Astrocytes, Microglia, Excitatory, Inhibitory, Endothelial) and residual vulnerability preservation.
- **3.6 Robustness & Sensitivity Checks**: Stability across GWAS thresholds ($5\times 10^{-8}$ vs $1\times 10^{-5}$) and leave-one-donor-out cross-validation agreement.

---

## Chapter 4: Discussion & Synthesis
- **4.1 Biological Implications**: What regional risk expression reveals about the selective vulnerability of association cortex vs sensory-motor cortex.
- **4.2 The Methodological Imperative of Spatial Nulls**: Why published imaging-transcriptomic findings using naive permutations must be re-evaluated.
- **4.3 Convergence and Divergence Across Clinical Phenotypes**: Interpretation of shared vulnerability between schizophrenia and bipolar disorder.

---

## Chapter 5: Limitations, Future Work & Conclusion
- **5.1 Limitations**: AHBA hemisphere bias (predominantly left hemisphere donors), adult post-mortem cross-sectional nature, and missing fine subcortical structures (e.g. substantia nigra).
- **5.2 Future Directions**: Single-cell spatial transcriptomics (MERFISH/Visium), longitudinal neuroimaging cohorts, and multi-ancestry GWAS.
- **5.3 Concluding Remarks**: Summary of VulnMap contributions to computational neurogenomics.
