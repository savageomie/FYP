# Theoretical Framework: Null Models in Imaging Transcriptomics

## Why Standard Permutations Fail
In traditional bioinformatics, a gene set's statistical significance is tested by comparing observed enrichment against randomly selected gene sets of identical size sampled uniformly from the genome (**Naive Null Model**).

In imaging transcriptomics, this assumption is fundamentally violated:
1. **Brain Gene Co-expression**: Genes are not independently expressed. Transcriptomic networks exhibit strong modular co-expression hubs.
2. **Spatial Autocorrelation**: Anatomically proximate brain regions have similar expression profiles. Gradients across cortical hierarchies (e.g., sensory-fugal axis) induce widespread spatial smoothness.

As demonstrated by Fulcher et al. (2021), a naive random draw of genes will almost never exhibit coordinated spatial patterns, creating an artificially narrow null distribution. Consequently, observed scores appear spuriously significant.

## The Three Null Model Classes in VulnMap

### 1. Naive Permutation (`NaiveNull`)
- Samples random gene sets of size $K$ uniformly without replacement from all retained AHBA genes.
- Serves as the benchmark baseline to quantify the magnitude of false-positive inflation.

### 2. Annotation-Matched Permutation (`MatchedNull`)
- Stratifies genes into multi-dimensional quantile bins:
  - Gene genomic length (bp)
  - GC nucleotide content (%)
  - Brain-wide mean expression intensity
- Samples null gene sets that preserve identical confounding structural properties.

### 3. Spatial Co-expression / Autocorrelation (`SpatialNull`)
- Quantifies empirical spatial autocorrelation using **Moran's I** across the brain parcellation.
- Matches candidate genes with background genes exhibiting identical spatial autocorrelation profiles within tolerance.
- Controls for spatial gradients, isolating true disorder-specific biological vulnerability.
