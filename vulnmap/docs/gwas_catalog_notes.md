# GWAS Catalog & MAGMA Integration Notes

## Polygenic Thresholding
- **Genome-Wide Significance ($p < 5\times 10^{-8}$)**:
  Stringent Bonferroni correction for ~1 million independent common variant tests. Genes mapped to these loci have highest specificity but smaller set sizes ($N \approx 20 - 150$).
- **Suggestive Significance ($p < 1\times 10^{-5}$)**:
  Captures polygenic sub-threshold signal. Increases power for understudied disorders, providing robustness comparisons.

## Verification of EFO IDs
All disease traits are mapped to Experimental Factor Ontology (EFO) terms in `config/disorders.yaml`.
Researchers must verify that EFO terms reflect the intended diagnosis (e.g., adult onset vs childhood subtype).
