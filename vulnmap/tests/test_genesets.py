"""
Tests for VulnMap Phase 2: Gene Set Retrieval & Harmonization
==============================================================

These tests verify:
1. Gene extraction from GWAS association records.
2. Gene symbol harmonization (case-insensitive matching).
3. Jaccard overlap computation.
4. MAGMA file parsing.
5. Underpowered disorder detection.

Tests use synthetic data — no network calls or real API access.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Fixtures: synthetic data
# =============================================================================

@pytest.fixture
def synthetic_associations():
    """Create synthetic GWAS Catalog association records."""
    return [
        {
            "pvalueMantissa": 3.2,
            "pvalueExponent": -10,
            "snps": [
                {
                    "rsId": "rs12345",
                    "genes": [
                        {"geneName": "BRCA1"},
                        {"geneName": "TP53"},
                    ],
                }
            ],
            "loci": [
                {
                    "authorReportedGenes": [
                        {"geneName": "BRCA1"},
                    ],
                    "strongestRiskAlleles": [],
                }
            ],
        },
        {
            "pvalueMantissa": 1.5,
            "pvalueExponent": -8,
            "snps": [
                {
                    "rsId": "rs67890",
                    "genes": [
                        {"geneName": "DISC1"},
                    ],
                }
            ],
            "loci": [
                {
                    "authorReportedGenes": [
                        {"geneName": "DISC1"},
                        {"geneName": "NRG1"},
                    ],
                    "strongestRiskAlleles": [],
                }
            ],
        },
        {
            # This association should NOT pass the 5e-8 threshold
            "pvalueMantissa": 5.0,
            "pvalueExponent": -6,
            "snps": [
                {
                    "rsId": "rs11111",
                    "genes": [
                        {"geneName": "FOXP2"},
                    ],
                }
            ],
            "loci": [
                {
                    "authorReportedGenes": [
                        {"geneName": "FOXP2"},
                    ],
                    "strongestRiskAlleles": [],
                }
            ],
        },
    ]


@pytest.fixture
def synthetic_reference_genes():
    """Synthetic AHBA gene list."""
    return [
        "BRCA1", "TP53", "DISC1", "NRG1", "COMT", "DRD2",
        "SLC6A4", "BDNF", "APOE", "HTT", "FOXP2", "GRM3",
        "CACNA1C", "ANK3", "TCF4", "NRGN",
    ]


# =============================================================================
# Tests: Gene extraction
# =============================================================================

class TestGeneExtraction:
    """Tests for extract_genes_from_associations."""

    def test_genome_wide_threshold(self, synthetic_associations):
        """Only associations with p < 5e-8 should pass."""
        from vulnmap.genesets import extract_genes_from_associations

        result = extract_genes_from_associations(
            synthetic_associations, p_threshold=5e-8
        )
        # Associations 1 and 2 pass (p=3.2e-10 and 1.5e-8)
        # Association 3 does NOT (p=5e-6)
        assert result["n_significant"] == 2
        assert "FOXP2" not in result["all_genes"]

    def test_suggestive_threshold(self, synthetic_associations):
        """All three associations should pass at suggestive threshold."""
        from vulnmap.genesets import extract_genes_from_associations

        result = extract_genes_from_associations(
            synthetic_associations, p_threshold=1e-5
        )
        assert result["n_significant"] == 3
        assert "FOXP2" in result["all_genes"]

    def test_genes_are_strings(self, synthetic_associations):
        """All returned genes should be strings."""
        from vulnmap.genesets import extract_genes_from_associations

        result = extract_genes_from_associations(
            synthetic_associations, p_threshold=1e-5
        )
        for gene in result["all_genes"]:
            assert isinstance(gene, str)

    def test_no_duplicates(self, synthetic_associations):
        """Gene sets should not contain duplicates."""
        from vulnmap.genesets import extract_genes_from_associations

        result = extract_genes_from_associations(
            synthetic_associations, p_threshold=1e-5
        )
        # all_genes is a set, so no duplicates by construction
        assert isinstance(result["all_genes"], set)

    def test_snps_extracted(self, synthetic_associations):
        """SNP rs IDs should be extracted."""
        from vulnmap.genesets import extract_genes_from_associations

        result = extract_genes_from_associations(
            synthetic_associations, p_threshold=1e-5
        )
        assert len(result["snps"]) > 0
        assert all(s.startswith("rs") for s in result["snps"])

    def test_empty_associations(self):
        """Empty input should return empty gene sets."""
        from vulnmap.genesets import extract_genes_from_associations

        result = extract_genes_from_associations([], p_threshold=5e-8)
        assert len(result["all_genes"]) == 0
        assert result["n_significant"] == 0


# =============================================================================
# Tests: Gene harmonization
# =============================================================================

class TestHarmonization:
    """Tests for gene symbol harmonization."""

    def test_exact_match(self, synthetic_reference_genes):
        """Exact case matches should work."""
        from vulnmap.genesets import harmonize_genes

        result = harmonize_genes(
            {"BRCA1", "TP53", "DISC1"},
            synthetic_reference_genes,
        )
        assert result["n_matched"] == 3
        assert result["n_input"] == 3
        assert result["match_rate"] == 1.0

    def test_case_insensitive(self, synthetic_reference_genes):
        """Case-insensitive matching should work."""
        from vulnmap.genesets import harmonize_genes

        result = harmonize_genes(
            {"brca1", "tp53"},
            synthetic_reference_genes,
        )
        assert result["n_matched"] == 2

    def test_unmatched_genes(self, synthetic_reference_genes):
        """Genes not in reference should be reported."""
        from vulnmap.genesets import harmonize_genes

        result = harmonize_genes(
            {"BRCA1", "NONEXISTENT_GENE", "FAKE_GENE"},
            synthetic_reference_genes,
        )
        assert result["n_matched"] == 1
        assert len(result["unmatched"]) == 2
        assert "NONEXISTENT_GENE" in result["unmatched"]

    def test_harmonized_subset_of_reference(self, synthetic_reference_genes):
        """Matched genes should be a subset of reference genes."""
        from vulnmap.genesets import harmonize_genes

        result = harmonize_genes(
            {"BRCA1", "TP53", "DISC1", "FAKE_GENE"},
            synthetic_reference_genes,
        )
        ref_set = set(synthetic_reference_genes)
        for gene in result["matched"]:
            assert gene in ref_set

    def test_empty_query(self, synthetic_reference_genes):
        """Empty query should return empty matches."""
        from vulnmap.genesets import harmonize_genes

        result = harmonize_genes(set(), synthetic_reference_genes)
        assert result["n_matched"] == 0
        assert result["match_rate"] == 0.0


# =============================================================================
# Tests: Jaccard overlap
# =============================================================================

class TestJaccardOverlap:
    """Tests for Jaccard similarity matrix."""

    def test_jaccard_range(self):
        """Jaccard values should be in [0, 1]."""
        from vulnmap.genesets import compute_jaccard_matrix

        gene_sets = {
            "A": {"G1", "G2", "G3"},
            "B": {"G2", "G3", "G4"},
            "C": {"G5", "G6"},
        }
        matrix = compute_jaccard_matrix(gene_sets)
        assert (matrix.values >= 0).all()
        assert (matrix.values <= 1).all()

    def test_jaccard_diagonal(self):
        """Diagonal should be 1.0 (self-similarity)."""
        from vulnmap.genesets import compute_jaccard_matrix

        gene_sets = {
            "A": {"G1", "G2"},
            "B": {"G3", "G4"},
        }
        matrix = compute_jaccard_matrix(gene_sets)
        np.testing.assert_array_equal(np.diag(matrix.values), [1.0, 1.0])

    def test_jaccard_symmetric(self):
        """Matrix should be symmetric."""
        from vulnmap.genesets import compute_jaccard_matrix

        gene_sets = {
            "A": {"G1", "G2", "G3"},
            "B": {"G2", "G3", "G4"},
            "C": {"G1"},
        }
        matrix = compute_jaccard_matrix(gene_sets)
        np.testing.assert_array_almost_equal(
            matrix.values, matrix.values.T
        )

    def test_jaccard_known_value(self):
        """Test with known Jaccard: |{G2,G3}|/|{G1,G2,G3,G4}| = 2/4 = 0.5."""
        from vulnmap.genesets import compute_jaccard_matrix

        gene_sets = {
            "A": {"G1", "G2", "G3"},
            "B": {"G2", "G3", "G4"},
        }
        matrix = compute_jaccard_matrix(gene_sets)
        assert abs(matrix.loc["A", "B"] - 0.5) < 1e-10

    def test_jaccard_disjoint(self):
        """Disjoint sets should have Jaccard = 0."""
        from vulnmap.genesets import compute_jaccard_matrix

        gene_sets = {
            "A": {"G1", "G2"},
            "B": {"G3", "G4"},
        }
        matrix = compute_jaccard_matrix(gene_sets)
        assert matrix.loc["A", "B"] == 0.0


# =============================================================================
# Tests: MAGMA integration
# =============================================================================

class TestMagmaIntegration:
    """Tests for MAGMA file loading."""

    def test_missing_file(self, tmp_path):
        """Missing MAGMA file should return empty results."""
        from vulnmap.genesets import load_magma_genes

        result = load_magma_genes(tmp_path / "nonexistent.genes.out")
        assert len(result["genes"]) == 0
        assert result["n_total"] == 0

    def test_valid_magma_file(self, tmp_path):
        """Parse a synthetic MAGMA output file."""
        from vulnmap.genesets import load_magma_genes

        # Create synthetic MAGMA output
        magma_data = pd.DataFrame({
            "GENE": ["BRCA1", "TP53", "DISC1", "NRG1", "COMT"],
            "NSNPS": [100, 80, 60, 40, 20],
            "ZSTAT": [4.5, 3.8, 2.1, 1.5, 0.3],
            "P": [3.4e-6, 7.2e-5, 0.018, 0.067, 0.382],
        })
        magma_file = tmp_path / "test.genes.out"
        magma_data.to_csv(magma_file, sep='\t', index=False)

        result = load_magma_genes(magma_file, fdr_threshold=0.05)
        assert result["n_total"] == 5
        # With BH-FDR at 0.05, the most significant genes should pass
        assert result["n_significant"] >= 1
        assert isinstance(result["genes"], set)
        for gene in result["genes"]:
            assert isinstance(gene, str)
