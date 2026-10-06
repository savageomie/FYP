"""
Tests for VulnMap Phase 8: Write-Up Support & Reporting
======================================================

Verifies:
1. Environment specs and SHA-256 file checksums.
2. Thesis outline generation (Chapters 1-5).
3. Figures and tables index mapping.
4. Hypotheses evaluation table (H1-H4).
5. Methods summary and reproducibility checklist.
6. BibTeX references validity.
7. Technical methodology guides in docs/.
8. Master orchestrator (generate_all_reports) and QC summary.
"""

from pathlib import Path
import pytest

from vulnmap.report import (
    compute_file_checksum,
    generate_all_reports,
    generate_docs_explainers,
    generate_methods_summary,
    generate_references_bib,
    generate_reproducibility_checklist,
    generate_results_hypotheses_table,
    generate_thesis_figures_index,
    generate_thesis_outline,
    get_environment_specs,
    report_qc_summary,
)


@pytest.fixture
def mock_params():
    return {
        "seed": 42,
        "mode": "FAST",
        "n_perm_fast": 1000,
        "expression": {"ds_threshold": 0.10},
        "enrichment": {"spatial_null": {"moran_tolerance": 0.1}},
        "damage": {"spin_n_perm": 10000},
    }


class TestEnvironmentAndProvenance:
    """Test environment inspection and file checksums."""

    def test_get_environment_specs(self):
        specs = get_environment_specs()
        assert "os" in specs
        assert "python" in specs
        assert "numpy" in specs
        assert "scipy" in specs
        assert "pandas" in specs

    def test_compute_file_checksum(self, tmp_path):
        test_file = tmp_path / "sample.txt"
        test_file.write_text("Hello VulnMap", encoding="utf-8")
        chk = compute_file_checksum(test_file)
        assert len(chk) == 16
        assert chk.isalnum()

        missing_chk = compute_file_checksum(tmp_path / "non_existent.txt")
        assert missing_chk == "file not found"


class TestDocumentGenerators:
    """Test markdown and BibTeX generator functions."""

    def test_generate_thesis_outline(self, mock_params):
        outline = generate_thesis_outline(mock_params)
        assert "Chapter 1: Introduction" in outline
        assert "Chapter 2: Materials & Methodology" in outline
        assert "Chapter 3: Results" in outline
        assert "Chapter 4: Discussion" in outline
        assert "Chapter 5: Limitations" in outline

    def test_generate_thesis_figures_index(self):
        idx = generate_thesis_figures_index()
        assert "Figure 1" in idx
        assert "Figure 2" in idx
        assert "Figure 3" in idx
        assert "Figure 4" in idx
        assert "Figure 5" in idx
        assert "Table 1" in idx
        assert "Table 6" in idx

    def test_generate_results_hypotheses_table(self, tmp_path):
        table = generate_results_hypotheses_table(tmp_path)
        assert "H1: Regional Brain Vulnerability" in table
        assert "H2: Damage Map Alignment" in table
        assert "H3: Shared Vulnerability Architecture" in table
        assert "H4: Cellular Mediation" in table

    def test_generate_methods_summary(self, mock_params, tmp_path):
        env_specs = get_environment_specs()
        summary = generate_methods_summary(mock_params, env_specs, tmp_path)
        assert "Algorithmic Specifications" in summary
        assert "Software Runtime Environment" in summary

    def test_generate_reproducibility_checklist(self, mock_params):
        chk = generate_reproducibility_checklist(mock_params)
        assert "seed=42" in chk
        assert "make expression" in chk
        assert "make app" in chk

    def test_generate_references_bib(self):
        bib = generate_references_bib()
        assert "@article{hawrylycz2012anatomically" in bib
        assert "@article{arnatkeviciute2019hub" in bib
        assert "@article{alexander2018spin" in bib
        assert "@article{fulcher2021autocorrelation" in bib

    def test_generate_docs_explainers(self, tmp_path):
        docs = generate_docs_explainers(tmp_path)
        assert len(docs) == 3
        for key, p in docs.items():
            assert p.exists()
            assert p.stat().st_size > 0


class TestPhase8Orchestration:
    """Test full execution of Phase 8."""

    def test_generate_all_reports_end_to_end(self, tmp_path, mock_params):
        rep_dir = tmp_path / "reports"
        docs_dir = tmp_path / "docs"

        outputs = generate_all_reports(
            params=mock_params,
            reports_dir=rep_dir,
            docs_dir=docs_dir,
        )

        assert len(outputs) == 9  # 6 reports + 3 docs
        for name, path in outputs.items():
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0

        summary = report_qc_summary(outputs)
        assert "Phase 8 QC Summary" in summary
        assert "thesis_outline" in summary
