"""
VulnMap: Regional Brain Vulnerability & Risk-Gene Expression
============================================================

A pipeline to test whether regional brain vulnerability follows
risk-gene expression, using spatially aware null models.

See README.md for full documentation.
"""

__version__ = "0.1.0"

from vulnmap.viz import generate_all_figures, viz_qc_report
from vulnmap.report import generate_all_reports, report_qc_summary
