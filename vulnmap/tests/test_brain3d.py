"""
Tests for 3D Anatomical Brain Topography Visualizer (Phase 7)
"""

import pandas as pd
import pytest
import plotly.graph_objects as go

from vulnmap.brain3d import (
    create_3d_brain_plot,
    create_realistic_3d_brain,
    create_dual_3d_comparison,
    get_all_82_coordinates,
)


@pytest.fixture
def sample_scores():
    regions = [
        "lh_bankssts", "lh_superiorfrontal", "rh_superiorfrontal",
        "Left-Thalamus-Proper", "Right-Hippocampus", "lh_precuneus"
    ]
    return pd.Series([1.5, -2.1, -1.8, 0.9, -0.4, 1.2], index=regions)


def test_get_all_82_coordinates():
    coords = get_all_82_coordinates()
    assert len(coords) == 82
    assert "x" in coords.columns and "y" in coords.columns and "z" in coords.columns
    assert "lh_bankssts" in coords.index
    assert "Left-Thalamus-Proper" in coords.index


def test_create_realistic_3d_brain(sample_scores):
    fig = create_realistic_3d_brain(sample_scores, title="Test Realistic Brain", surface_type="inflated")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2  # Left and right mesh traces (+ subcortical markers)


def test_create_3d_brain_plot_realistic_mode(sample_scores):
    fig = create_3d_brain_plot(sample_scores, render_mode="realistic")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2


def test_create_3d_brain_plot_glass_mode(sample_scores):
    fig = create_3d_brain_plot(sample_scores, render_mode="glass_centroids")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Glass outline + parcel spheres


def test_create_dual_3d_comparison(sample_scores):
    fig_v, fig_d, r = create_dual_3d_comparison(sample_scores, sample_scores, render_mode="realistic")
    assert isinstance(fig_v, go.Figure)
    assert isinstance(fig_d, go.Figure)
    assert isinstance(r, float)
    assert pytest.approx(r, rel=1e-3) == 1.0
