"""
VulnMap Utilities
=================

Global seed management, parameter loading, caching, logging, and
directory helpers used throughout the pipeline.

Design decisions:
- Global seed is set once from params.yaml and propagated to all RNGs.
- Caching uses parquet for DataFrames and JSON for metadata.
- Run parameters are logged to JSON for reproducibility.
"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import yaml

# =============================================================================
# Logging setup
# =============================================================================

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the vulnmap logger."""
    logger = logging.getLogger("vulnmap")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    return logger


logger = setup_logging()


# =============================================================================
# Path helpers
# =============================================================================

def get_project_root() -> Path:
    """Return the vulnmap project root (the directory containing config/)."""
    # Walk up from this file to find the vulnmap root
    current = Path(__file__).resolve()
    # __file__ is in src/vulnmap/utils.py → go up 3 levels to vulnmap/
    root = current.parent.parent.parent
    if (root / "config").exists():
        return root
    # Fallback: try current working directory
    cwd = Path.cwd()
    if (cwd / "config").exists():
        return cwd
    raise FileNotFoundError(
        f"Cannot find VulnMap project root. Expected 'config/' directory. "
        f"Searched: {root}, {cwd}"
    )


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist. Returns the Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# Configuration
# =============================================================================

def load_params(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load pipeline parameters from params.yaml.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to params.yaml. If None, auto-discovers from project root.

    Returns
    -------
    dict
        Nested dictionary of parameters.
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "params.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {config_path}")
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
    logger.info(f"Loaded parameters from {config_path}")
    return params


def load_disorders(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load disorder definitions from disorders.yaml.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to disorders.yaml. If None, auto-discovers from project root.

    Returns
    -------
    dict
        Nested dictionary of disorder definitions.
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "disorders.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Disorders file not found: {config_path}")
    with open(config_path, "r") as f:
        disorders = yaml.safe_load(f)
    logger.info(f"Loaded {len(disorders.get('disorders', {}))} disorders from {config_path}")
    return disorders


def get_n_perm(params: Dict[str, Any]) -> int:
    """Return the number of permutations based on the mode setting."""
    mode = params.get("mode", "FAST").upper()
    if mode == "FAST":
        return params.get("n_perm_fast", 1000)
    elif mode == "FULL":
        return params.get("n_perm_full", 10000)
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'FAST' or 'FULL'.")


# =============================================================================
# Random seed management
# =============================================================================

def set_global_seed(seed: int = 42) -> np.random.Generator:
    """
    Set the global random seed for reproducibility.

    Sets numpy legacy seed (for libraries that use it) and returns
    a numpy Generator for new-style RNG.

    Parameters
    ----------
    seed : int
        Random seed.

    Returns
    -------
    np.random.Generator
        A seeded Generator instance.
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    logger.info(f"Global seed set to {seed}")
    return rng


# =============================================================================
# Caching
# =============================================================================

def _cache_path(cache_dir: Union[str, Path], name: str, suffix: str = ".parquet") -> Path:
    """Build the cache file path."""
    return Path(cache_dir) / f"{name}{suffix}"


def cache_dataframe(
    df: pd.DataFrame,
    cache_dir: Union[str, Path],
    name: str,
) -> Path:
    """
    Save a DataFrame to parquet in the cache directory.

    Parameters
    ----------
    df : pd.DataFrame
        Data to cache.
    cache_dir : str or Path
        Directory to save in.
    name : str
        Base name for the file (without extension).

    Returns
    -------
    Path
        Path to the saved file.
    """
    ensure_dir(cache_dir)
    path = _cache_path(cache_dir, name)
    df.to_parquet(path, engine="pyarrow")
    logger.info(f"Cached DataFrame ({df.shape}) to {path}")
    return path


def load_cached_dataframe(
    cache_dir: Union[str, Path],
    name: str,
) -> Optional[pd.DataFrame]:
    """
    Load a cached DataFrame if it exists.

    Returns
    -------
    pd.DataFrame or None
        The cached data, or None if no cache exists.
    """
    path = _cache_path(cache_dir, name)
    if path.exists():
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info(f"Loaded cached DataFrame ({df.shape}) from {path}")
        return df
    return None


def cache_json(data: Any, cache_dir: Union[str, Path], name: str) -> Path:
    """Save data as JSON."""
    ensure_dir(cache_dir)
    path = _cache_path(cache_dir, name, suffix=".json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Cached JSON to {path}")
    return path


def load_cached_json(cache_dir: Union[str, Path], name: str) -> Optional[Any]:
    """Load cached JSON if it exists."""
    path = _cache_path(cache_dir, name, suffix=".json")
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded cached JSON from {path}")
        return data
    return None


# =============================================================================
# Run logging
# =============================================================================

def log_run(
    phase: str,
    params: Dict[str, Any],
    output_dir: Union[str, Path],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Log run parameters to a JSON file for reproducibility.

    Parameters
    ----------
    phase : str
        Phase name (e.g., "expression", "enrichment").
    params : dict
        The parameters used for this run.
    output_dir : str or Path
        Directory to save the log.
    extra : dict, optional
        Additional metadata (e.g., runtime, output shapes).

    Returns
    -------
    Path
        Path to the log file.
    """
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_data = {
        "phase": phase,
        "timestamp": timestamp,
        "params": params,
    }
    if extra:
        log_data["extra"] = extra
    path = Path(output_dir) / f"run_{phase}_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    logger.info(f"Run log saved to {path}")
    return path


# =============================================================================
# Timer utility
# =============================================================================

def estimate_runtime(n_perm: int, n_items: int, time_per_item_per_perm: float = 0.0001) -> str:
    """
    Print an estimated runtime before heavy loops.

    Parameters
    ----------
    n_perm : int
        Number of permutations.
    n_items : int
        Number of items (regions, genes, etc.) per permutation.
    time_per_item_per_perm : float
        Estimated seconds per item per permutation.

    Returns
    -------
    str
        Human-readable estimate.
    """
    total_seconds = n_perm * n_items * time_per_item_per_perm
    if total_seconds < 60:
        estimate = f"{total_seconds:.0f} seconds"
    elif total_seconds < 3600:
        estimate = f"{total_seconds / 60:.1f} minutes"
    else:
        estimate = f"{total_seconds / 3600:.1f} hours"
    msg = f"Estimated runtime: {estimate} ({n_perm} perms × {n_items} items)"
    logger.info(msg)
    return msg


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        logger.info(f"Timer [{self.name}]: {self.elapsed:.2f}s")
