"""
Scoring Normalization Module

Provides robust percentile-based normalization for scoring metrics.
Uses 5th/95th percentiles to handle outliers effectively.
"""

from typing import Optional
import numpy as np


def percentile_norm(
    series: np.ndarray,
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
    invert: bool = False,
) -> np.ndarray:
    """
    Percentile-based normalization to [0, 1] range.
    
    Uses percentiles instead of min/max to handle outliers robustly.
    For ETA and other "lower is better" metrics, set invert=True.
    
    Args:
        series: Input array to normalize
        low_percentile: Lower percentile (default 5th)
        high_percentile: Upper percentile (default 95th)
        invert: If True, inverts the normalization (for "lower is better" metrics)
    
    Returns:
        Normalized array in [0, 1] range
    
    Examples:
        >>> # Rating normalization (higher is better)
        >>> ratings = np.array([3.5, 4.0, 4.5, 5.0, 2.0, 4.8])
        >>> normalized = percentile_norm(ratings)
        
        >>> # ETA normalization (lower is better)
        >>> etas = np.array([300, 600, 900, 1200, 450])
        >>> normalized = percentile_norm(etas, invert=True)
    """
    # Handle empty or all-NaN arrays
    if len(series) == 0 or np.all(np.isnan(series)):
        return np.zeros_like(series, dtype=float)
    
    # Calculate percentiles (ignoring NaN values)
    lo = np.nanpercentile(series, low_percentile)
    hi = np.nanpercentile(series, high_percentile)
    
    # Handle degenerate case where all values are the same
    if hi - lo <= 1e-9:
        return np.zeros_like(series, dtype=float)
    
    # Normalize to [0, 1]
    normalized = (series - lo) / (hi - lo)
    
    # Clip to [0, 1] range (values outside percentiles)
    normalized = np.clip(normalized, 0, 1)
    
    # Invert if requested (for "lower is better" metrics like ETA)
    if invert:
        normalized = 1.0 - normalized
    
    return normalized


def robust_norm(series: np.ndarray) -> np.ndarray:
    """
    Legacy wrapper for backward compatibility.
    
    DEPRECATED: Use percentile_norm() instead for clearer semantics.
    This function maintains the old behavior but uses percentile-based
    normalization under the hood.
    
    Args:
        series: Input array to normalize
    
    Returns:
        Normalized array in [0, 1] range
    """
    return percentile_norm(series, low_percentile=5.0, high_percentile=95.0, invert=False)


def normalize_rating(ratings: np.ndarray) -> np.ndarray:
    """
    Normalize rating scores (higher is better).
    
    Uses 5th/95th percentiles to handle outliers.
    
    Args:
        ratings: Array of rating values (typically 0-5 scale)
    
    Returns:
        Normalized ratings in [0, 1] range
    """
    return percentile_norm(ratings, low_percentile=5.0, high_percentile=95.0, invert=False)


def normalize_eta(etas: np.ndarray, min_eta_sec: int = 180) -> np.ndarray:
    """
    Normalize ETA (travel time) scores - lower is better.
    
    Applies minimum ETA clamp, then inverts normalization so that
    shorter travel times get higher scores.
    
    Args:
        etas: Array of ETA values in seconds
        min_eta_sec: Minimum ETA to clamp to (default 180s = 3 min)
    
    Returns:
        Normalized inverted ETAs in [0, 1] range (lower ETA = higher score)
    """
    # Clamp minimum ETA for foot/indoor routing
    clamped = np.maximum(etas, min_eta_sec)
    
    # Invert: lower ETA = higher score
    return percentile_norm(clamped, low_percentile=5.0, high_percentile=95.0, invert=True)


def normalize_crowd_proxy(crowd_values: np.ndarray) -> np.ndarray:
    """
    Normalize crowd proxy scores (typically based on review counts).
    
    Higher values indicate more crowded places. The normalization is
    straightforward without inversion.
    
    Args:
        crowd_values: Array of crowd proxy values (e.g., review counts)
    
    Returns:
        Normalized crowd scores in [0, 1] range
    """
    return percentile_norm(crowd_values, low_percentile=5.0, high_percentile=95.0, invert=False)


def normalize_diversity(diversity_values: np.ndarray) -> np.ndarray:
    """
    Normalize diversity gain scores (higher is better).
    
    Diversity represents how unique/rare a place type is relative to
    what's already been visited or what's common in the area.
    
    Args:
        diversity_values: Array of diversity gain values
    
    Returns:
        Normalized diversity scores in [0, 1] range
    """
    return percentile_norm(diversity_values, low_percentile=5.0, high_percentile=95.0, invert=False)


def normalize_localness(localness_values: np.ndarray) -> np.ndarray:
    """
    Normalize localness scores (higher is better).
    
    Localness represents how "local" vs "touristy" a place is,
    based on hex-level aggregated ratings and tourist anchor density.
    
    Args:
        localness_values: Array of localness values
    
    Returns:
        Normalized localness scores in [0, 1] range
    """
    return percentile_norm(localness_values, low_percentile=5.0, high_percentile=95.0, invert=False)
