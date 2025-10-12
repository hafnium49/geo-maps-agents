"""
Robust HDBSCAN clustering with comprehensive fallback logic.

This module provides:
1. Degenerate case detection (too few points or clusters)
2. Over-clustering detection and correction
3. Deterministic cluster labeling
4. Fallback to score-only selection
5. Comprehensive diagnostics for A/B testing

Key improvements over naive HDBSCAN:
- Handles sparse data gracefully with actionable suggestions
- Prevents over-fragmentation with adaptive min_cluster_size
- Provides quality metrics (silhouette scores, cluster sizes)
- Deterministic labeling using dominant type tokens
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import h3
import hdbscan
from sklearn.metrics import silhouette_score


@dataclass
class ClusteringConfig:
    """Configuration for clustering with fallback logic."""
    
    min_cluster_size: int = 12
    """Minimum points per cluster (HDBSCAN parameter)."""
    
    min_samples: Optional[int] = None
    """Minimum samples for core points (HDBSCAN parameter). None = auto."""
    
    min_clusters: int = 2
    """Minimum clusters required for valid result (< this triggers fallback)."""
    
    max_clusters: int = 10
    """Maximum clusters allowed (> this triggers re-clustering with larger min_size)."""
    
    enable_refitting: bool = True
    """Whether to automatically re-fit when over-clustering detected."""
    
    max_refit_attempts: int = 3
    """Maximum attempts to refit with increased min_cluster_size."""


@dataclass
class ClusterInfo:
    """Information about a single cluster."""
    
    cluster_id: int
    """Cluster ID (-1 for noise)."""
    
    label: str
    """Human-readable label (e.g., 'restaurant + cafe')."""
    
    hex_ids: List[str]
    """H3 hex IDs belonging to this cluster."""
    
    centroid_lat: float
    """Latitude of cluster centroid."""
    
    centroid_lng: float
    """Longitude of cluster centroid."""
    
    size: int = 0
    """Number of points in cluster."""


@dataclass
class ClusteringDiagnostics:
    """Comprehensive diagnostics for clustering quality assessment."""
    
    num_points: int
    """Total number of points provided."""
    
    num_clusters: int
    """Number of clusters found (excluding noise)."""
    
    num_noise: int
    """Number of noise points (cluster_id == -1)."""
    
    cluster_sizes: List[int] = field(default_factory=list)
    """Size of each cluster."""
    
    silhouette_score: Optional[float] = None
    """Silhouette score (higher = better separation, range [-1, 1])."""
    
    fallback_triggered: bool = False
    """Whether fallback to score-only selection was triggered."""
    
    fallback_reason: Optional[str] = None
    """Reason for fallback (if triggered)."""
    
    degenerate_case: bool = False
    """Whether input was too sparse for clustering."""
    
    over_clustering_detected: bool = False
    """Whether too many clusters were detected."""
    
    refit_attempts: int = 0
    """Number of refitting attempts made."""
    
    suggestions: List[str] = field(default_factory=list)
    """Actionable suggestions for improving clustering."""
    
    config_used: Optional[ClusteringConfig] = None
    """Final configuration used (may differ from input if refitted)."""


def label_cluster(
    hex_df: pd.DataFrame,
    cluster_id: int,
    top_n_tokens: int = 2
) -> str:
    """
    Generate deterministic, human-readable label for a cluster.
    
    Uses dominant place types from the hexagons in the cluster.
    Deterministic: same data always produces same label.
    
    Args:
        hex_df: DataFrame with 'cluster' and 'types' columns
        cluster_id: The cluster ID to label
        top_n_tokens: Number of top tokens to include in label
        
    Returns:
        Human-readable label (e.g., "restaurant + cafe" or "Mixed POIs")
    """
    sub = hex_df[hex_df["cluster"] == cluster_id]
    
    if len(sub) == 0:
        return "Empty Cluster"
    
    # Collect all types from all hexes in this cluster
    all_types = []
    for types_str in sub.get("types", pd.Series([])):
        if pd.isna(types_str) or not types_str:
            continue
        all_types.extend(str(types_str).split("|"))
    
    if not all_types:
        return "Mixed POIs"
    
    # Count occurrences and get top N
    type_counts = pd.Series(all_types).value_counts()
    
    # Filter out empty strings and generic tokens
    generic_tokens = {"point_of_interest", "establishment", ""}
    type_counts = type_counts[~type_counts.index.isin(generic_tokens)]
    
    if len(type_counts) == 0:
        return "Mixed POIs"
    
    # Get top N tokens deterministically (already sorted by count, then alphabetically for ties)
    top_tokens = type_counts.head(top_n_tokens).index.tolist()
    
    # Format as readable label
    if len(top_tokens) == 0:
        return "Mixed POIs"
    elif len(top_tokens) == 1:
        return top_tokens[0].replace("_", " ").title()
    else:
        return " + ".join(tok.replace("_", " ") for tok in top_tokens)


def _compute_cluster_quality(
    X: np.ndarray,
    labels: np.ndarray,
    num_clusters: int
) -> Optional[float]:
    """
    Compute silhouette score for cluster quality assessment.
    
    Returns None if quality cannot be computed (e.g., < 2 clusters).
    """
    if num_clusters < 2:
        return None
    
    # Only compute for non-noise points
    mask = labels != -1
    if mask.sum() < 2:
        return None
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = silhouette_score(X[mask], labels[mask])
        return float(score)
    except Exception:
        return None


def _detect_degenerate_case(
    num_points: int,
    config: ClusteringConfig
) -> Tuple[bool, List[str]]:
    """
    Detect if input is too sparse for meaningful clustering.
    
    Returns:
        (is_degenerate, suggestions)
    """
    suggestions = []
    
    # Too few points overall
    if num_points < config.min_cluster_size:
        suggestions.append(
            f"Only {num_points} points provided, need at least {config.min_cluster_size} "
            "for clustering. Consider widening search radius or reducing min_cluster_size."
        )
        return True, suggestions
    
    # Barely enough points (< 2x min_cluster_size)
    if num_points < 2 * config.min_cluster_size:
        suggestions.append(
            f"Only {num_points} points provided, which may be too sparse. "
            "Consider widening search radius for better clustering."
        )
        # Not degenerate, but warn
        return False, suggestions
    
    return False, suggestions


def _handle_over_clustering(
    hex_df: pd.DataFrame,
    config: ClusteringConfig,
    attempt: int
) -> Tuple[pd.DataFrame, List[ClusterInfo], ClusteringDiagnostics]:
    """
    Handle case where too many clusters were detected.
    
    Strategy: Increase min_cluster_size and refit HDBSCAN.
    """
    new_min_size = int(config.min_cluster_size * (1.5 ** attempt))
    
    print(f"⚠️ Over-clustering detected (attempt {attempt}). "
          f"Increasing min_cluster_size: {config.min_cluster_size} → {new_min_size}")
    
    # Create new config with larger min_cluster_size
    new_config = ClusteringConfig(
        min_cluster_size=new_min_size,
        min_samples=config.min_samples,
        min_clusters=config.min_clusters,
        max_clusters=config.max_clusters,
        enable_refitting=config.enable_refitting,
        max_refit_attempts=config.max_refit_attempts,
    )
    
    # Refit with new config
    return cluster_with_fallback(hex_df, new_config, refit_attempt=attempt + 1)


def _fallback_to_scores(
    hex_df: pd.DataFrame,
    reason: str
) -> Tuple[pd.DataFrame, List[ClusterInfo], ClusteringDiagnostics]:
    """
    Fallback to score-only selection (no clustering).
    
    All points assigned cluster_id = -1 (noise).
    """
    print(f"🔄 Clustering fallback triggered: {reason}")
    
    hex_df = hex_df.copy()
    hex_df["cluster"] = -1
    
    diagnostics = ClusteringDiagnostics(
        num_points=len(hex_df),
        num_clusters=0,
        num_noise=len(hex_df),
        cluster_sizes=[],
        silhouette_score=None,
        fallback_triggered=True,
        fallback_reason=reason,
        degenerate_case=True,
        suggestions=[
            "Clustering failed. Using score-only selection.",
            "Consider: (1) widening search radius, (2) reducing min_cluster_size, "
            "(3) using rule-based isochrone corridors instead."
        ]
    )
    
    return hex_df, [], diagnostics


def cluster_with_fallback(
    hex_df: pd.DataFrame,
    config: Optional[ClusteringConfig] = None,
    refit_attempt: int = 0
) -> Tuple[pd.DataFrame, List[ClusterInfo], ClusteringDiagnostics]:
    """
    Robust HDBSCAN clustering with comprehensive fallback logic.
    
    Handles:
    1. Degenerate cases (too few points/clusters)
    2. Over-clustering (too many clusters)
    3. Quality assessment (silhouette scores)
    4. Fallback to score-only selection
    
    Args:
        hex_df: DataFrame with at least 'lat', 'lng', 'hex', 'types' columns
        config: Clustering configuration (uses defaults if None)
        refit_attempt: Internal parameter for recursive refitting
        
    Returns:
        (hex_df_with_clusters, cluster_infos, diagnostics)
        
    The returned hex_df will have a 'cluster' column with cluster IDs.
    Cluster ID -1 indicates noise (unclustered points).
    """
    if config is None:
        config = ClusteringConfig()
    
    num_points = len(hex_df)
    
    # Check for degenerate case
    is_degenerate, suggestions = _detect_degenerate_case(num_points, config)
    if is_degenerate:
        return _fallback_to_scores(hex_df, f"Too few points ({num_points})")
    
    # Extract coordinates for clustering
    if "lat" not in hex_df.columns or "lng" not in hex_df.columns:
        return _fallback_to_scores(hex_df, "Missing 'lat' or 'lng' columns")
    
    X = hex_df[["lat", "lng"]].to_numpy()
    
    # Run HDBSCAN
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples
        )
        labels = clusterer.fit_predict(X)
    except Exception as e:
        return _fallback_to_scores(hex_df, f"HDBSCAN failed: {e}")
    
    # Analyze results
    unique_labels = set(labels)
    num_clusters = len([c for c in unique_labels if c != -1])
    num_noise = (labels == -1).sum()
    
    # Check for too few clusters
    if num_clusters < config.min_clusters:
        reason = f"Too few clusters ({num_clusters} < {config.min_clusters})"
        suggestions.append(
            "Consider: (1) reducing min_cluster_size, (2) widening search radius, "
            "(3) using rule-based isochrone corridors."
        )
        return _fallback_to_scores(hex_df, reason)
    
    # Check for over-clustering
    if num_clusters > config.max_clusters and config.enable_refitting:
        if refit_attempt < config.max_refit_attempts:
            return _handle_over_clustering(hex_df, config, refit_attempt)
        else:
            suggestions.append(
                f"Max refit attempts reached ({config.max_refit_attempts}). "
                "Using fragmented clustering result."
            )
    
    # Clustering succeeded - add labels to dataframe
    hex_df = hex_df.copy()
    hex_df["cluster"] = labels
    
    # Build ClusterInfo objects
    clusters: List[ClusterInfo] = []
    cluster_sizes: List[int] = []
    
    for cid in sorted([c for c in unique_labels if c != -1]):
        sub = hex_df[hex_df["cluster"] == cid]
        label = label_cluster(hex_df, cid)
        
        clusters.append(ClusterInfo(
            cluster_id=cid,
            label=label,
            hex_ids=sub["hex"].tolist() if "hex" in sub.columns else [],
            centroid_lat=float(sub["lat"].mean()),
            centroid_lng=float(sub["lng"].mean()),
            size=len(sub)
        ))
        cluster_sizes.append(len(sub))
    
    # Compute quality metrics
    silhouette = _compute_cluster_quality(X, labels, num_clusters)
    
    # Add quality-based suggestions
    if silhouette is not None:
        if silhouette < 0.2:
            suggestions.append(
                f"Low silhouette score ({silhouette:.3f}). Clusters may be poorly separated. "
                "Consider adjusting min_cluster_size or widening search radius."
            )
        elif silhouette > 0.5:
            suggestions.append(f"Good cluster separation (silhouette={silhouette:.3f}).")
    
    if num_noise > num_points * 0.5:
        suggestions.append(
            f"High noise ratio ({num_noise}/{num_points} = {num_noise/num_points:.1%}). "
            "Consider reducing min_cluster_size."
        )
    
    # Build diagnostics
    diagnostics = ClusteringDiagnostics(
        num_points=num_points,
        num_clusters=num_clusters,
        num_noise=int(num_noise),
        cluster_sizes=cluster_sizes,
        silhouette_score=silhouette,
        fallback_triggered=False,
        fallback_reason=None,
        degenerate_case=False,
        over_clustering_detected=(num_clusters > config.max_clusters),
        refit_attempts=refit_attempt,
        suggestions=suggestions,
        config_used=config,
    )
    
    return hex_df, clusters, diagnostics
