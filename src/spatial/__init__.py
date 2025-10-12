"""
src/spatial: Spatial analysis, clustering, and geographic utilities.

This module provides robust HDBSCAN clustering with comprehensive fallback logic.
"""

from .clustering import (
    ClusteringConfig,
    ClusteringDiagnostics,
    ClusterInfo,
    cluster_with_fallback,
    label_cluster,
)

__all__ = [
    "ClusteringConfig",
    "ClusteringDiagnostics",
    "ClusterInfo",
    "cluster_with_fallback",
    "label_cluster",
]
