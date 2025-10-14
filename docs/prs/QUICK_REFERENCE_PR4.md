# Quick Reference: PR #4 - HDBSCAN Fallback Logic

## Installation & Setup

No additional dependencies required. Uses existing packages:
- `hdbscan` (spatial clustering)
- `scikit-learn` (silhouette scores)
- `pandas`, `numpy` (data manipulation)

## Quick Start

### Basic Clustering

```python
from src.spatial import cluster_with_fallback, ClusteringConfig

# Default configuration
hex_df_result, clusters, diagnostics = cluster_with_fallback(hex_df)

# Custom configuration
config = ClusteringConfig(
    min_cluster_size=15,
    max_clusters=8,
    enable_refitting=True
)
hex_df_result, clusters, diagnostics = cluster_with_fallback(hex_df, config)
```

### Check Results

```python
if diagnostics.fallback_triggered:
    print(f"⚠️ Fallback: {diagnostics.fallback_reason}")
else:
    print(f"✅ {diagnostics.num_clusters} clusters")
    print(f"Quality: {diagnostics.silhouette_score:.3f}")
```

## API Reference

### cluster_with_fallback()

```python
def cluster_with_fallback(
    hex_df: pd.DataFrame,
    config: Optional[ClusteringConfig] = None,
    refit_attempt: int = 0
) -> Tuple[pd.DataFrame, List[ClusterInfo], ClusteringDiagnostics]
```

**Arguments**:
- `hex_df`: DataFrame with columns: `lat`, `lng`, `hex`, `types`
- `config`: ClusteringConfig or None (uses defaults)
- `refit_attempt`: Internal (don't set manually)

**Returns**:
- `hex_df_result`: Input DataFrame with added `cluster` column
- `clusters`: List of ClusterInfo objects (one per cluster)
- `diagnostics`: ClusteringDiagnostics with quality metrics

**Required DataFrame Columns**:
- `lat`: Latitude (float)
- `lng`: Longitude (float)
- `hex`: H3 hex ID (string)
- `types`: Pipe-separated place types (string, e.g., "restaurant|cafe")

### ClusteringConfig

```python
@dataclass
class ClusteringConfig:
    min_cluster_size: int = 12          # Minimum points per cluster
    min_samples: Optional[int] = None    # HDBSCAN min_samples (auto if None)
    min_clusters: int = 2                # Minimum valid clusters
    max_clusters: int = 10               # Maximum before refitting
    enable_refitting: bool = True        # Auto-correct over-clustering
    max_refit_attempts: int = 3          # Maximum refit iterations
```

**Parameter Tuning**:
- **Sparse data**: Reduce `min_cluster_size` to 8-10
- **Dense data**: Increase to 15-20
- **Urban areas**: `max_clusters=8` (more fine-grained)
- **Rural areas**: `max_clusters=5` (broader regions)

### ClusterInfo

```python
@dataclass
class ClusterInfo:
    cluster_id: int           # Unique ID (-1 = noise)
    label: str                # Human-readable label
    hex_ids: List[str]        # H3 hexagons in cluster
    centroid_lat: float       # Cluster center latitude
    centroid_lng: float       # Cluster center longitude
    size: int                 # Number of points
```

### ClusteringDiagnostics

```python
@dataclass
class ClusteringDiagnostics:
    num_points: int                      # Total input points
    num_clusters: int                    # Clusters found (excluding noise)
    num_noise: int                       # Noise points (cluster_id=-1)
    cluster_sizes: List[int]             # Size of each cluster
    silhouette_score: Optional[float]    # Quality metric [-1, 1]
    fallback_triggered: bool             # True if fallback used
    fallback_reason: Optional[str]       # Why fallback occurred
    degenerate_case: bool                # True if too sparse
    over_clustering_detected: bool       # True if > max_clusters
    refit_attempts: int                  # Refitting iterations
    suggestions: List[str]               # Actionable recommendations
    config_used: Optional[ClusteringConfig]  # Final config
```

### label_cluster()

```python
def label_cluster(
    hex_df: pd.DataFrame,
    cluster_id: int,
    top_n_tokens: int = 2
) -> str
```

**Arguments**:
- `hex_df`: DataFrame with `cluster` and `types` columns
- `cluster_id`: Cluster to label
- `top_n_tokens`: Number of top tokens to include (default: 2)

**Returns**: Human-readable label (e.g., "restaurant + cafe")

**Features**:
- Deterministic (same input → same output)
- Filters generic tokens ("point_of_interest", "establishment")
- Handles empty/missing data gracefully

## Usage Patterns

### Pattern 1: Basic Clustering with Error Handling

```python
from src.spatial import cluster_with_fallback

try:
    hex_df_result, clusters, diagnostics = cluster_with_fallback(hex_df)
    
    if diagnostics.fallback_triggered:
        # Handle fallback case
        print(f"Fallback: {diagnostics.fallback_reason}")
        # Use score-only selection
        top_places = scored_df.nlargest(30, 'score')
    else:
        # Use clustered results
        print(f"Found {diagnostics.num_clusters} clusters")
        for cluster in clusters:
            print(f"  - {cluster.label}: {cluster.size} places")
            
except Exception as e:
    print(f"Clustering error: {e}")
    # Fall back to unclustered selection
```

### Pattern 2: Adaptive Configuration

```python
from src.spatial import cluster_with_fallback, ClusteringConfig

def adaptive_clustering(hex_df, city_type="urban"):
    """Adjust parameters based on city density."""
    
    if city_type == "dense":
        config = ClusteringConfig(
            min_cluster_size=18,
            max_clusters=12
        )
    elif city_type == "suburban":
        config = ClusteringConfig(
            min_cluster_size=12,
            max_clusters=8
        )
    else:  # rural
        config = ClusteringConfig(
            min_cluster_size=8,
            max_clusters=5
        )
    
    return cluster_with_fallback(hex_df, config)
```

### Pattern 3: Quality Assessment

```python
hex_df_result, clusters, diagnostics = cluster_with_fallback(hex_df)

# Assess clustering quality
if diagnostics.silhouette_score is not None:
    if diagnostics.silhouette_score > 0.5:
        quality = "excellent"
    elif diagnostics.silhouette_score > 0.2:
        quality = "acceptable"
    else:
        quality = "poor"
    
    print(f"Clustering quality: {quality} (score={diagnostics.silhouette_score:.3f})")
    
# Check noise ratio
noise_ratio = diagnostics.num_noise / diagnostics.num_points
if noise_ratio > 0.3:
    print(f"High noise: {noise_ratio:.1%} of points unclustered")
```

### Pattern 4: Integration with geotrip_agent.py

```python
# In spatial_context_and_scoring()
hex_df2, clusters, diagnostics = _hdbscan_clusters(hex_df, cfg.cluster_min_size)

# Diagnostics automatically logged:
# ✅ Clustering Success:
#   Points: 60
#   Clusters: 3
#   Noise: 5
#   Quality (silhouette): 0.782

# Use clusters for neighborhood-aware routing
for cluster in clusters:
    print(f"Cluster '{cluster.label}' at ({cluster.centroid_lat:.4f}, {cluster.centroid_lng:.4f})")
```

## Common Scenarios

### Scenario: Too Few Points

**Problem**: Only 5 places found in search area

**Behavior**:
```python
diagnostics.fallback_triggered = True
diagnostics.fallback_reason = "Too few points (5)"
diagnostics.suggestions = [
    "Only 5 points provided, need at least 12 for clustering. "
    "Consider widening search radius or reducing min_cluster_size."
]
```

**Solution**:
- Increase search radius (e.g., 4000m → 6000m)
- Reduce `min_cluster_size` (e.g., 12 → 8)
- Accept score-only selection

### Scenario: Over-Clustering

**Problem**: HDBSCAN produces 15 tiny clusters

**Behavior**:
```
⚠️ Over-clustering detected (attempt 1). Increasing min_cluster_size: 12 → 18
✅ Clustering Success:
  Points: 100
  Clusters: 7
  Refit attempts: 1
```

**Automatic correction**: Increases `min_cluster_size` by 1.5x and refits

### Scenario: Poor Separation

**Problem**: Clusters overlap significantly

**Behavior**:
```python
diagnostics.silhouette_score = 0.15
diagnostics.suggestions = [
    "Low silhouette score (0.150). Clusters may be poorly separated. "
    "Consider adjusting min_cluster_size or widening search radius."
]
```

**Solution**:
- Increase `min_cluster_size` to merge weak clusters
- Widen search radius to get more spatial diversity
- Accept result if neighborhoods are naturally mixed

## Diagnostics Interpretation

### Silhouette Score

| Score | Quality | Action |
|-------|---------|--------|
| > 0.5 | Excellent | Use clusters confidently |
| 0.2 - 0.5 | Acceptable | Monitor but OK to use |
| < 0.2 | Poor | Consider refitting or fallback |

### Noise Ratio

| Ratio | Status | Action |
|-------|--------|--------|
| < 20% | Normal | Acceptable |
| 20-40% | Moderate | Monitor cluster sizes |
| > 40% | High | Reduce min_cluster_size |

### Cluster Count

| Count | Status | Action |
|-------|--------|--------|
| 2-5 | Ideal | Good neighborhood separation |
| 6-10 | Acceptable | May be over-fragmented |
| > 10 | Over-clustered | Auto-refits if enabled |
| < 2 | Under-clustered | Triggers fallback |

## Testing

### Run Verification

```bash
uv run verify_pr4.py
```

**Expected Output**:
```
✅ File Structure
✅ Imports
✅ Degenerate Case Handling
✅ Over-Clustering Detection
✅ Successful Clustering
✅ Label Determinism
✅ Integration

Passed: 7/7
🎉 All checks passed!
```

### Unit Test Example

```python
import pandas as pd
import numpy as np
from src.spatial import cluster_with_fallback, ClusteringConfig

def test_sparse_data():
    """Test fallback on sparse data."""
    sparse_df = pd.DataFrame({
        "lat": [35.6895, 35.6905],
        "lng": [139.6917, 139.6927],
        "hex": ["hex1", "hex2"],
        "types": ["restaurant", "cafe"],
    })
    
    config = ClusteringConfig(min_cluster_size=12)
    _, _, diagnostics = cluster_with_fallback(sparse_df, config)
    
    assert diagnostics.fallback_triggered
    assert "Too few points" in diagnostics.fallback_reason
    assert len(diagnostics.suggestions) > 0
```

## Troubleshooting

### Issue: "Missing 'lat' or 'lng' columns"

**Cause**: Input DataFrame doesn't have required columns

**Solution**: Ensure DataFrame has `lat`, `lng`, `hex`, `types` columns

### Issue: All points assigned cluster_id = -1

**Cause**: Clustering failed or fallback triggered

**Check**: `diagnostics.fallback_reason` for explanation

### Issue: Silhouette score is None

**Cause**: < 2 clusters or all points are noise

**Interpretation**: Normal for degenerate cases

### Issue: ImportError on sklearn

**Cause**: scikit-learn not installed

**Solution**: 
```bash
uv pip install scikit-learn
```

## Performance Tips

1. **Large datasets (1000+ points)**: Consider pre-filtering by score
2. **Real-time applications**: Set `max_refit_attempts=1` to limit retries
3. **Memory constraints**: Process clusters in batches
4. **CPU-bound**: Run clustering in background thread

## Next Steps

After PR #4:
- **PR #5**: OR-Tools VRPTW will use cluster centroids for multi-neighborhood routing
- **PR #6**: CI/CD will add automated cluster quality tests

## Resources

- **HDBSCAN docs**: https://hdbscan.readthedocs.io/
- **Silhouette score**: https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
- **PR #4 summary**: See `PR4_SUMMARY.md`

## Support

For issues or questions:
1. Check `diagnostics.suggestions` for actionable recommendations
2. Review fallback_reason if clustering failed
3. Adjust ClusteringConfig parameters
4. Consider widening search radius or reducing min_cluster_size
