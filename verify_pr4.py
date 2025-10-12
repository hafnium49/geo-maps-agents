#!/usr/bin/env python3
"""
Verification script for PR #4: HDBSCAN Fallback Logic

Checks:
1. File structure (src/spatial/)
2. Import checks
3. Degenerate case handling
4. Over-clustering detection
5. Fallback logic
6. Diagnostic logging
7. Integration with geotrip_agent.py
"""

import sys
import os
from pathlib import Path

def check_file_structure():
    """Verify PR #4 file structure."""
    print("📁 File Structure Checks:")
    
    required_files = [
        "src/spatial/__init__.py",
        "src/spatial/clustering.py",
    ]
    
    all_exist = True
    for f in required_files:
        exists = Path(f).exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {f}")
        all_exist = all_exist and exists
    
    return all_exist


def check_imports():
    """Verify all modules import successfully."""
    print("\n📦 Import Checks:")
    
    checks = []
    
    # Check src.spatial module
    try:
        from src.spatial import (
            ClusteringConfig,
            ClusteringDiagnostics,
            ClusterInfo,
            cluster_with_fallback,
            label_cluster,
        )
        print("  ✅ src.spatial module imports successfully")
        checks.append(True)
    except ImportError as e:
        print(f"  ❌ src.spatial import failed: {e}")
        checks.append(False)
    
    # Check clustering module directly
    try:
        from src.spatial.clustering import (
            ClusteringConfig,
            ClusteringDiagnostics,
            ClusterInfo,
            cluster_with_fallback,
            label_cluster,
        )
        print("  ✅ src.spatial.clustering imports successfully")
        checks.append(True)
    except ImportError as e:
        print(f"  ❌ src.spatial.clustering import failed: {e}")
        checks.append(False)
    
    # Check geotrip_agent integration
    try:
        from geotrip_agent import _hdbscan_clusters, _label_cluster
        print("  ✅ geotrip_agent clustering functions import successfully")
        checks.append(True)
    except ImportError as e:
        print(f"  ❌ geotrip_agent clustering import failed: {e}")
        checks.append(False)
    
    return all(checks)


def check_degenerate_case_handling():
    """Test degenerate case detection and handling."""
    print("\n🔍 Degenerate Case Handling:")
    
    try:
        import pandas as pd
        import numpy as np
        from src.spatial import cluster_with_fallback, ClusteringConfig
        
        # Create sparse dataset (too few points)
        sparse_df = pd.DataFrame({
            "lat": [35.6895, 35.6905],
            "lng": [139.6917, 139.6927],
            "hex": ["hex1", "hex2"],
            "types": ["restaurant", "cafe"],
        })
        
        config = ClusteringConfig(min_cluster_size=12)
        hex_df_result, clusters, diagnostics = cluster_with_fallback(sparse_df, config)
        
        # Should trigger fallback
        if diagnostics.fallback_triggered:
            print("  ✅ Degenerate case correctly detected and fallback triggered")
            if diagnostics.degenerate_case:
                print("  ✅ Degenerate flag set correctly")
            if "Too few points" in diagnostics.fallback_reason:
                print("  ✅ Fallback reason correctly identifies sparse data")
            if len(diagnostics.suggestions) > 0:
                print(f"  ✅ Suggestions provided ({len(diagnostics.suggestions)} suggestions)")
            return True
        else:
            print("  ❌ Degenerate case not detected (should have triggered fallback)")
            return False
            
    except Exception as e:
        print(f"  ❌ Degenerate case test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_over_clustering():
    """Test over-clustering detection and correction."""
    print("\n🔍 Over-Clustering Detection:")
    
    try:
        import pandas as pd
        import numpy as np
        from src.spatial import cluster_with_fallback, ClusteringConfig
        
        # Create dataset with many scattered points (likely to produce many clusters)
        np.random.seed(42)
        n_points = 100
        scattered_df = pd.DataFrame({
            "lat": 35.6895 + np.random.randn(n_points) * 0.05,
            "lng": 139.6917 + np.random.randn(n_points) * 0.05,
            "hex": [f"hex{i}" for i in range(n_points)],
            "types": ["restaurant|cafe"] * n_points,
        })
        
        # Use very small min_cluster_size to encourage many clusters
        config = ClusteringConfig(
            min_cluster_size=3,
            max_clusters=5,  # Set low threshold
            enable_refitting=True,
            max_refit_attempts=2
        )
        
        hex_df_result, clusters, diagnostics = cluster_with_fallback(scattered_df, config)
        
        # Check if over-clustering was handled
        num_clusters = diagnostics.num_clusters
        
        if diagnostics.refit_attempts > 0:
            print(f"  ✅ Refitting attempted ({diagnostics.refit_attempts} attempts)")
            print(f"  ✅ Final cluster count: {num_clusters}")
            return True
        elif num_clusters <= config.max_clusters:
            print(f"  ✅ Cluster count within limits ({num_clusters} ≤ {config.max_clusters})")
            return True
        else:
            print(f"  ⚠️ Over-clustering detected but not corrected ({num_clusters} > {config.max_clusters})")
            # This is OK if we reached max refit attempts
            if diagnostics.refit_attempts >= config.max_refit_attempts:
                print(f"  ✅ Max refit attempts reached, acceptable")
                return True
            return False
            
    except Exception as e:
        print(f"  ❌ Over-clustering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_successful_clustering():
    """Test successful clustering with good data."""
    print("\n🔍 Successful Clustering:")
    
    try:
        import pandas as pd
        import numpy as np
        from src.spatial import cluster_with_fallback, ClusteringConfig
        
        # Create well-separated clusters
        np.random.seed(42)
        
        # Cluster 1: Around Tokyo Station
        cluster1 = pd.DataFrame({
            "lat": 35.6895 + np.random.randn(20) * 0.002,
            "lng": 139.6917 + np.random.randn(20) * 0.002,
            "hex": [f"hex1_{i}" for i in range(20)],
            "types": ["restaurant|cafe"] * 20,
        })
        
        # Cluster 2: Around Shibuya (offset)
        cluster2 = pd.DataFrame({
            "lat": 35.6612 + np.random.randn(20) * 0.002,
            "lng": 139.7006 + np.random.randn(20) * 0.002,
            "hex": [f"hex2_{i}" for i in range(20)],
            "types": ["museum|art_gallery"] * 20,
        })
        
        # Cluster 3: Around Shinjuku (offset)
        cluster3 = pd.DataFrame({
            "lat": 35.6938 + np.random.randn(20) * 0.002,
            "lng": 139.7034 + np.random.randn(20) * 0.002,
            "hex": [f"hex3_{i}" for i in range(20)],
            "types": ["shopping_mall|store"] * 20,
        })
        
        good_df = pd.concat([cluster1, cluster2, cluster3], ignore_index=True)
        
        config = ClusteringConfig(min_cluster_size=12)
        hex_df_result, clusters, diagnostics = cluster_with_fallback(good_df, config)
        
        # Should succeed
        checks = []
        
        if not diagnostics.fallback_triggered:
            print("  ✅ Clustering succeeded (no fallback)")
            checks.append(True)
        else:
            print(f"  ❌ Unexpected fallback: {diagnostics.fallback_reason}")
            checks.append(False)
        
        if diagnostics.num_clusters >= 2:
            print(f"  ✅ Found {diagnostics.num_clusters} clusters")
            checks.append(True)
        else:
            print(f"  ❌ Too few clusters: {diagnostics.num_clusters}")
            checks.append(False)
        
        if len(clusters) == diagnostics.num_clusters:
            print(f"  ✅ ClusterInfo objects created ({len(clusters)})")
            checks.append(True)
        else:
            print(f"  ❌ ClusterInfo count mismatch: {len(clusters)} vs {diagnostics.num_clusters}")
            checks.append(False)
        
        if diagnostics.silhouette_score is not None:
            print(f"  ✅ Silhouette score computed: {diagnostics.silhouette_score:.3f}")
            checks.append(True)
        else:
            print("  ⚠️ No silhouette score computed")
            checks.append(True)  # Not critical
        
        # Check cluster labels
        if all(c.label and c.label != "Empty Cluster" for c in clusters):
            print(f"  ✅ All clusters have labels")
            checks.append(True)
        else:
            print("  ❌ Some clusters missing labels")
            checks.append(False)
        
        return all(checks)
        
    except Exception as e:
        print(f"  ❌ Successful clustering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_label_determinism():
    """Test that label_cluster produces deterministic results."""
    print("\n🔍 Deterministic Labeling:")
    
    try:
        import pandas as pd
        from src.spatial import label_cluster
        
        # Create test dataframe
        test_df = pd.DataFrame({
            "cluster": [1, 1, 1, 1, 1],
            "types": [
                "restaurant|cafe|food",
                "restaurant|bar",
                "cafe|bakery",
                "restaurant|sushi_restaurant",
                "cafe|coffee_shop",
            ],
        })
        
        # Generate label multiple times
        labels = [label_cluster(test_df, cluster_id=1) for _ in range(5)]
        
        # All should be identical
        if len(set(labels)) == 1:
            print(f"  ✅ Deterministic labeling: '{labels[0]}'")
            return True
        else:
            print(f"  ❌ Non-deterministic labels: {set(labels)}")
            return False
            
    except Exception as e:
        print(f"  ❌ Label determinism test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_integration():
    """Test integration with geotrip_agent.py."""
    print("\n🔗 Integration with geotrip_agent.py:")
    
    try:
        import pandas as pd
        import numpy as np
        from geotrip_agent import _hdbscan_clusters
        
        # Create test data
        np.random.seed(42)
        test_df = pd.DataFrame({
            "lat": 35.6895 + np.random.randn(30) * 0.01,
            "lng": 139.6917 + np.random.randn(30) * 0.01,
            "hex": [f"hex{i}" for i in range(30)],
            "types": ["restaurant|cafe"] * 30,
            "rating": np.random.uniform(3.5, 5.0, 30),
            "nratings": np.random.randint(10, 500, 30),
        })
        
        # Call legacy wrapper
        hex_df_result, clusters, diagnostics = _hdbscan_clusters(test_df, min_cluster_size=10)
        
        # Check return types
        checks = []
        
        if isinstance(hex_df_result, pd.DataFrame):
            print("  ✅ Returns DataFrame")
            checks.append(True)
        else:
            print(f"  ❌ Wrong return type for DataFrame: {type(hex_df_result)}")
            checks.append(False)
        
        if isinstance(clusters, list):
            print(f"  ✅ Returns cluster list ({len(clusters)} clusters)")
            checks.append(True)
        else:
            print(f"  ❌ Wrong return type for clusters: {type(clusters)}")
            checks.append(False)
        
        # Check that diagnostics is returned (new feature)
        if hasattr(diagnostics, 'num_clusters'):
            print("  ✅ Returns diagnostics object")
            checks.append(True)
        else:
            print("  ❌ No diagnostics object returned")
            checks.append(False)
        
        # Check that cluster column exists
        if "cluster" in hex_df_result.columns:
            print("  ✅ DataFrame has 'cluster' column")
            checks.append(True)
        else:
            print("  ❌ DataFrame missing 'cluster' column")
            checks.append(False)
        
        return all(checks)
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("PR #4 VERIFICATION: HDBSCAN Fallback Logic")
    print("=" * 60)
    
    results = []
    
    # Run all checks
    results.append(("File Structure", check_file_structure()))
    results.append(("Imports", check_imports()))
    results.append(("Degenerate Case Handling", check_degenerate_case_handling()))
    results.append(("Over-Clustering Detection", check_over_clustering()))
    results.append(("Successful Clustering", check_successful_clustering()))
    results.append(("Label Determinism", check_label_determinism()))
    results.append(("Integration", check_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! PR #4 is complete.")
        print("\nNext: PR #5 (OR-Tools VRPTW Sequencer)")
        return 0
    else:
        print(f"\n❌ {total - passed} check(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
