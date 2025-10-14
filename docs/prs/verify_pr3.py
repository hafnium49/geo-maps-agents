#!/usr/bin/env python3
"""
Verification script for PR #3: Scoring Normalization & A/B Harness

Run this to verify that all PR #3 components are working correctly.
"""

import sys
import os
from pathlib import Path


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists and print result."""
    path = Path(filepath)
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists


def check_import(module_path: str, description: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_path)
        print(f"✅ {description}: {module_path}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_path} - {e}")
        return False


def test_percentile_normalization():
    """Test that percentile-based normalization works correctly."""
    try:
        import numpy as np
        from src.scoring import percentile_norm, normalize_eta
        
        # Test basic normalization
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # outlier at 100
        normalized = percentile_norm(values)
        
        # Values should be normalized to [0, 1] range
        assert np.all((normalized >= 0) & (normalized <= 1)), "Values not in [0, 1] range"
        
        # Test ETA inversion (lower ETA = higher score)
        etas = np.array([300, 600, 900, 1200])
        normalized_etas = normalize_eta(etas)
        
        # Lower ETA should get higher score
        assert normalized_etas[0] > normalized_etas[-1], "ETA not properly inverted"
        
        print("✅ Percentile-based normalization works correctly")
        return True
    except Exception as e:
        print(f"❌ Normalization test failed: {e}")
        return False


def test_weight_variants():
    """Test that weight variants are defined correctly."""
    try:
        from src.scoring import (
            DEFAULT_WEIGHTS,
            VARIANT_A,
            VARIANT_B,
            VARIANT_C,
            WEIGHT_VARIANTS,
        )
        
        # Check all variants exist
        assert "default" in WEIGHT_VARIANTS, "Missing default variant"
        assert "variant-a" in WEIGHT_VARIANTS, "Missing variant-a"
        assert "variant-b" in WEIGHT_VARIANTS, "Missing variant-b"
        assert "variant-c" in WEIGHT_VARIANTS, "Missing variant-c"
        
        # Check variant differentiation
        assert VARIANT_A.w_rating != VARIANT_B.w_rating or \
               VARIANT_A.w_diversity != VARIANT_B.w_diversity, \
               "Variants A and B are identical"
        
        # Check variant names are set
        assert DEFAULT_WEIGHTS.variant_name == "default", "Default variant name not set"
        assert VARIANT_A.variant_name == "variant-a", "Variant A name not set"
        
        print("✅ Weight variants are defined correctly")
        return True
    except Exception as e:
        print(f"❌ Weight variants test failed: {e}")
        return False


def test_ab_variant_selection():
    """Test session-sticky A/B variant selection."""
    try:
        from src.scoring import select_ab_variant
        
        # Test deterministic selection
        variant1 = select_ab_variant(user_id="user_123")
        variant2 = select_ab_variant(user_id="user_123")
        
        assert variant1.variant_name == variant2.variant_name, \
            "Same user should get same variant (session sticky)"
        
        # Test different users might get different variants
        # (not guaranteed, but very likely with SHA256)
        variants = set()
        for i in range(10):
            v = select_ab_variant(user_id=f"user_{i}")
            variants.add(v.variant_name)
        
        # Should have at least 2 different variants in 10 users
        assert len(variants) >= 2, "A/B selection not distributing variants"
        
        print("✅ Session-sticky A/B variant selection works correctly")
        return True
    except Exception as e:
        print(f"❌ A/B variant selection test failed: {e}")
        return False


def test_weights_yaml():
    """Test that weights.yaml exists and can be loaded."""
    try:
        from src.scoring import load_weights_from_yaml
        import yaml
        
        yaml_path = Path("configs/weights.yaml")
        
        if not yaml_path.exists():
            print("⚠️  weights.yaml not found - using built-in defaults")
            return True
        
        # Load variants from YAML
        variants = load_weights_from_yaml()
        
        # Check that at least default and 3 A/B variants exist
        assert "default" in variants, "Missing default in weights.yaml"
        assert len(variants) >= 4, f"Expected at least 4 variants, got {len(variants)}"
        
        print("✅ weights.yaml exists and loads correctly")
        return True
    except Exception as e:
        print(f"❌ weights.yaml test failed: {e}")
        return False


def test_telemetry_logging():
    """Test that telemetry is logged correctly."""
    try:
        import numpy as np
        import pandas as pd
        from src.scoring import PlaceScorer, WeightConfig
        
        # Create minimal test data
        places_df = pd.DataFrame({
            "id": ["place1", "place2"],
            "name": ["Place 1", "Place 2"],
            "lat": [35.6895, 35.6812],
            "lng": [139.6917, 139.7671],
            "rating": [4.5, 4.0],
            "nratings": [100, 50],
            "open_now": [1, 0],
            "primary": ["restaurant", "cafe"],
            "types": ["restaurant|food", "cafe|food"],
        })
        
        etas = {"place1": 300, "place2": 600}
        
        # Create minimal hex_df
        hex_df = pd.DataFrame({
            "hex": ["hex1"],
            "localness": [0.5],
            "lat": [35.6895],
            "lng": [139.6917],
            "cluster": [-1],
        })
        hex_df.attrs["h3_res"] = 9
        
        # Score with telemetry
        scorer = PlaceScorer(enable_telemetry=True)
        scored_df = scorer.score_places(places_df, etas, hex_df)
        
        # Check telemetry was collected
        telemetry = scorer.get_telemetry()
        assert len(telemetry) == 2, f"Expected 2 telemetry entries, got {len(telemetry)}"
        
        # Check telemetry structure
        t = telemetry[0]
        assert hasattr(t, 'place_id'), "Missing place_id in telemetry"
        assert hasattr(t, 'breakdown'), "Missing breakdown in telemetry"
        assert hasattr(t.breakdown, 'final_score'), "Missing final_score in breakdown"
        assert hasattr(t, 'variant_name'), "Missing variant_name in telemetry"
        
        print("✅ Telemetry logging works correctly")
        return True
    except Exception as e:
        print(f"❌ Telemetry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_geotrip_agent():
    """Test that geotrip_agent.py can import and use new scoring."""
    try:
        # This will fail if geotrip_agent.py has import errors
        import geotrip_agent
        
        # Check that WeightConfig has variant_name
        config = geotrip_agent.WeightConfig()
        assert hasattr(config, 'variant_name'), "WeightConfig missing variant_name"
        
        print("✅ Integration with geotrip_agent.py successful")
        return True
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("PR #3 Verification: Scoring Normalization & A/B Harness")
    print("=" * 60)
    print()
    
    checks = []
    
    # File structure checks
    print("📁 File Structure Checks:")
    checks.append(check_file_exists("src/scoring/__init__.py", "Scoring module init"))
    checks.append(check_file_exists("src/scoring/normalization.py", "Normalization module"))
    checks.append(check_file_exists("src/scoring/weights.py", "Weights module"))
    checks.append(check_file_exists("src/scoring/scorer.py", "Scorer module"))
    checks.append(check_file_exists("configs/weights.yaml", "Weight configurations"))
    print()
    
    # Import checks
    print("📦 Import Checks:")
    checks.append(check_import("src.scoring", "Scoring package"))
    checks.append(check_import("src.scoring.normalization", "Normalization module"))
    checks.append(check_import("src.scoring.weights", "Weights module"))
    checks.append(check_import("src.scoring.scorer", "Scorer module"))
    print()
    
    # Functional checks
    print("⚙️  Functional Checks:")
    checks.append(test_percentile_normalization())
    checks.append(test_weight_variants())
    checks.append(test_ab_variant_selection())
    checks.append(test_weights_yaml())
    checks.append(test_telemetry_logging())
    print()
    
    # Integration check
    print("🔗 Integration Checks:")
    checks.append(test_integration_with_geotrip_agent())
    print()
    
    # Summary
    print("=" * 60)
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"🎉 All checks passed! ({passed}/{total})")
        print()
        print("✅ PR #3 is complete and working correctly!")
        print()
        print("Key improvements:")
        print("  - Percentile-based normalization (5th/95th)")
        print("  - Proper ETA inversion (lower ETA = higher score)")
        print("  - A/B testing with session-sticky variants")
        print("  - Detailed per-stop telemetry logging")
        print("  - Modular src/scoring/ package")
        print()
        print("Next: PR #4 (HDBSCAN Fallback Logic)")
        return 0
    else:
        print(f"⚠️  Some checks failed: {passed}/{total} passed")
        print()
        print("Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
