#!/usr/bin/env python3
"""
Verification script for PR #2: Matrix Guardrails & Caching

Run this to verify that all PR #2 components are working correctly.
"""

import sys
import asyncio
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


def test_matrix_limits():
    """Test matrix limit validation."""
    try:
        from src.routing import get_matrix_limits, TravelMode, RoutingPreference
        
        # Test TRANSIT limits
        limits = get_matrix_limits(TravelMode.TRANSIT, RoutingPreference.TRAFFIC_AWARE)
        assert limits.max_elements == 100, f"Expected 100, got {limits.max_elements}"
        
        # Test TRAFFIC_AWARE_OPTIMAL limits
        limits = get_matrix_limits(TravelMode.WALK, RoutingPreference.TRAFFIC_AWARE_OPTIMAL)
        assert limits.max_elements == 100, f"Expected 100, got {limits.max_elements}"
        
        # Test general limits
        limits = get_matrix_limits(TravelMode.DRIVE, RoutingPreference.TRAFFIC_AWARE)
        assert limits.max_elements == 625, f"Expected 625, got {limits.max_elements}"
        
        print("✅ Matrix limit calculation works correctly")
        return True
    except Exception as e:
        print(f"❌ Matrix limit test failed: {e}")
        return False


def test_validation_error_messages():
    """Test that validation provides helpful error messages."""
    try:
        from src.routing import validate_matrix_request, MatrixRequest, Location, TravelMode, RoutingPreference
        
        # Create a request that exceeds TRANSIT limits
        origins = [Location(lat=35.0 + i*0.01, lng=139.0) for i in range(15)]  # 15 origins
        destinations = [Location(lat=35.0, lng=139.0 + i*0.01) for i in range(15)]  # 15 destinations
        
        request = MatrixRequest(
            origins=origins,
            destinations=destinations,
            mode=TravelMode.TRANSIT,
            routing_preference=RoutingPreference.TRAFFIC_AWARE,
        )
        
        try:
            validate_matrix_request(request)
            print("❌ Should have raised ValueError for oversized request")
            return False
        except ValueError as e:
            error_msg = str(e)
            # Check that error message contains helpful suggestions
            if "suggestions" in error_msg.lower() or "💡" in error_msg:
                print("✅ Validation error messages are helpful")
                return True
            else:
                print(f"⚠️  Error message lacks suggestions: {error_msg[:100]}...")
                return False
                
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False


def test_cache_separation():
    """Test that traffic and static caches are separate."""
    try:
        from src.routing.matrix import MatrixCache
        
        cache = MatrixCache()
        stats = cache.stats()
        
        # Check both caches exist
        assert "traffic_cache" in stats, "Missing traffic_cache"
        assert "static_cache" in stats, "Missing static_cache"
        
        # Check they have different TTLs
        traffic_ttl = stats["traffic_cache"]["ttl"]
        static_ttl = stats["static_cache"]["ttl"]
        
        assert traffic_ttl == 300, f"Traffic TTL should be 300s, got {traffic_ttl}"
        assert static_ttl == 3600, f"Static TTL should be 3600s, got {static_ttl}"
        
        print("✅ Dual-TTL cache system works correctly")
        return True
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return False


def test_backoff_with_jitter():
    """Test exponential backoff calculation."""
    try:
        from src.routing.matrix import exponential_backoff_with_jitter
        
        # Test that backoff increases exponentially
        backoff_0 = exponential_backoff_with_jitter(0)
        backoff_1 = exponential_backoff_with_jitter(1)
        backoff_2 = exponential_backoff_with_jitter(2)
        
        assert 0 < backoff_0 <= 2, f"Attempt 0: expected (0, 2], got {backoff_0}"
        assert 0 < backoff_1 <= 4, f"Attempt 1: expected (0, 4], got {backoff_1}"
        assert 0 < backoff_2 <= 6, f"Attempt 2: expected (0, 6], got {backoff_2}"
        
        # Test that it caps at max
        backoff_10 = exponential_backoff_with_jitter(10)
        assert backoff_10 <= 8, f"Should cap at 8s, got {backoff_10}"
        
        print("✅ Exponential backoff with jitter works correctly")
        return True
    except Exception as e:
        print(f"❌ Backoff test failed: {e}")
        return False


async def test_matrix_request_types():
    """Test that matrix request types work correctly."""
    try:
        from src.routing import MatrixRequest, Location, TravelMode, RoutingPreference
        
        # Create a simple request
        request = MatrixRequest(
            origins=[Location(lat=35.6895, lng=139.6917)],
            destinations=[Location(lat=35.6812, lng=139.7671)],
            mode=TravelMode.WALK,
            routing_preference=RoutingPreference.TRAFFIC_AWARE,
            language="en",
        )
        
        assert len(request.origins) == 1
        assert len(request.destinations) == 1
        assert request.mode == TravelMode.WALK
        assert request.routing_preference == RoutingPreference.TRAFFIC_AWARE
        
        print("✅ Matrix request types work correctly")
        return True
    except Exception as e:
        print(f"❌ Request types test failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("PR #2 Verification: Matrix Guardrails & Caching")
    print("=" * 60)
    print()
    
    checks = []
    
    # File structure checks
    print("📁 File Structure Checks:")
    checks.append(check_file_exists("src/routing/__init__.py", "Routing module init"))
    checks.append(check_file_exists("src/routing/matrix.py", "Matrix computation module"))
    print()
    
    # Import checks
    print("📦 Import Checks:")
    checks.append(check_import("src.routing", "Routing package"))
    checks.append(check_import("src.routing.matrix", "Matrix module"))
    print()
    
    # Functional checks
    print("⚙️  Functional Checks:")
    checks.append(test_matrix_limits())
    checks.append(test_validation_error_messages())
    checks.append(test_cache_separation())
    checks.append(test_backoff_with_jitter())
    
    # Async functional checks
    print()
    print("🔄 Async Functional Checks:")
    loop = asyncio.get_event_loop()
    checks.append(loop.run_until_complete(test_matrix_request_types()))
    print()
    
    # Integration check with main file
    print("🔗 Integration Checks:")
    try:
        # Check that geotrip_agent.py can import the new module
        import geotrip_agent
        print("✅ Main agent file imports successfully")
        checks.append(True)
    except Exception as e:
        print(f"❌ Main agent import failed: {e}")
        checks.append(False)
    print()
    
    # Summary
    print("=" * 60)
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"🎉 All checks passed! ({passed}/{total})")
        print()
        print("✅ PR #2 is complete and working correctly!")
        print()
        print("Key improvements:")
        print("  - Enhanced error messages with suggestions")
        print("  - Dual-TTL caching (5min traffic / 60min static)")
        print("  - Exponential backoff with jitter")
        print("  - Modular routing package")
        print()
        print("Next: PR #3 (Scoring Normalization & A/B Harness)")
        return 0
    else:
        print(f"⚠️  Some checks failed: {passed}/{total} passed")
        print()
        print("Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
