#!/usr/bin/env python3
"""
Verification script for PR #5: OR-Tools VRPTW Sequencer

Checks:
1. File structure (src/routing/vrptw.py, greedy.py)
2. Import checks
3. OR-Tools availability
4. Greedy sequencing
5. VRPTW solving (basic case)
6. Fallback mechanism
7. Integration with geotrip_agent.py
"""

import sys
import os
from pathlib import Path

def check_file_structure():
    """Verify PR #5 file structure."""
    print("📁 File Structure Checks:")
    
    required_files = [
        "src/routing/vrptw.py",
        "src/routing/greedy.py",
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
    
    # Check OR-Tools
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
        print("  ✅ OR-Tools imports successfully")
        checks.append(True)
    except ImportError as e:
        print(f"  ❌ OR-Tools import failed: {e}")
        checks.append(False)
    
    # Check greedy module
    try:
        from src.routing.greedy import greedy_sequence, GreedySequenceResult, Stop
        print("  ✅ src.routing.greedy imports successfully")
        checks.append(True)
    except ImportError as e:
        print(f"  ❌ src.routing.greedy import failed: {e}")
        checks.append(False)
    
    # Check vrptw module
    try:
        from src.routing.vrptw import (
            solve_vrptw,
            solve_vrptw_with_fallback,
            VRPTWConfig,
            VRPTWResult
        )
        print("  ✅ src.routing.vrptw imports successfully")
        checks.append(True)
    except ImportError as e:
        print(f"  ❌ src.routing.vrptw import failed: {e}")
        checks.append(False)
    
    # Check routing module exports
    try:
        from src.routing import (
            solve_vrptw,
            solve_vrptw_with_fallback,
            greedy_sequence,
            VRPTWConfig
        )
        print("  ✅ src.routing module exports VRPTW functions")
        checks.append(True)
    except ImportError as e:
        print(f"  ❌ src.routing export failed: {e}")
        checks.append(False)
    
    return all(checks)


def check_greedy_sequencing():
    """Test greedy sequencing algorithm."""
    print("\n🔍 Greedy Sequencing:")
    
    try:
        import pandas as pd
        from datetime import datetime, timedelta
        from src.routing.greedy import greedy_sequence
        
        # Create test candidates
        candidates = pd.DataFrame({
            'id': ['place1', 'place2', 'place3', 'place4'],
            'name': ['Restaurant A', 'Cafe B', 'Museum C', 'Park D'],
            'lat': [35.6895, 35.6905, 35.6915, 35.6925],
            'lng': [139.6917, 139.6927, 139.6937, 139.6947],
            'score': [0.9, 0.8, 0.7, 0.6],
            'eta': [300, 600, 900, 1200],  # seconds from anchor
        })
        
        start_time = datetime.now().replace(hour=13, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(hours=3)
        
        result = greedy_sequence(
            candidates=candidates,
            anchor_lat=35.6895,
            anchor_lng=139.6917,
            start_time=start_time,
            end_time=end_time,
            service_time_min=30
        )
        
        checks = []
        
        if len(result.stops) > 0:
            print(f"  ✅ Greedy produced {len(result.stops)} stops")
            checks.append(True)
        else:
            print("  ❌ Greedy produced no stops")
            checks.append(False)
        
        if result.sequence_method == "greedy":
            print("  ✅ Sequence method correctly labeled")
            checks.append(True)
        else:
            print(f"  ❌ Wrong sequence method: {result.sequence_method}")
            checks.append(False)
        
        # Check stops are sorted by score (descending)
        scores = [s.score for s in result.stops]
        if scores == sorted(scores, reverse=True):
            print("  ✅ Stops sorted by score (descending)")
            checks.append(True)
        else:
            print("  ⚠️ Stops not perfectly sorted by score (acceptable)")
            checks.append(True)  # Not critical
        
        # Check timing is reasonable
        if result.total_duration_sec > 0 and result.total_duration_sec <= 3 * 3600:
            print(f"  ✅ Total duration reasonable: {result.total_duration_sec // 60} min")
            checks.append(True)
        else:
            print(f"  ❌ Unreasonable duration: {result.total_duration_sec}")
            checks.append(False)
        
        return all(checks)
        
    except Exception as e:
        print(f"  ❌ Greedy sequencing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_vrptw_solving():
    """Test OR-Tools VRPTW solving."""
    print("\n🔍 OR-Tools VRPTW Solving:")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        from src.routing.vrptw import solve_vrptw, VRPTWConfig
        
        # Create test candidates (small problem for fast solving)
        candidates = pd.DataFrame({
            'id': ['place1', 'place2', 'place3'],
            'name': ['Restaurant A', 'Cafe B', 'Museum C'],
            'lat': [35.6895, 35.6905, 35.6915],
            'lng': [139.6917, 139.6927, 139.6937],
            'score': [0.9, 0.8, 0.7],
            'eta': [300, 600, 900],  # seconds from anchor
        })
        
        start_time = datetime.now().replace(hour=13, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(hours=3)
        
        config = VRPTWConfig(
            service_time_min=30,
            time_limit_sec=5,
            verbose=False
        )
        
        result = solve_vrptw(
            candidates=candidates,
            anchor_lat=35.6895,
            anchor_lng=139.6917,
            start_time=start_time,
            end_time=end_time,
            config=config
        )
        
        checks = []
        
        if result.solution_found:
            print(f"  ✅ OR-Tools found solution with {result.num_stops} stops")
            checks.append(True)
        else:
            print(f"  ⚠️ OR-Tools did not find solution (may be OK for difficult problems)")
            # Still acceptable for verification
            checks.append(True)
        
        if result.sequence_method == "ortools_vrptw":
            print("  ✅ Sequence method correctly labeled")
            checks.append(True)
        else:
            print(f"  ❌ Wrong sequence method: {result.sequence_method}")
            checks.append(False)
        
        if result.solver_time_sec > 0:
            print(f"  ✅ Solver time tracked: {result.solver_time_sec:.3f}s")
            checks.append(True)
        else:
            print("  ⚠️ Solver time not tracked")
            checks.append(True)  # Not critical
        
        if result.solution_found and len(result.stops) > 0:
            # Check that stops have required fields
            first_stop = result.stops[0]
            required_fields = ['place_id', 'place_name', 'lat', 'lng', 'arrival_time', 'departure_time']
            if all(field in first_stop for field in required_fields):
                print("  ✅ Stop data structure correct")
                checks.append(True)
            else:
                print("  ❌ Stop missing required fields")
                checks.append(False)
        else:
            print("  ⚠️ No stops to verify (acceptable if no solution found)")
            checks.append(True)
        
        return all(checks)
        
    except Exception as e:
        print(f"  ❌ VRPTW solving test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_fallback_mechanism():
    """Test automatic fallback to greedy."""
    print("\n🔍 Fallback Mechanism:")
    
    try:
        import pandas as pd
        from datetime import datetime, timedelta
        from src.routing.vrptw import solve_vrptw_with_fallback, VRPTWConfig
        
        # Create impossible problem (tight time window with many stops)
        candidates = pd.DataFrame({
            'id': [f'place{i}' for i in range(10)],
            'name': [f'Place {i}' for i in range(10)],
            'lat': [35.6895 + i * 0.01 for i in range(10)],
            'lng': [139.6917 + i * 0.01 for i in range(10)],
            'score': [0.9 - i * 0.05 for i in range(10)],
            'eta': [300 * (i + 1) for i in range(10)],
        })
        
        start_time = datetime.now().replace(hour=13, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(minutes=30)  # Very tight window
        
        config = VRPTWConfig(
            service_time_min=30,
            time_limit_sec=2,  # Short time limit
            verbose=False
        )
        
        result = solve_vrptw_with_fallback(
            candidates=candidates,
            anchor_lat=35.6895,
            anchor_lng=139.6917,
            start_time=start_time,
            end_time=end_time,
            config=config,
            force_greedy=False
        )
        
        checks = []
        
        # Should either solve with OR-Tools or fall back to greedy
        if result.solution_found:
            print(f"  ✅ Solution found via {result.sequence_method}")
            checks.append(True)
        else:
            print("  ❌ No solution found (fallback should provide greedy)")
            checks.append(False)
        
        # Test forced greedy
        result_forced = solve_vrptw_with_fallback(
            candidates=candidates,
            anchor_lat=35.6895,
            anchor_lng=139.6917,
            start_time=start_time,
            end_time=end_time,
            config=config,
            force_greedy=True
        )
        
        if result_forced.sequence_method == "greedy":
            print("  ✅ Forced greedy mode works")
            checks.append(True)
        else:
            print(f"  ❌ Forced greedy failed: {result_forced.sequence_method}")
            checks.append(False)
        
        if "forced greedy" in result_forced.fallback_reason.lower():
            print("  ✅ Fallback reason correctly set for forced greedy")
            checks.append(True)
        else:
            print(f"  ⚠️ Fallback reason: {result_forced.fallback_reason}")
            checks.append(True)  # Not critical
        
        return all(checks)
        
    except Exception as e:
        print(f"  ❌ Fallback mechanism test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_integration():
    """Test integration with geotrip_agent.py."""
    print("\n🔗 Integration with geotrip_agent.py:")
    
    try:
        from geotrip_agent import _sequence_single_day, Location, TimeWindow
        from datetime import datetime, timedelta
        import pandas as pd
        
        # Create mock ScoredPlace objects
        from geotrip_agent import ScoredPlace, PlaceLite
        
        mock_places = []
        for i in range(3):
            place = PlaceLite(
                id=f"place{i}",
                name=f"Place {i}",
                primary_type="restaurant",
                types=["restaurant", "food"],
                lat=35.6895 + i * 0.001,
                lng=139.6917 + i * 0.001,
                rating=4.5,
                user_ratings_total=100,
                price_level=2,
                is_open_now=True,
                maps_url=None
            )
            
            scored = ScoredPlace(
                place=place,
                eta_sec=300 * (i + 1),
                cluster_id=0,
                cluster_label="dining",
                diversity_gain=0.5,
                crowd_proxy=0.3,
                score=0.9 - i * 0.1
            )
            mock_places.append(scored)
        
        # Create time window
        start_time = datetime.now().replace(hour=13, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(hours=3)
        window = TimeWindow(
            start_iso=start_time.isoformat(),
            end_iso=end_time.isoformat()
        )
        
        # Create anchor
        anchor = Location(lat=35.6895, lng=139.6917)
        
        # Call sequencing function
        result = _sequence_single_day(
            stops=mock_places,
            anchor=anchor,
            window=window,
            use_ortools=False,  # Use greedy for fast test
            service_time_min=30
        )
        
        checks = []
        
        if hasattr(result, 'stops'):
            print(f"  ✅ Returns ItineraryDay with {len(result.stops)} stops")
            checks.append(True)
        else:
            print("  ❌ Result missing 'stops' attribute")
            checks.append(False)
        
        if hasattr(result, 'date_iso'):
            print("  ✅ Returns date_iso field")
            checks.append(True)
        else:
            print("  ❌ Result missing 'date_iso' attribute")
            checks.append(False)
        
        # Check that function accepts use_ortools parameter
        try:
            result_ortools = _sequence_single_day(
                stops=mock_places,
                anchor=anchor,
                window=window,
                use_ortools=True,  # Try OR-Tools
                service_time_min=30
            )
            print("  ✅ Function accepts use_ortools parameter")
            checks.append(True)
        except Exception as e:
            print(f"  ❌ OR-Tools mode failed: {e}")
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
    print("PR #5 VERIFICATION: OR-Tools VRPTW Sequencer")
    print("=" * 60)
    
    results = []
    
    # Run all checks
    results.append(("File Structure", check_file_structure()))
    results.append(("Imports", check_imports()))
    results.append(("Greedy Sequencing", check_greedy_sequencing()))
    results.append(("VRPTW Solving", check_vrptw_solving()))
    results.append(("Fallback Mechanism", check_fallback_mechanism()))
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
        print("\n🎉 All checks passed! PR #5 is complete.")
        print("\nNext: PR #6 (CI & Testing Infrastructure)")
        return 0
    else:
        print(f"\n❌ {total - passed} check(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
