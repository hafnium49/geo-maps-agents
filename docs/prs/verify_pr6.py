#!/usr/bin/env python3
"""
PR #6 Verification Script: CI/CD & Testing Infrastructure

This script validates that PR #6 has been correctly implemented with:
- Complete test suite (104 tests)
- Test infrastructure (pytest.ini, .coveragerc, conftest.py)
- Mock API fixtures (JSON files)
- GitHub Actions CI/CD workflow
- Documentation updates (README, CHANGELOG)
- Coverage ≥80%

Run: python verify_pr6.py
"""

import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print a bold header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}\n")


def print_check(name: str, passed: bool, details: str = ""):
    """Print a check result with color coding."""
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"{status} | {name}")
    if details:
        print(f"       {details}")


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists."""
    exists = Path(filepath).exists()
    if exists:
        size = Path(filepath).stat().st_size
        print_check(description, True, f"Found: {filepath} ({size} bytes)")
    else:
        print_check(description, False, f"Missing: {filepath}")
    return exists


def check_directory_exists(dirpath: str, description: str) -> bool:
    """Check if a directory exists."""
    exists = Path(dirpath).is_dir()
    if exists:
        file_count = len(list(Path(dirpath).rglob("*")))
        print_check(description, True, f"Found: {dirpath} ({file_count} items)")
    else:
        print_check(description, False, f"Missing: {dirpath}")
    return exists


def validate_json_file(filepath: str, description: str) -> bool:
    """Validate that a file is valid JSON."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        key_count = len(data) if isinstance(data, dict) else len(data)
        print_check(description, True, f"Valid JSON with {key_count} top-level keys/items")
        return True
    except json.JSONDecodeError as e:
        print_check(description, False, f"Invalid JSON: {e}")
        return False
    except Exception as e:
        print_check(description, False, f"Error reading file: {e}")
        return False


def validate_yaml_file(filepath: str, description: str) -> bool:
    """Validate that a file is valid YAML."""
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        print_check(description, True, f"Valid YAML")
        return True
    except yaml.YAMLError as e:
        print_check(description, False, f"Invalid YAML: {e}")
        return False
    except Exception as e:
        print_check(description, False, f"Error reading file: {e}")
        return False


def check_pytest_config() -> bool:
    """Validate pytest.ini configuration."""
    if not Path("pytest.ini").exists():
        print_check("pytest.ini configuration", False, "File not found")
        return False
    
    try:
        with open("pytest.ini", 'r') as f:
            content = f.read()
        
        # Check for required sections
        checks = [
            ("testpaths = tests", "testpaths configured"),
            ("python_files = test_*.py", "test file pattern"),
            ("markers =", "test markers defined"),
            ("-p no:launch_testing_ros", "ROS plugin disabled"),
        ]
        
        all_passed = True
        for pattern, desc in checks:
            if pattern in content:
                print(f"       ✓ {desc}")
            else:
                print(f"       ✗ {desc} - NOT FOUND")
                all_passed = False
        
        print_check("pytest.ini configuration", all_passed, "")
        return all_passed
    except Exception as e:
        print_check("pytest.ini configuration", False, f"Error: {e}")
        return False


def check_coverage_config() -> bool:
    """Validate .coveragerc configuration."""
    if not Path(".coveragerc").exists():
        print_check(".coveragerc configuration", False, "File not found")
        return False
    
    try:
        with open(".coveragerc", 'r') as f:
            content = f.read()
        
        checks = [
            ("source = src", "source directory configured"),
            ("omit =", "omit patterns defined"),
            ("show_missing = True", "show missing lines enabled"),
        ]
        
        all_passed = True
        for pattern, desc in checks:
            if pattern in content:
                print(f"       ✓ {desc}")
            else:
                print(f"       ✗ {desc} - NOT FOUND")
                all_passed = False
        
        print_check(".coveragerc configuration", all_passed, "")
        return all_passed
    except Exception as e:
        print_check(".coveragerc configuration", False, f"Error: {e}")
        return False

def check_conftest() -> bool:
    """Validate tests/conftest.py has required fixtures."""
    filepath = "tests/conftest.py"
    if not Path(filepath).exists():
        print_check("conftest.py fixtures", False, "File not found")
        return False
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        required_fixtures = [
            "sample_places",
            "sample_distance_matrix",
            "mock_places_api",
            "mock_routes_api",
        ]
        
        all_found = True
        for fixture in required_fixtures:
            if f"def {fixture}" in content or f"def {fixture}(" in content:
                print(f"       ✓ {fixture} fixture defined")
            else:
                print(f"       ✗ {fixture} fixture NOT FOUND")
                all_found = False
        
        print_check("conftest.py fixtures", all_found, "")
        return all_found
    except Exception as e:
        print_check("conftest.py fixtures", False, f"Error: {e}")
        return False


def count_tests() -> Tuple[bool, Dict[str, int]]:
    """Count tests in each test file."""
    test_files = {
        "tests/test_scoring.py": 30,
        "tests/test_spatial.py": 28,
        "tests/test_routing.py": 28,
        "tests/test_integration.py": 18,
    }
    
    counts = {}
    all_passed = True
    
    for filepath, expected in test_files.items():
        if not Path(filepath).exists():
            print(f"       ✗ {filepath} - NOT FOUND")
            all_passed = False
            continue
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Count test functions
            actual = content.count("def test_")
            counts[filepath] = actual
            
            if actual >= expected:
                print(f"       ✓ {filepath}: {actual} tests (expected ≥{expected})")
            else:
                print(f"       ✗ {filepath}: {actual} tests (expected ≥{expected})")
                all_passed = False
        except Exception as e:
            print(f"       ✗ {filepath} - Error: {e}")
            all_passed = False
    
    total = sum(counts.values())
    print_check(f"Test count ({total} total)", all_passed, "")
    return all_passed, counts


def run_tests() -> Tuple[bool, int, float]:
    """Run pytest and check if all tests pass."""
    print(f"\n{Colors.YELLOW}Running full test suite (this may take ~25 seconds)...{Colors.END}")
    
    try:
        env = os.environ.copy()
        env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
        
        result = subprocess.run(
            [
                'uv', 'run', 'pytest', 'tests/',
                '-v',
                '-p', 'no:launch_testing_ros',
                '-p', 'no:cov',
                '--tb=short',
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        output = result.stdout + result.stderr
        
        # Parse output for test count
        passed = 0
        failed = 0
        duration = 0.0
        
        for line in output.split('\n'):
            if 'passed' in line.lower():
                # Look for pattern like "104 passed in 25.22s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed':
                        try:
                            passed = int(parts[i-1])
                        except:
                            pass
                    if 'in' in part and i+1 < len(parts):
                        try:
                            duration = float(parts[i+1].rstrip('s'))
                        except:
                            pass
            if 'failed' in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'failed':
                        try:
                            failed = int(parts[i-1])
                        except:
                            pass
        
        success = result.returncode == 0 and failed == 0
        print_check(
            "All tests pass", 
            success, 
            f"{passed} passed, {failed} failed in {duration:.2f}s"
        )
        return success, passed, duration
        
    except subprocess.TimeoutExpired:
        print_check("All tests pass", False, "Tests timed out after 60s")
        return False, 0, 0.0
    except Exception as e:
        print_check("All tests pass", False, f"Error running tests: {e}")
        return False, 0, 0.0


def check_coverage() -> Tuple[bool, float]:
    """Run tests with coverage and check if ≥80%."""
    print(f"\n{Colors.YELLOW}Running tests with coverage (this may take ~30 seconds)...{Colors.END}")
    
    try:
        # Use PYTEST_DISABLE_PLUGIN_AUTOLOAD to avoid ROS plugin, but explicitly load pytest-cov
        env = os.environ.copy()
        env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
        
        result = subprocess.run(
            [
                'uv', 'run', 'pytest', 'tests/',
                '-p', 'pytest_cov.plugin',
                '-p', 'no:launch_testing_ros',
                '--cov=src',
                '--cov-report=term-missing',
                '--cov-report=html',
                '-q',
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        
        output = result.stdout + result.stderr
        
        # Parse coverage percentage
        coverage = 0.0
        for line in output.split('\n'):
            if 'TOTAL' in line and '%' in line:
                parts = line.split()
                for part in parts:
                    if '%' in part:
                        try:
                            coverage = float(part.rstrip('%'))
                            break
                        except:
                            pass
        
        passed = coverage >= 80.0
        print_check(
            "Coverage ≥80%", 
            passed, 
            f"Coverage: {coverage:.1f}%"
        )
        return passed, coverage
        
    except subprocess.TimeoutExpired:
        print_check("Coverage ≥80%", False, "Coverage check timed out after 90s")
        return False, 0.0
    except Exception as e:
        print_check("Coverage ≥80%", False, f"Error running coverage: {e}")
        return False, 0.0


def check_ci_workflow() -> bool:
    """Validate GitHub Actions workflow."""
    filepath = ".github/workflows/ci.yml"
    if not Path(filepath).exists():
        print_check("CI/CD workflow", False, "File not found")
        return False
    
    try:
        with open(filepath, 'r') as f:
            content = yaml.safe_load(f)
        
        checks = []
        
        # Check workflow name
        if 'name' in content:
            checks.append(("Workflow name defined", True))
        else:
            checks.append(("Workflow name defined", False))
        
        # Check triggers (YAML 'on' becomes True in Python)
        trigger_key = True if True in content else 'on' if 'on' in content else None
        if trigger_key:
            on_config = content[trigger_key]
            has_push = 'push' in on_config
            has_pr = 'pull_request' in on_config
            checks.append(("Push trigger", has_push))
            checks.append(("PR trigger", has_pr))
        else:
            checks.append(("Triggers configured", False))
        
        # Check jobs
        if 'jobs' in content:
            jobs = content['jobs']
            checks.append(("Lint job defined", 'lint' in jobs))
            checks.append(("Test job defined", 'test' in jobs))
        else:
            checks.append(("Jobs defined", False))
        
        all_passed = all(c[1] for c in checks)
        
        for desc, passed in checks:
            symbol = "✓" if passed else "✗"
            print(f"       {symbol} {desc}")
        
        print_check("CI/CD workflow", all_passed, "")
        return all_passed
        
    except Exception as e:
        print_check("CI/CD workflow", False, f"Error: {e}")
        return False


def check_readme() -> bool:
    """Validate README.md has testing section."""
    filepath = "README.md"
    if not Path(filepath).exists():
        print_check("README.md testing section", False, "File not found")
        return False
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        checks = [
            ("## 🧪 Testing", "Testing section header"),
            ("104 tests", "Test count mentioned"),
            ("pytest", "pytest mentioned"),
            ("coverage", "coverage mentioned"),
            ("CI/CD", "CI/CD mentioned"),
        ]
        
        all_passed = True
        for pattern, desc in checks:
            if pattern in content:
                print(f"       ✓ {desc}")
            else:
                print(f"       ✗ {desc} - NOT FOUND")
                all_passed = False
        
        print_check("README.md testing section", all_passed, "")
        return all_passed
        
    except Exception as e:
        print_check("README.md testing section", False, f"Error: {e}")
        return False


def check_changelog() -> bool:
    """Validate CHANGELOG.md has v0.6.0 entry."""
    filepath = "CHANGELOG.md"
    if not Path(filepath).exists():
        print_check("CHANGELOG.md v0.6.0", False, "File not found")
        return False
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        checks = [
            ("[v0.6.0]", "v0.6.0 version entry"),
            ("PR #6", "PR #6 mentioned"),
            ("104 tests", "Test count mentioned"),
            ("CI/CD", "CI/CD mentioned"),
        ]
        
        all_passed = True
        for pattern, desc in checks:
            if pattern in content:
                print(f"       ✓ {desc}")
            else:
                print(f"       ✗ {desc} - NOT FOUND")
                all_passed = False
        
        print_check("CHANGELOG.md v0.6.0", all_passed, "")
        return all_passed
        
    except Exception as e:
        print_check("CHANGELOG.md v0.6.0", False, f"Error: {e}")
        return False


def main():
    """Run all verification checks."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                                                                    ║")
    print("║          PR #6 Verification: CI/CD & Testing Infrastructure        ║")
    print("║                                                                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(Colors.END)
    
    results = {}
    
    # Check 1: Test directory structure
    print_header("CHECK 1: Test Directory Structure")
    results['test_structure'] = (
        check_directory_exists("tests", "tests/ directory") and
        check_directory_exists("tests/fixtures", "tests/fixtures/ directory") and
        check_file_exists("tests/__init__.py", "tests/__init__.py") and
        check_file_exists("tests/conftest.py", "tests/conftest.py") and
        check_file_exists("tests/test_scoring.py", "tests/test_scoring.py") and
        check_file_exists("tests/test_spatial.py", "tests/test_spatial.py") and
        check_file_exists("tests/test_routing.py", "tests/test_routing.py") and
        check_file_exists("tests/test_integration.py", "tests/test_integration.py")
    )
    
    # Check 2: Mock fixtures
    print_header("CHECK 2: Mock API Fixtures")
    results['fixtures'] = (
        check_file_exists("tests/fixtures/places_api.json", "places_api.json") and
        check_file_exists("tests/fixtures/routes_api.json", "routes_api.json") and
        validate_json_file("tests/fixtures/places_api.json", "places_api.json validation") and
        validate_json_file("tests/fixtures/routes_api.json", "routes_api.json validation")
    )
    
    # Check 3: Test configuration
    print_header("CHECK 3: Test Configuration Files")
    results['config'] = (
        check_pytest_config() and
        check_coverage_config() and
        check_conftest()
    )
    
    # Check 4: Test count
    print_header("CHECK 4: Test Count Validation")
    test_count_passed, test_counts = count_tests()
    results['test_count'] = test_count_passed
    
    # Check 5: Run tests
    print_header("CHECK 5: Test Execution")
    test_passed, passed_count, duration = run_tests()
    results['test_execution'] = test_passed
    
    # Check 6: Coverage
    print_header("CHECK 6: Code Coverage")
    coverage_passed, coverage_pct = check_coverage()
    results['coverage'] = coverage_passed
    
    # Check 7: CI/CD workflow
    print_header("CHECK 7: GitHub Actions CI/CD")
    results['cicd'] = (
        check_directory_exists(".github/workflows", ".github/workflows/ directory") and
        check_file_exists(".github/workflows/ci.yml", "ci.yml workflow") and
        validate_yaml_file(".github/workflows/ci.yml", "ci.yml validation") and
        check_ci_workflow()
    )
    
    # Check 8: Documentation
    print_header("CHECK 8: Documentation Updates")
    results['documentation'] = (
        check_readme() and
        check_changelog()
    )
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    total_checks = len(results)
    passed_checks = sum(1 for v in results.values() if v)
    
    print(f"\n{Colors.BOLD}Results by Category:{Colors.END}\n")
    
    for i, (name, passed) in enumerate(results.items(), 1):
        status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
        name_display = name.replace('_', ' ').title()
        print(f"  {i}. {status} | {name_display}")
    
    print(f"\n{Colors.BOLD}Overall Score:{Colors.END}")
    print(f"  {passed_checks}/{total_checks} checks passed ({passed_checks/total_checks*100:.1f}%)")
    
    if passed_checks == total_checks:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 SUCCESS! All verification checks passed!{Colors.END}")
        print(f"\n{Colors.GREEN}PR #6 is complete and ready for review.{Colors.END}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  INCOMPLETE: {total_checks - passed_checks} check(s) failed{Colors.END}")
        print(f"\n{Colors.YELLOW}Please review the failed checks above and fix the issues.{Colors.END}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
