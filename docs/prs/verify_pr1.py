#!/usr/bin/env python3
"""
Verification script for PR #1: Config & Secrets Infrastructure

Run this to verify that all PR #1 components are working correctly.
"""

import sys
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


def check_config_load():
    """Check if config loader works."""
    try:
        from src.tools.config_loader import get_config
        config = get_config()
        assert "name" in config
        assert "h3" in config
        print(f"✅ Config loader works - loaded profile: {config['name']}")
        return True
    except Exception as e:
        print(f"❌ Config loader failed: {e}")
        return False


def check_fieldmask_functions():
    """Check if FieldMask helper functions work."""
    try:
        from src.tools.fields import (
            get_places_search_mask,
            get_places_details_mask,
            get_routes_matrix_mask,
        )
        
        mask1 = get_places_search_mask()
        mask2 = get_places_details_mask()
        mask3 = get_routes_matrix_mask()
        
        assert "X-Goog-FieldMask" in mask1
        assert "X-Goog-FieldMask" in mask2
        assert "X-Goog-FieldMask" in mask3
        
        print("✅ FieldMask helper functions work")
        return True
    except Exception as e:
        print(f"❌ FieldMask functions failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("PR #1 Verification: Config & Secrets Infrastructure")
    print("=" * 60)
    print()
    
    checks = []
    
    # File existence checks
    print("📁 File Structure Checks:")
    checks.append(check_file_exists("pyproject.toml", "Project config"))
    checks.append(check_file_exists(".env.sample", "Environment template"))
    checks.append(check_file_exists("requirements.txt", "Requirements file"))
    checks.append(check_file_exists("requirements-dev.txt", "Dev requirements"))
    checks.append(check_file_exists("src/tools/fields.py", "FieldMask definitions"))
    checks.append(check_file_exists("src/tools/config_loader.py", "Config loader"))
    checks.append(check_file_exists("configs/dense-city.yaml", "Dense city config"))
    checks.append(check_file_exists("configs/suburban.yaml", "Suburban config"))
    checks.append(check_file_exists("configs/rural.yaml", "Rural config"))
    checks.append(check_file_exists(".gitignore", "Git ignore rules"))
    print()
    
    # Import checks
    print("📦 Import Checks:")
    checks.append(check_import("src.tools.fields", "FieldMask module"))
    checks.append(check_import("src.tools.config_loader", "Config loader module"))
    print()
    
    # Functional checks
    print("⚙️  Functional Checks:")
    checks.append(check_config_load())
    checks.append(check_fieldmask_functions())
    print()
    
    # Summary
    print("=" * 60)
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"🎉 All checks passed! ({passed}/{total})")
        print()
        print("✅ PR #1 is complete and working correctly!")
        print()
        print("Next steps:")
        print("1. Copy .env.sample to .env and add your API keys")
        print("2. Install dependencies: pip install -e .")
        print("3. Run the main script: python geotrip_agent.py")
        return 0
    else:
        print(f"⚠️  Some checks failed: {passed}/{total} passed")
        print()
        print("Please review the errors above and ensure all files are created.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
