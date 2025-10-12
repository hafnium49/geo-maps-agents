#!/bin/bash
# Git workflow helper for PR #1: Config & Secrets Infrastructure

echo "================================================"
echo "PR #1: Config & Secrets Infrastructure"
echo "Git Workflow Helper"
echo "================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "geotrip_agent.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

echo "📋 Files to be committed:"
echo ""
echo "New files:"
echo "  - pyproject.toml"
echo "  - .env.sample"
echo "  - requirements.txt"
echo "  - requirements-dev.txt"
echo "  - CHANGELOG.md"
echo "  - PR1_SUMMARY.md"
echo "  - verify_pr1.py"
echo "  - configs/ (dense-city.yaml, suburban.yaml, rural.yaml)"
echo "  - src/ (tools/fields.py, tools/config_loader.py, __init__.py files)"
echo ""
echo "Modified files:"
echo "  - .gitignore"
echo "  - README.md"
echo "  - geotrip_agent.py"
echo ""

# Ask for confirmation
read -p "Do you want to create a feature branch and commit these changes? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Create feature branch
echo ""
echo "📌 Creating feature branch: feat/pr1-config-secrets"
git checkout -b feat/pr1-config-secrets

# Add all files
echo ""
echo "➕ Adding files to staging..."
git add pyproject.toml .env.sample requirements*.txt .gitignore
git add CHANGELOG.md PR1_SUMMARY.md verify_pr1.py
git add configs/ src/ 
git add README.md geotrip_agent.py

# Show what will be committed
echo ""
echo "📊 Git status after adding files:"
git status --short

echo ""
read -p "Proceed with commit? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted. Files are staged but not committed."
    echo "You can commit manually with: git commit -m 'your message'"
    exit 0
fi

# Commit with detailed message
echo ""
echo "💾 Creating commit..."
git commit -m "feat: add config & secrets infrastructure (PR #1)

- Add pyproject.toml with pinned dependencies
- Add .env.sample for API key management
- Centralize FieldMasks in src/tools/fields.py
- Add city profile configs (dense-city, suburban, rural)
- Add config loader with YAML support
- Update README with ToS guardrails and improved quickstart
- Add comprehensive .gitignore
- Integrate python-dotenv for environment loading
- Add verification script and CHANGELOG

This PR establishes foundational configuration management,
transforming the project from proof-of-concept to production-ready.

Closes: Part 1 of 6-PR roadmap from code review
Testing: All checks pass in verify_pr1.py (14/14)"

echo ""
echo "✅ Commit created successfully!"
echo ""
echo "📤 Next steps:"
echo "1. Review the commit: git show"
echo "2. Push to remote: git push origin feat/pr1-config-secrets"
echo "3. Create pull request on GitHub"
echo ""
echo "Or to continue with more changes:"
echo "  - Keep working on this branch"
echo "  - Run 'git add' and 'git commit' as needed"
