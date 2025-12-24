#!/bin/bash

set -e
# Install dependencies
echo "Installing dependencies..."
python3 -m pip install -r requirements.txt

# Build the project
echo "Building the project..."
python3 setup.py clean
python3 -m build --wheel --no-isolation
python3 -m pip install dist/*.whl --force-reinstall --no-deps

# Run tests
echo "Running tests..."
pytest tests/

# professional, simple green status message (no blink), kernel-like "[  OK  ]" style
_green=$(tput setaf 2 2>/dev/null || printf '\033[0;32m')
_bold=$(tput bold 2>/dev/null || printf '\033[1m')
_reset=$(tput sgr0 2>/dev/null || printf '\033[0m')
printf '%s%s[  OK  ]%s BUILD & TESTS completed successfully — 构建与测试成功\n' "$_green" "$_bold" "$_reset"
