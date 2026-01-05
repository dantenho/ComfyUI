#!/bin/bash
# Test runner script for ComfyUI Numba unit tests
# Runs in isolated git worktree or main repository

set -e

echo "================================"
echo "ComfyUI Numba Unit Test Runner"
echo "================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check Python version
echo -e "${BLUE}Python Version:${NC}"
python3 --version
echo ""

# Check if numba is installed
echo -e "${BLUE}Checking Dependencies:${NC}"
if python3 -c "import numba" 2>/dev/null; then
    numba_version=$(python3 -c "import numba; print(numba.__version__)")
    echo -e "  ${GREEN}✓${NC} Numba $numba_version"
else
    echo -e "  ${YELLOW}⚠${NC} Numba not installed"
    echo "     Installing numba>=0.63.0..."
    pip3 install "numba>=0.63.0" --quiet --user
    echo -e "  ${GREEN}✓${NC} Numba installed"
fi

if python3 -c "import numpy" 2>/dev/null; then
    numpy_version=$(python3 -c "import numpy; print(numpy.__version__)")
    echo -e "  ${GREEN}✓${NC} NumPy $numpy_version"
else
    echo -e "  ${RED}✗${NC} NumPy not found (required)"
    exit 1
fi

echo ""

# Check if we're in a worktree
if [ -f .git ] && grep -q "gitdir:" .git 2>/dev/null; then
    echo -e "${BLUE}Environment:${NC} Git worktree"
else
    echo -e "${BLUE}Environment:${NC} Main repository"
fi
echo ""

# Run unit tests
echo "------------------------------------"
echo -e "${BLUE}Running Unit Tests${NC}"
echo "------------------------------------"
echo ""

# Try pytest first, fallback to unittest
if command -v pytest &> /dev/null; then
    echo "Using pytest..."
    if pytest tests-unit/test_numba_utils.py -v --tb=short; then
        echo ""
        echo -e "${GREEN}════════════════════════════════════${NC}"
        echo -e "${GREEN}✓ All tests passed successfully!${NC}"
        echo -e "${GREEN}════════════════════════════════════${NC}"
        exit 0
    else
        echo ""
        echo -e "${RED}════════════════════════════════════${NC}"
        echo -e "${RED}✗ Some tests failed${NC}"
        echo -e "${RED}════════════════════════════════════${NC}"
        exit 1
    fi
else
    echo "Using unittest..."
    if python3 tests-unit/test_numba_utils.py; then
        echo ""
        echo -e "${GREEN}════════════════════════════════════${NC}"
        echo -e "${GREEN}✓ All tests passed successfully!${NC}"
        echo -e "${GREEN}════════════════════════════════════${NC}"
        exit 0
    else
        echo ""
        echo -e "${RED}════════════════════════════════════${NC}"
        echo -e "${RED}✗ Some tests failed${NC}"
        echo -e "${RED}════════════════════════════════════${NC}"
        exit 1
    fi
fi
