#!/bin/bash
# Complete setup and run proof of concept

echo "=== Simpulse Proof of Concept ==="
echo ""

# Step 1: Check Python
echo "1. Checking Python..."
if command -v python3 &> /dev/null; then
    echo "✓ Python installed: $(python3 --version)"
else
    echo "✗ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Step 2: Check/Install Lean 4
echo ""
echo "2. Checking Lean 4..."
if command -v lean &> /dev/null; then
    echo "✓ Lean installed: $(lean --version)"
else
    echo "✗ Lean not found. Installing..."
    ./scripts/setup_lean_env.sh
    
    # Reload PATH
    export PATH="$HOME/.elan/bin:$PATH"
    
    # Verify again
    if command -v lean &> /dev/null; then
        echo "✓ Lean installed successfully"
    else
        echo "✗ Lean installation failed. Please install manually:"
        echo "  curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh"
        exit 1
    fi
fi

# Step 3: Create virtual environment (optional but recommended)
echo ""
echo "3. Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Created virtual environment"
fi

# Activate venv
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || true

# Install any requirements (currently none needed)
pip install -q --upgrade pip

# Step 4: Run proof of concept
echo ""
echo "4. Running proof of concept..."
echo "================================"
echo ""

python3 scripts/proof_of_concept.py

echo ""
echo "================================"
echo "Proof of concept complete!"