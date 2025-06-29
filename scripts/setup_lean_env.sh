#!/bin/bash
# Setup Lean 4 environment for Simpulse testing

echo "Setting up Lean 4 environment..."

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux"
    OS="linux"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# Create a directory for Lean installation
LEAN_DIR="$HOME/.lean4"
mkdir -p "$LEAN_DIR"

# Install elan (Lean version manager)
if ! command -v elan &> /dev/null; then
    echo "Installing elan (Lean version manager)..."
    curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain none
    
    # Add to PATH
    export PATH="$HOME/.elan/bin:$PATH"
    
    # Add to shell profile
    if [[ "$SHELL" == *"zsh"* ]]; then
        echo 'export PATH="$HOME/.elan/bin:$PATH"' >> ~/.zshrc
    else
        echo 'export PATH="$HOME/.elan/bin:$PATH"' >> ~/.bashrc
    fi
else
    echo "elan is already installed"
fi

# Install latest stable Lean 4
echo "Installing Lean 4..."
elan default stable

# Verify installation
echo ""
echo "Verifying installation..."
if command -v lean &> /dev/null; then
    echo "✓ Lean installed:"
    lean --version
else
    echo "✗ Lean installation failed"
    exit 1
fi

if command -v lake &> /dev/null; then
    echo "✓ Lake installed:"
    lake --version
else
    echo "✗ Lake installation failed"
    exit 1
fi

echo ""
echo "Lean 4 setup complete!"
echo "You may need to restart your terminal or run: source ~/.zshrc"