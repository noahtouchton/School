#!/bin/bash
# Quick fix script for tkinter on Apple Silicon Mac

echo "🔧 Fixing Tkinter for GUI Support"
echo "=================================="
echo ""

# Check if running on Apple Silicon
if [[ $(uname -m) == 'arm64' ]]; then
    echo "✅ Detected Apple Silicon Mac"
else
    echo "ℹ️  Not Apple Silicon, using standard commands"
fi

echo ""
echo "This script will:"
echo "1. Fix Homebrew permissions"
echo "2. Install python-tk@3.9 for GUI support"
echo ""
echo "⚠️  This requires sudo (administrator) access"
echo ""
read -p "Do you want to continue? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Step 1: Fixing Homebrew permissions..."
    sudo chown -R $(whoami) /opt/homebrew
    
    if [ $? -eq 0 ]; then
        echo "✅ Permissions fixed"
    else
        echo "❌ Failed to fix permissions"
        exit 1
    fi
    
    echo ""
    echo "Step 2: Installing python-tk@3.9..."
    arch -arm64 brew install python-tk@3.9
    
    if [ $? -eq 0 ]; then
        echo "✅ python-tk@3.9 installed successfully"
    else
        echo "❌ Failed to install python-tk@3.9"
        exit 1
    fi
    
    echo ""
    echo "Step 3: Testing tkinter..."
    python3.9 -c "import tkinter; print('✅ Tkinter is working!')"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Success! Tkinter is now available"
        echo ""
        echo "You can now run:"
        echo "  ./setup.sh test-gui    # Test GUI without camera"
        echo "  ./setup.sh gui         # Run full GUI application"
    else
        echo ""
        echo "⚠️  Tkinter test failed. Try alternative solutions:"
        echo ""
        echo "Alternative 1: Use system Python"
        echo "  /usr/bin/python3 -m tkinter"
        echo ""
        echo "Alternative 2: Use console demos (work perfectly!)"
        echo "  ./setup.sh simple-demo"
    fi
else
    echo ""
    echo "Setup cancelled."
    echo ""
    echo "💡 You can still use the console demos:"
    echo "  ./setup.sh simple-demo    # Works without tkinter!"
    echo "  ./setup.sh demo           # Full functionality"
fi











