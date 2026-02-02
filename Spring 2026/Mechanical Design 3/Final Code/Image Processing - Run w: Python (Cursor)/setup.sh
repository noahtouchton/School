#!/bin/bash
# Setup script for Soil Classification System
# This script helps manage the virtual environment and run applications

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment exists
check_venv() {
    if [ ! -d "soil_classification_env" ]; then
        print_error "Virtual environment not found!"
        print_status "Please run: ./setup.sh create"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment with Python 3.9..."
    
    # Check if Python 3.9 is available
    if ! command -v python3.9 &> /dev/null; then
        print_error "Python 3.9 not found!"
        print_status "Installing Python 3.9 via Homebrew..."
        
        if command -v brew &> /dev/null; then
            arch -arm64 brew install python@3.9
        else
            print_error "Homebrew not found. Please install Python 3.9 manually."
            exit 1
        fi
    fi
    
    # Create virtual environment
    python3.9 -m venv soil_classification_env
    print_success "Virtual environment created!"
    
    # Install dependencies
    print_status "Installing dependencies..."
    source soil_classification_env/bin/activate
    pip install --upgrade pip
    pip install -r requirements_python39.txt
    print_success "Dependencies installed!"
}

# Activate virtual environment
activate_venv() {
    check_venv
    print_status "Activating virtual environment..."
    source soil_classification_env/bin/activate
    print_success "Virtual environment activated!"
    print_status "Python version: $(python --version)"
    print_status "Pip version: $(pip --version)"
}

# Run GUI application
run_gui() {
    check_venv
    activate_venv
    
    print_status "Starting Enhanced Soil Classification GUI..."
    print_warning "Make sure your USB camera is connected!"
    
    python launch_gui.py
}

# Run test GUI (without camera)
run_test_gui() {
    check_venv
    activate_venv
    
    print_status "Starting GUI Test Application (no camera required)..."
    python test_gui_components.py
}

# Run demo script (full demo)
run_demo() {
    check_venv
    activate_venv
    
    print_status "Running soil classification demo..."
    python demo.py
}

# Run simple demo
run_simple_demo() {
    check_venv
    activate_venv
    
    print_status "Running simple console demo..."
    python demo_simple.py
}

# Train ML model
train_model() {
    check_venv
    activate_venv
    
    print_status "Training Random Forest model..."
    python train_sklearn_model.py
}

# Train YOLO model
train_yolo() {
    check_venv
    activate_venv
    
    print_status "Training YOLOv11-cls model..."
    print_warning "This will take a few minutes..."
    python train_yolo_model.py
}

# Run comprehensive evaluation
evaluate() {
    check_venv
    activate_venv
    
    print_status "Evaluating all models..."
    print_status "This will compare Random Forest and YOLOv11-cls performance..."
    
    python compare_models.py
    python evaluate_model.py
    
    print_success "Evaluation complete! Check the generated reports."
}

# Prepare dataset
prepare_dataset() {
    check_venv
    activate_venv
    
    print_status "Preparing dataset for training..."
    python prepare_yolo_dataset.py
}

# Show help
show_help() {
    echo "Soil Classification System Setup Script"
    echo "======================================"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  create      Create virtual environment and install dependencies"
    echo "  activate    Activate virtual environment"
    echo "  gui         Run the main GUI application"
    echo "  test-gui    Run GUI test application (no camera required)"
    echo "  demo        Run soil classification demo (detailed)"
    echo "  simple-demo Run simple console demo (quick test)"
    echo "  train       Train Random Forest model"
    echo "  train-yolo  Train YOLOv11-cls model (recommended)"
    echo "  dataset     Prepare dataset for training"
    echo "  evaluate    Run comprehensive model evaluation"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 create     # First time setup"
    echo "  $0 gui        # Run GUI application"
    echo "  $0 test-gui   # Test GUI without camera"
    echo ""
}

# Main script logic
case "${1:-help}" in
    create)
        create_venv
        ;;
    activate)
        activate_venv
        ;;
    gui)
        run_gui
        ;;
    test-gui)
        run_test_gui
        ;;
    demo)
        run_demo
        ;;
    simple-demo)
        run_simple_demo
        ;;
    train)
        train_model
        ;;
    train-yolo)
        train_yolo
        ;;
    dataset)
        prepare_dataset
        ;;
    evaluate)
        evaluate
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac


