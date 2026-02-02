# How to Run setup.sh

## Quick Start

### 1. Make the script executable (if needed)
```bash
chmod +x setup.sh
```

### 2. View all available commands
```bash
./setup.sh help
```

## Available Commands

### Environment Setup
```bash
# Create virtual environment (first time setup)
./setup.sh create

# Activate virtual environment
./setup.sh activate
```

### Running Applications
```bash
# Run GUI application (with camera)
./setup.sh gui

# Run GUI test (without camera)
./setup.sh test-gui

# Run detailed demo
./setup.sh demo

# Run simple console demo
./setup.sh simple-demo
```

### Training & Evaluation
```bash
# Train Random Forest model
./setup.sh train

# Train YOLOv11-cls model (recommended)
./setup.sh train-yolo

# Prepare dataset for training
./setup.sh dataset

# Run comprehensive evaluation
./setup.sh evaluate
```

### Help
```bash
# Show help message
./setup.sh help
```

## Quick Examples

### First Time Setup
```bash
# 1. Create virtual environment
./setup.sh create

# 2. Activate it
source soil_classification_env/bin/activate

# 3. Run a demo
./setup.sh simple-demo
```

### Daily Usage
```bash
# Run GUI application
./setup.sh gui

# Run console demo
./setup.sh demo
```

### Training Models
```bash
# Train YOLOv11 (recommended, takes ~2 minutes)
./setup.sh train-yolo

# Train Random Forest backup
./setup.sh train
```

## Full Command List

| Command | Description |
|---------|-------------|
| `create` | Create virtual environment with Python 3.9 |
| `activate` | Activate virtual environment |
| `gui` | Run main GUI application (camera required) |
| `test-gui` | Run GUI test (no camera needed) |
| `demo` | Run detailed demo |
| `simple-demo` | Run simple console demo |
| `train` | Train Random Forest model |
| `train-yolo` | Train YOLOv11-cls model |
| `dataset` | Prepare dataset for training |
| `evaluate` | Run comprehensive model evaluation |
| `help` | Show this help message |

## Common Workflows

### Workflow 1: Quick Test
```bash
./setup.sh simple-demo
```

### Workflow 2: Full Demo with GUI
```bash
./setup.sh gui
```

### Workflow 3: Train New Model
```bash
# Prepare dataset
./setup.sh dataset

# Train YOLOv11
./setup.sh train-yolo

# Evaluate results
./setup.sh evaluate
```

### Workflow 4: Test Both Models
```bash
# Train both models
./setup.sh train
./setup.sh train-yolo

# Compare results
./setup.sh evaluate
```

## Troubleshooting

### Permission Denied
```bash
chmod +x setup.sh
```

### Virtual Environment Not Found
```bash
./setup.sh create
```

### Command Not Found
```bash
# Make sure you're in the project directory
cd /Users/anthonytorla/Image_Processing

# Try with explicit path
bash setup.sh help
```

## Notes

- All commands activate the virtual environment automatically
- The GUI requires a USB camera (use `test-gui` for testing without camera)
- YOLOv11 training takes about 2-4 minutes
- Models are saved automatically after training




