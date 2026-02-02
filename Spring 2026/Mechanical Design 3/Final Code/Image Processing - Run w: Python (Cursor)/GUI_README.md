# Enhanced Soil Classification GUI

A comprehensive graphical user interface for the Enhanced Soil Classification System with USB camera support and real-time classification capabilities.

## Features

### 🎥 **Live Camera Feed**
- Real-time USB camera display
- Multiple camera device support
- Camera selection dropdown
- Camera refresh functionality

### 🤖 **Intelligent Classification**
- ML-based classification with scikit-learn
- Rule-based fallback system
- Configurable confidence threshold (0.1 - 1.0)
- Real-time method switching (ML/Rule-based)

### 📸 **Capture & Analysis**
- Manual capture and classification
- Auto-classification mode (every 1-30 seconds)
- Real-time results display
- Detailed analysis metrics

### 📊 **Results Management**
- Live results display with timestamps
- Session statistics tracking
- CSV export functionality
- Session save/load capabilities
- Results clearing

### ⚙️ **Advanced Controls**
- ML enable/disable toggle
- Confidence threshold slider
- Auto-classification interval control
- Camera device selection
- Real-time status updates

## Installation

### Prerequisites
```bash
# Install required dependencies
pip install opencv-python Pillow numpy scikit-learn joblib pandas
```

### Quick Start
```bash
# Run the GUI application
python3 launch_gui.py
```

## Usage

### 1. **Starting the Application**
```bash
python3 launch_gui.py
```

The launcher will:
- Check for missing dependencies
- Import the GUI module
- Start the application
- Provide fallback options if needed

### 2. **Camera Setup**
1. Connect USB camera to your computer
2. Click "🔄 Refresh Cameras" to detect available devices
3. Select camera from dropdown if multiple devices available
4. Camera feed will appear in the left panel

### 3. **Classification Settings**
- **🤖 Enable ML Classification**: Toggle between ML and rule-based classification
- **🎯 Confidence Threshold**: Adjust threshold for ML predictions (0.1 - 1.0)
- **⏰ Auto-classify**: Enable automatic classification every N seconds

### 4. **Capturing Samples**
- **Manual Capture**: Click "📸 Capture & Classify" to capture current frame
- **Auto Capture**: Enable auto-classification for continuous monitoring

### 5. **Viewing Results**
Results are displayed in the right panel with:
- Sample ID and timestamp
- Classification result and confidence
- Method used (ML/Rule-based)
- Detailed analysis metrics
- Processing time and lighting conditions

### 6. **Exporting Data**
- Click "📊 Export to CSV" to save all results
- Click "💾 Save Session" to save current settings
- Click "📁 Load Session" to restore previous settings

## GUI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                    Enhanced Soil Classification GUI              │
├─────────────────────────┬───────────────────────────────────────┤
│ Camera Feed & Controls  │ Classification Results               │
│                         │                                       │
│ [Camera Selection]      │ [Results Display Area]               │
│                         │                                       │
│ [Live Video Feed]       │                                       │
│                         │                                       │
│ [Capture] [Stop] [Ref]  │                                       │
│                         │                                       │
│ Classification Settings │ [Export] [Clear] [Save] [Load]       │
│ ☑ Enable ML            │                                       │
│ 🎯 Threshold: [====]    │ Session Statistics                    │
│ ☑ Auto-classify every   │ [Statistics Display]                 │
│                         │                                       │
├─────────────────────────┴───────────────────────────────────────┤
│ Status: Ready                                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration Options

### ML Settings
- **ML Threshold**: Minimum confidence for ML predictions
- **Fallback**: Automatic fallback to rule-based when ML confidence < threshold

### Camera Settings
- **Resolution**: 640x480 display resolution
- **Frame Rate**: ~30 FPS
- **Device Selection**: Support for multiple camera devices

### Auto-classification
- **Interval**: 1-30 seconds between automatic captures
- **Mode**: Continuous monitoring with configurable frequency

## File Structure

```
soil_classification_gui.py          # Basic GUI implementation
enhanced_soil_classification_gui.py  # Enhanced GUI with advanced features
launch_gui.py                       # Launcher script with error handling
soil_classifier_enhanced.py         # Enhanced classification system
soil_classifier_sklearn.pkl         # Trained ML model
```

## Troubleshooting

### Common Issues

1. **Camera Not Detected**
   - Ensure camera is connected and recognized by system
   - Click "🔄 Refresh Cameras" to rescan devices
   - Check camera permissions in system settings

2. **ML Model Not Available**
   - Ensure `soil_classifier_sklearn.pkl` exists
   - Run `python3 train_sklearn_model.py` to retrain model
   - Check file permissions

3. **Import Errors**
   - Install missing dependencies: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)
   - Verify all files are in the same directory

4. **Performance Issues**
   - Reduce auto-classification frequency
   - Lower camera resolution if needed
   - Close other applications to free resources

### Error Messages

- **"Missing Dependencies"**: Install required packages
- **"Camera Error"**: Check camera connection and permissions
- **"Classification Error"**: Verify ML model file exists
- **"Import Error"**: Check file locations and dependencies

## Advanced Features

### Session Management
- Save current settings and configuration
- Load previous sessions
- Preserve classification history

### Statistics Tracking
- Real-time session statistics
- ML vs Rule-based usage tracking
- Classification accuracy monitoring
- Processing time analysis

### Export Options
- CSV export with all classification data
- Session backup and restore
- Results clearing and management

## Performance Tips

1. **Optimal Settings**
   - Use ML classification for best accuracy
   - Set confidence threshold to 0.6-0.8
   - Enable auto-classification for continuous monitoring

2. **Resource Management**
   - Close unnecessary applications
   - Use appropriate camera resolution
   - Monitor system resources

3. **Accuracy Optimization**
   - Ensure good lighting conditions
   - Use consistent camera positioning
   - Regular model retraining with new samples

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Check file permissions and locations
4. Review error messages in the GUI

## Future Enhancements

- **Confidence Heatmap**: Visual overlay showing classification confidence
- **Batch Processing**: Process multiple images simultaneously
- **Model Retraining**: Built-in model retraining interface
- **Advanced Visualization**: Charts and graphs for results analysis
- **Network Support**: Remote camera and processing capabilities















