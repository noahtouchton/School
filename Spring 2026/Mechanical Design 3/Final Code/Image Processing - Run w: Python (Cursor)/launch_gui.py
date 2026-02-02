#!/usr/bin/env python3
"""
Launcher script for the Enhanced Soil Classification GUI

This script handles potential import issues and provides fallback options.
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        from PIL import Image, ImageTk
    except ImportError:
        missing_deps.append("Pillow")
    
    try:
        import numpy as np
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import joblib
    except ImportError:
        missing_deps.append("joblib")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        missing_deps.append("scikit-learn")
    
    return missing_deps

def show_dependency_error(missing_deps):
    """Show error dialog for missing dependencies."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    error_msg = f"Missing dependencies:\n\n{', '.join(missing_deps)}\n\n"
    error_msg += "Please install them using:\n"
    error_msg += f"pip install {' '.join(missing_deps)}\n\n"
    error_msg += "Or install all requirements:\n"
    error_msg += "pip install -r requirements.txt"
    
    messagebox.showerror("Missing Dependencies", error_msg)
    root.destroy()

def main():
    """Main launcher function."""
    print("🚀 Starting Enhanced Soil Classification GUI...")
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        show_dependency_error(missing_deps)
        return
    
    print("✅ All dependencies available")
    
    # Try to import the GUI
    try:
        from enhanced_soil_classification_gui import EnhancedSoilClassificationGUI
        
        print("✅ GUI module imported successfully")
        
        # Create and run the GUI
        root = tk.Tk()
        app = EnhancedSoilClassificationGUI(root)
        
        # Handle window closing
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        print("🎯 Starting GUI application...")
        root.mainloop()
        
    except ImportError as e:
        print(f"❌ Failed to import GUI module: {e}")
        
        # Try fallback to basic GUI
        try:
            from soil_classification_gui import SoilClassificationGUI
            
            print("✅ Using fallback GUI module")
            
            root = tk.Tk()
            app = SoilClassificationGUI(root)
            root.protocol("WM_DELETE_WINDOW", app.on_closing)
            root.mainloop()
            
        except ImportError as e2:
            print(f"❌ Fallback GUI also failed: {e2}")
            
            # Show error dialog
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Import Error", 
                               f"Failed to import GUI modules:\n\n{e}\n\n{e2}\n\n"
                               "Please check that all files are present and dependencies are installed.")
            root.destroy()
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Unexpected Error", f"An unexpected error occurred:\n\n{e}")
        root.destroy()

if __name__ == "__main__":
    main()














