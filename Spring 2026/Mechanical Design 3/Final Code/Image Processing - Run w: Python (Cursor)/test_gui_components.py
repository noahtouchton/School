#!/usr/bin/env python3
"""
Test script for GUI components without camera dependency

This script tests the GUI components and classification system
without requiring a physical camera.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from datetime import datetime

# Import our enhanced classifier
from soil_classifier_enhanced import SoilClassifier

class GUITestApp:
    """Test application for GUI components."""
    
    def __init__(self, root):
        """Initialize the test application."""
        self.root = root
        self.root.title("GUI Test - Soil Classification System")
        self.root.geometry("800x600")
        
        # Initialize classifier
        self.classifier = SoilClassifier(use_ml=True, ml_threshold=0.6)
        
        # Test image path
        self.test_images = ["soil1.jpeg", "soil2.jpeg", "soil3.jpeg", "soil4.jpeg", "soil5.jpeg"]
        self.current_image_index = 0
        
        self.setup_gui()
        self.load_test_image()
    
    def setup_gui(self):
        """Setup the test GUI."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image display
        self.image_label = ttk.Label(main_frame, text="No image loaded", 
                                   background='black', foreground='white')
        self.image_label.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Controls
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        
        # Image navigation
        ttk.Button(control_frame, text="◀ Previous", command=self.previous_image).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(control_frame, text="Next ▶", command=self.next_image).grid(row=0, column=1, padx=(0, 5))
        
        # Classification button
        ttk.Button(control_frame, text="🔍 Classify", command=self.classify_image).grid(row=0, column=2, padx=(0, 5))
        
        # Settings
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="5")
        settings_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # ML toggle
        self.ml_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Enable ML Classification", 
                       variable=self.ml_enabled).grid(row=0, column=0, sticky=tk.W)
        
        # Confidence threshold
        threshold_frame = ttk.Frame(settings_frame)
        threshold_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        ttk.Label(threshold_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky=tk.W)
        self.threshold_scale = ttk.Scale(threshold_frame, from_=0.1, to=1.0, 
                                        orient=tk.HORIZONTAL, command=self.update_threshold)
        self.threshold_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        self.threshold_label = ttk.Label(threshold_frame, text="0.60")
        self.threshold_label.grid(row=0, column=2, padx=(5, 0))
        
        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Classification Results", padding="5")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.results_text = tk.Text(results_frame, height=15, width=70, wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        threshold_frame.columnconfigure(1, weight=1)
        
        # Initialize results
        self.add_result("🚀 GUI Test Application initialized")
        yolo_available = hasattr(self.classifier, 'yolo_classifier') and self.classifier.yolo_classifier.model is not None
        rf_available = hasattr(self.classifier, 'rf_classifier') and self.classifier.rf_classifier.model is not None
        self.add_result(f"🤖 YOLOv11: {'✅ Available' if yolo_available else '❌ Not Available'}")
        self.add_result(f"🤖 Random Forest: {'✅ Available' if rf_available else '❌ Not Available'}")
        self.add_result(f"📁 Test Images: {len(self.test_images)}")
        self.add_result("=" * 60)
    
    def load_test_image(self):
        """Load the current test image."""
        if self.current_image_index < len(self.test_images):
            image_path = self.test_images[self.current_image_index]
            if os.path.exists(image_path):
                # Load and display image
                img = cv2.imread(image_path)
                if img is not None:
                    # Resize for display
                    display_img = cv2.resize(img, (400, 300))
                    rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image
                    pil_img = Image.fromarray(rgb_img)
                    img_tk = ImageTk.PhotoImage(image=pil_img)
                    
                    # Update display
                    self.image_label.configure(image=img_tk, text="")
                    self.image_label.image = img_tk
                    
                    self.add_result(f"📸 Loaded image: {image_path}")
                else:
                    self.image_label.configure(text=f"Failed to load: {image_path}")
            else:
                self.image_label.configure(text=f"File not found: {image_path}")
        else:
            self.image_label.configure(text="No more test images")
    
    def previous_image(self):
        """Load previous test image."""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_test_image()
    
    def next_image(self):
        """Load next test image."""
        if self.current_image_index < len(self.test_images) - 1:
            self.current_image_index += 1
            self.load_test_image()
    
    def classify_image(self):
        """Classify the current test image."""
        if self.current_image_index < len(self.test_images):
            image_path = self.test_images[self.current_image_index]
            if os.path.exists(image_path):
                try:
                    # Generate sample ID
                    sample_id = f"test_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Update classifier settings
                    self.classifier.use_ml = self.ml_enabled.get()
                    self.classifier.ml_threshold = self.threshold_scale.get()
                    
                    # Classify the image
                    sample = self.classifier.process_image(image_path, sample_id)
                    
                    # Display result
                    result_text = f"\n🔍 Classification Result - {datetime.now().strftime('%H:%M:%S')}\n"
                    result_text += f"🆔 Sample ID: {sample.sample_id}\n"
                    result_text += f"📁 Image: {image_path}\n"
                    result_text += f"🏷️ Classification: {sample.classification}\n"
                    result_text += f"📊 Confidence: {sample.confidence:.1%}\n"
                    result_text += f"⚙️ Method: {sample.classification_method}\n"
                    
                    if sample.ml_classification:
                        result_text += f"🤖 ML Classification: {sample.ml_classification}\n"
                        result_text += f"🎯 ML Confidence: {sample.ml_confidence:.1%}\n"
                    
                    result_text += f"💡 Lighting: {sample.lighting_lux:.0f} lux\n"
                    result_text += f"⏱️ Processing Time: {sample.processing_time:.2f}s\n"
                    result_text += f"🎨 Color Bins: {sample.bin_count}\n"
                    
                    # Add analysis details
                    analysis = sample.bin_values
                    result_text += f"🔍 Analysis:\n"
                    result_text += f"   • Dark proportion: {analysis['total_dark_proportion']:.1%}\n"
                    result_text += f"   • Brown proportion: {analysis['total_brown_proportion']:.1%}\n"
                    result_text += f"   • Gray proportion: {analysis['total_gray_proportion']:.1%}\n"
                    result_text += f"   • Avg Munsell Value: {analysis['avg_value']:.1f}\n"
                    result_text += f"   • Dominant Hue: {analysis['dominant_hue']}\n"
                    
                    result_text += "─" * 60 + "\n"
                    
                    self.add_result(result_text)
                    
                except Exception as e:
                    self.add_result(f"❌ Classification error: {e}")
            else:
                self.add_result(f"❌ Image file not found: {image_path}")
    
    def update_threshold(self, value):
        """Update confidence threshold display."""
        threshold = float(value)
        self.threshold_label.config(text=f"{threshold:.2f}")
    
    def add_result(self, message):
        """Add a message to the results display."""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()

def main():
    """Main function to run the test application."""
    root = tk.Tk()
    app = GUITestApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()








