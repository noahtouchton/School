#!/usr/bin/env python3
"""
GUI Application for Enhanced Soil Classification System

This application provides a user-friendly interface for real-time soil classification
using USB camera input and the enhanced SoilClassifier with ML capabilities.

Features:
- Live USB camera feed
- Real-time soil classification
- ML/Rule-based toggle
- Confidence threshold adjustment
- Results export to CSV
- Sample capture and storage
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
from pathlib import Path
import os
import sys

# Import our enhanced classifier
from soil_classifier_enhanced import SoilClassifier

class SoilClassificationGUI:
    """Main GUI application for soil classification."""
    
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("Enhanced Soil Classification System - UF/IFAS")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize classifier
        self.classifier = SoilClassifier(use_ml=True, ml_threshold=0.6)
        
        # Camera and video variables
        self.camera = None
        self.camera_running = False
        self.current_frame = None
        self.captured_frame = None
        
        # GUI variables
        self.ml_enabled = tk.BooleanVar(value=True)
        self.confidence_threshold = tk.DoubleVar(value=0.6)
        self.auto_classify = tk.BooleanVar(value=False)
        self.auto_classify_interval = tk.IntVar(value=5)  # seconds
        
        # Results storage
        self.classification_results = []
        
        # Setup GUI
        self.setup_gui()
        
        # Start camera
        self.start_camera()
        
        # Start video update loop
        self.update_video()
    
    def setup_gui(self):
        """Setup the GUI layout."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Video and controls
        left_panel = ttk.LabelFrame(main_frame, text="Camera Feed", padding="5")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Video display
        self.video_label = ttk.Label(left_panel, text="Initializing camera...", 
                                   background='black', foreground='white')
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Control buttons
        control_frame = ttk.Frame(left_panel)
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Capture button
        self.capture_btn = ttk.Button(control_frame, text="Capture & Classify", 
                                    command=self.capture_and_classify, state='disabled')
        self.capture_btn.grid(row=0, column=0, padx=(0, 5))
        
        # Camera controls
        self.camera_btn = ttk.Button(control_frame, text="Stop Camera", 
                                   command=self.toggle_camera)
        self.camera_btn.grid(row=0, column=1, padx=(0, 5))
        
        # Settings frame
        settings_frame = ttk.LabelFrame(left_panel, text="Settings", padding="5")
        settings_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # ML toggle
        ml_frame = ttk.Frame(settings_frame)
        ml_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Checkbutton(ml_frame, text="Enable ML Classification", 
                        variable=self.ml_enabled, command=self.update_classifier_settings).grid(row=0, column=0)
        
        # Confidence threshold
        threshold_frame = ttk.Frame(settings_frame)
        threshold_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Label(threshold_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky=tk.W)
        self.threshold_scale = ttk.Scale(threshold_frame, from_=0.1, to=1.0, 
                                        variable=self.confidence_threshold, 
                                        orient=tk.HORIZONTAL, command=self.update_threshold)
        self.threshold_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        self.threshold_label = ttk.Label(threshold_frame, text="0.6")
        self.threshold_label.grid(row=0, column=2, padx=(5, 0))
        
        # Auto-classify
        auto_frame = ttk.Frame(settings_frame)
        auto_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Checkbutton(auto_frame, text="Auto-classify every", 
                       variable=self.auto_classify).grid(row=0, column=0)
        self.auto_interval = ttk.Spinbox(auto_frame, from_=1, to=30, width=5,
                                       textvariable=self.auto_classify_interval)
        self.auto_interval.grid(row=0, column=1, padx=(5, 0))
        ttk.Label(auto_frame, text="seconds").grid(row=0, column=2, padx=(5, 0))
        
        # Right panel - Results
        right_panel = ttk.LabelFrame(main_frame, text="Classification Results", padding="5")
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # Results display
        self.results_text = tk.Text(right_panel, height=20, width=50, wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(right_panel, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S), pady=(0, 10))
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Results controls
        results_control_frame = ttk.Frame(right_panel)
        results_control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Export button
        self.export_btn = ttk.Button(results_control_frame, text="Export to CSV", 
                                   command=self.export_results)
        self.export_btn.grid(row=0, column=0, padx=(0, 5))
        
        # Clear results button
        self.clear_btn = ttk.Button(results_control_frame, text="Clear Results", 
                                   command=self.clear_results)
        self.clear_btn.grid(row=0, column=1, padx=(0, 5))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Configure grid weights
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(0, weight=1)
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        threshold_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Initialize results display
        self.add_result_message("Enhanced Soil Classification System initialized")
        yolo_available = hasattr(self.classifier, 'yolo_classifier') and self.classifier.yolo_classifier.model is not None
        rf_available = hasattr(self.classifier, 'rf_classifier') and self.classifier.rf_classifier.model is not None
        self.add_result_message(f"YOLOv11: {'Available' if yolo_available else 'Not Available'}")
        self.add_result_message(f"Random Forest: {'Available' if rf_available else 'Not Available'}")
        self.add_result_message(f"ML Threshold: {self.confidence_threshold.get():.2f}")
    
    def start_camera(self):
        """Start the camera capture."""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.camera_running = True
            self.capture_btn.config(state='normal')
            self.status_var.set("Camera started")
            self.add_result_message("Camera initialized successfully")
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {e}")
            self.status_var.set("Camera error")
            self.add_result_message(f"Camera error: {e}")
    
    def stop_camera(self):
        """Stop the camera capture."""
        if self.camera:
            self.camera.release()
            self.camera = None
        self.camera_running = False
        self.capture_btn.config(state='disabled')
        self.status_var.set("Camera stopped")
        self.add_result_message("Camera stopped")
    
    def toggle_camera(self):
        """Toggle camera on/off."""
        if self.camera_running:
            self.stop_camera()
            self.camera_btn.config(text="Start Camera")
        else:
            self.start_camera()
            self.camera_btn.config(text="Stop Camera")
    
    def update_video(self):
        """Update the video display."""
        if self.camera_running and self.camera:
            ret, frame = self.camera.read()
            if ret:
                self.current_frame = frame.copy()
                
                # Resize frame for display
                display_frame = cv2.resize(frame, (640, 480))
                
                # Convert to RGB for Tkinter
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Update display
                self.video_label.configure(image=img_tk, text="")
                self.video_label.image = img_tk
                
                # Auto-classify if enabled
                if self.auto_classify.get():
                    current_time = time.time()
                    if not hasattr(self, 'last_auto_classify'):
                        self.last_auto_classify = current_time
                    
                    if current_time - self.last_auto_classify >= self.auto_classify_interval.get():
                        self.last_auto_classify = current_time
                        self.capture_and_classify(auto_mode=True)
        
        # Schedule next update
        self.root.after(33, self.update_video)  # ~30 FPS
    
    def capture_and_classify(self, auto_mode=False):
        """Capture current frame and classify it."""
        if not self.current_frame is not None:
            messagebox.showwarning("No Frame", "No frame available to capture")
            return
        
        try:
            # Capture current frame
            self.captured_frame = self.current_frame.copy()
            
            # Resize to 224x224 for classification
            resized_frame = cv2.resize(self.captured_frame, (224, 224))
            
            # Save temporary image for classification
            temp_path = "temp_capture.jpg"
            cv2.imwrite(temp_path, resized_frame)
            
            # Generate sample ID
            sample_id = f"gui_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Classify the image
            self.status_var.set("Classifying...")
            self.root.update()
            
            sample = self.classifier.process_image(temp_path, sample_id)
            
            # Store result
            self.classification_results.append(sample)
            
            # Display result
            mode_text = "Auto" if auto_mode else "Manual"
            result_text = f"\n[{mode_text} Capture] {datetime.now().strftime('%H:%M:%S')}\n"
            result_text += f"Sample ID: {sample.sample_id}\n"
            result_text += f"Classification: {sample.classification}\n"
            result_text += f"Confidence: {sample.confidence:.1%}\n"
            result_text += f"Method: {sample.classification_method}\n"
            if sample.ml_classification:
                result_text += f"ML Classification: {sample.ml_classification}\n"
                result_text += f"ML Confidence: {sample.ml_confidence:.1%}\n"
            result_text += f"Lighting: {sample.lighting_lux:.0f} lux\n"
            result_text += f"Processing Time: {sample.processing_time:.2f}s\n"
            result_text += f"Color Bins: {sample.bin_count}\n"
            result_text += "-" * 50 + "\n"
            
            self.add_result_message(result_text)
            
            # Update status
            self.status_var.set(f"Classified as {sample.classification} ({sample.confidence:.1%})")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
        except Exception as e:
            messagebox.showerror("Classification Error", f"Failed to classify image: {e}")
            self.status_var.set("Classification error")
            self.add_result_message(f"Classification error: {e}")
    
    def update_classifier_settings(self):
        """Update classifier settings based on GUI controls."""
        self.classifier.use_ml = self.ml_enabled.get()
        self.classifier.ml_threshold = self.confidence_threshold.get()
        
        status_text = "ML enabled" if self.ml_enabled.get() else "Rule-based only"
        self.add_result_message(f"Settings updated: {status_text}, threshold: {self.confidence_threshold.get():.2f}")
    
    def update_threshold(self, value):
        """Update confidence threshold."""
        threshold = float(value)
        self.threshold_label.config(text=f"{threshold:.2f}")
        self.classifier.ml_threshold = threshold
        self.add_result_message(f"Confidence threshold updated: {threshold:.2f}")
    
    def add_result_message(self, message):
        """Add a message to the results display."""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()
    
    def export_results(self):
        """Export classification results to CSV."""
        if not self.classification_results:
            messagebox.showinfo("No Results", "No classification results to export")
            return
        
        try:
            # Ask user for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save Classification Results"
            )
            
            if filename:
                # Export using classifier's method
                self.classifier.export_to_csv(filename)
                messagebox.showinfo("Export Complete", f"Results exported to {filename}")
                self.add_result_message(f"Results exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {e}")
            self.add_result_message(f"Export error: {e}")
    
    def clear_results(self):
        """Clear all classification results."""
        if messagebox.askyesno("Clear Results", "Are you sure you want to clear all results?"):
            self.classification_results.clear()
            self.results_text.delete(1.0, tk.END)
            self.add_result_message("Results cleared")
    
    def on_closing(self):
        """Handle application closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.stop_camera()
            self.root.destroy()

def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = SoilClassificationGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()








