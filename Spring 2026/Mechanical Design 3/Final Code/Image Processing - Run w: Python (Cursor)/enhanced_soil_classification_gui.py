#!/usr/bin/env python3
"""
Enhanced GUI Application for Soil Classification System

This application provides a comprehensive user interface for real-time soil classification
with advanced features including camera selection, confidence visualization, and batch processing.

Features:
- Live USB camera feed with device selection
- Real-time soil classification with ML/Rule-based toggle
- Confidence threshold adjustment with visual feedback
- Results export and management
- Sample capture with preview
- Auto-classification mode
- Confidence heatmap overlay (stretch goal)
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
import json

# Import our enhanced classifier
from soil_classifier_enhanced import SoilClassifier

class EnhancedSoilClassificationGUI:
    """Enhanced GUI application for soil classification."""
    
    def __init__(self, root):
        """Initialize the enhanced GUI application."""
        self.root = root
        self.root.title("Enhanced Soil Classification System - UF/IFAS Analytical Services")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize classifier
        self.classifier = SoilClassifier(use_ml=True, ml_threshold=0.6)
        
        # Camera and video variables
        self.camera = None
        self.camera_running = False
        self.current_frame = None
        self.captured_frame = None
        self.available_cameras = []
        
        # GUI variables
        self.ml_enabled = tk.BooleanVar(value=True)
        self.confidence_threshold = tk.DoubleVar(value=0.6)
        self.auto_classify = tk.BooleanVar(value=False)
        self.auto_classify_interval = tk.IntVar(value=5)
        self.selected_camera = tk.IntVar(value=0)
        self.selected_camera_display = tk.StringVar(value="Camera 0")
        
        # Results storage
        self.classification_results = []
        self.last_auto_classify = 0
        
        # Setup GUI
        self.setup_gui()
        
        # Detect available cameras
        self.detect_cameras()
        
        # Start camera
        self.start_camera()
        
        # Start video update loop
        self.update_video()
    
    def setup_gui(self):
        """Setup the enhanced GUI layout."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Video and controls
        left_panel = ttk.LabelFrame(main_frame, text="Camera Feed & Controls", padding="5")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Camera selection
        camera_frame = ttk.Frame(left_panel)
        camera_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        ttk.Label(camera_frame, text="Camera:").grid(row=0, column=0, sticky=tk.W)
        self.camera_combo = ttk.Combobox(camera_frame, textvariable=self.selected_camera_display, 
                                        state="readonly", width=15)
        self.camera_combo.grid(row=0, column=1, padx=(5, 0))
        self.camera_combo.bind('<<ComboboxSelected>>', self.on_camera_change)
        
        # Video display
        self.video_label = ttk.Label(left_panel, text="Initializing camera...", 
                                   background='black', foreground='white', font=('Arial', 12))
        self.video_label.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Control buttons
        control_frame = ttk.Frame(left_panel)
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Capture button
        self.capture_btn = ttk.Button(control_frame, text="📸 Capture & Classify", 
                                    command=self.capture_and_classify, state='disabled')
        self.capture_btn.grid(row=0, column=0, padx=(0, 5))
        
        # Camera controls
        self.camera_btn = ttk.Button(control_frame, text="⏹️ Stop Camera", 
                                   command=self.toggle_camera)
        self.camera_btn.grid(row=0, column=1, padx=(0, 5))
        
        # Refresh cameras button
        self.refresh_btn = ttk.Button(control_frame, text="🔄 Refresh Cameras", 
                                    command=self.detect_cameras)
        self.refresh_btn.grid(row=0, column=2, padx=(0, 5))
        
        # Settings frame
        settings_frame = ttk.LabelFrame(left_panel, text="Classification Settings", padding="5")
        settings_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # ML toggle
        ml_frame = ttk.Frame(settings_frame)
        ml_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.ml_checkbox = ttk.Checkbutton(ml_frame, text="🤖 Enable ML Classification", 
                        variable=self.ml_enabled, command=self.update_classifier_settings)
        self.ml_checkbox.grid(row=0, column=0)
        
        # Confidence threshold
        threshold_frame = ttk.Frame(settings_frame)
        threshold_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Label(threshold_frame, text="🎯 Confidence Threshold:").grid(row=0, column=0, sticky=tk.W)
        self.threshold_scale = ttk.Scale(threshold_frame, from_=0.1, to=1.0, 
                                        variable=self.confidence_threshold, 
                                        orient=tk.HORIZONTAL, command=self.update_threshold)
        self.threshold_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        self.threshold_label = ttk.Label(threshold_frame, text="0.60")
        self.threshold_label.grid(row=0, column=2, padx=(5, 0))
        
        # Auto-classify
        auto_frame = ttk.Frame(settings_frame)
        auto_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.auto_checkbox = ttk.Checkbutton(auto_frame, text="⏰ Auto-classify every", 
                       variable=self.auto_classify)
        self.auto_checkbox.grid(row=0, column=0)
        self.auto_interval = ttk.Spinbox(auto_frame, from_=1, to=30, width=5,
                                       textvariable=self.auto_classify_interval)
        self.auto_interval.grid(row=0, column=1, padx=(5, 0))
        ttk.Label(auto_frame, text="seconds").grid(row=0, column=2, padx=(5, 0))
        
        # Right panel - Results
        right_panel = ttk.LabelFrame(main_frame, text="Classification Results", padding="5")
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        
        # Results display
        self.results_text = tk.Text(right_panel, height=25, width=60, wrap=tk.WORD, 
                                   font=('Consolas', 10))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(right_panel, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S), pady=(0, 10))
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Results controls
        results_control_frame = ttk.Frame(right_panel)
        results_control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Export button
        self.export_btn = ttk.Button(results_control_frame, text="📊 Export to CSV", 
                                   command=self.export_results)
        self.export_btn.grid(row=0, column=0, padx=(0, 5))
        
        # Clear results button
        self.clear_btn = ttk.Button(results_control_frame, text="🗑️ Clear Results", 
                                   command=self.clear_results)
        self.clear_btn.grid(row=0, column=1, padx=(0, 5))
        
        # Save session button
        self.save_btn = ttk.Button(results_control_frame, text="💾 Save Session", 
                                  command=self.save_session)
        self.save_btn.grid(row=0, column=2, padx=(0, 5))
        
        # Load session button
        self.load_btn = ttk.Button(results_control_frame, text="📁 Load Session", 
                                  command=self.load_session)
        self.load_btn.grid(row=0, column=3, padx=(0, 5))
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(right_panel, text="Session Statistics", padding="5")
        stats_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.stats_text = tk.Text(stats_frame, height=6, width=60, wrap=tk.WORD, 
                                 font=('Consolas', 9))
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Configure grid weights
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        threshold_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Initialize results display
        self.add_result_message("🚀 Enhanced Soil Classification System initialized")
        yolo_available = hasattr(self.classifier, 'yolo_classifier') and self.classifier.yolo_classifier.model is not None
        rf_available = hasattr(self.classifier, 'rf_classifier') and self.classifier.rf_classifier.model is not None
        self.add_result_message(f"🤖 YOLOv11: {'✅ Available' if yolo_available else '❌ Not Available'}")
        self.add_result_message(f"🤖 Random Forest: {'✅ Available' if rf_available else '❌ Not Available'}")
        self.add_result_message(f"🎯 ML Threshold: {self.confidence_threshold.get():.2f}")
        self.add_result_message(f"📊 Database: {self.classifier.db_path}")
        self.add_result_message("=" * 60)
        
        # Update statistics
        self.update_statistics()
    
    def detect_cameras(self):
        """Detect available camera devices."""
        self.available_cameras = []
        camera_names = []
        
        # Test cameras 0-9
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.available_cameras.append(i)
                    camera_names.append(f"Camera {i}")
                cap.release()
        
        # Update camera combo
        self.camera_combo['values'] = camera_names
        if camera_names:
            self.camera_combo.current(0)
            self.selected_camera.set(0)
            self.add_result_message(f"📹 Found {len(camera_names)} camera(s): {', '.join(camera_names)}")
        else:
            self.add_result_message("⚠️ No cameras detected")
    
    def on_camera_change(self, event=None):
        """Handle camera selection change."""
        # Extract camera index from selection
        selected = self.camera_combo.current()
        if selected >= 0:
            self.selected_camera.set(selected)
            if self.camera_running:
                self.stop_camera()
            self.start_camera()
    
    def start_camera(self):
        """Start the camera capture."""
        try:
            camera_index = self.selected_camera.get()
            if camera_index >= len(self.available_cameras):
                camera_index = 0
            
            self.camera = cv2.VideoCapture(self.available_cameras[camera_index])
            if not self.camera.isOpened():
                raise Exception(f"Could not open camera {camera_index}")
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.camera_running = True
            self.capture_btn.config(state='normal')
            self.status_var.set(f"Camera {camera_index} started")
            self.add_result_message(f"📹 Camera {camera_index} initialized successfully")
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {e}")
            self.status_var.set("Camera error")
            self.add_result_message(f"❌ Camera error: {e}")
    
    def stop_camera(self):
        """Stop the camera capture."""
        if self.camera:
            self.camera.release()
            self.camera = None
        self.camera_running = False
        self.capture_btn.config(state='disabled')
        self.status_var.set("Camera stopped")
        self.add_result_message("⏹️ Camera stopped")
    
    def toggle_camera(self):
        """Toggle camera on/off."""
        if self.camera_running:
            self.stop_camera()
            self.camera_btn.config(text="▶️ Start Camera")
        else:
            self.start_camera()
            self.camera_btn.config(text="⏹️ Stop Camera")
    
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
                    if current_time - self.last_auto_classify >= self.auto_classify_interval.get():
                        self.last_auto_classify = current_time
                        self.capture_and_classify(auto_mode=True)
        
        # Schedule next update
        self.root.after(33, self.update_video)  # ~30 FPS
    
    def capture_and_classify(self, auto_mode=False):
        """Capture current frame and classify it."""
        if self.current_frame is None:
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
            self.status_var.set("🔍 Classifying...")
            self.root.update()
            
            sample = self.classifier.process_image(temp_path, sample_id)
            
            # Store result
            self.classification_results.append(sample)
            
            # Display result with enhanced formatting
            mode_text = "🤖 Auto" if auto_mode else "👤 Manual"
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            result_text = f"\n[{mode_text} Capture] {timestamp}\n"
            result_text += f"🆔 Sample ID: {sample.sample_id}\n"
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
            
            self.add_result_message(result_text)
            
            # Update status with emoji
            status_emoji = "✅" if sample.confidence > 0.8 else "⚠️" if sample.confidence > 0.6 else "❌"
            self.status_var.set(f"{status_emoji} Classified as {sample.classification} ({sample.confidence:.1%})")
            
            # Update statistics
            self.update_statistics()
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
        except Exception as e:
            messagebox.showerror("Classification Error", f"Failed to classify image: {e}")
            self.status_var.set("❌ Classification error")
            self.add_result_message(f"❌ Classification error: {e}")
    
    def update_classifier_settings(self):
        """Update classifier settings based on GUI controls."""
        self.classifier.use_ml = self.ml_enabled.get()
        self.classifier.ml_threshold = self.confidence_threshold.get()
        
        status_text = "🤖 ML enabled" if self.ml_enabled.get() else "📋 Rule-based only"
        self.add_result_message(f"⚙️ Settings updated: {status_text}, threshold: {self.confidence_threshold.get():.2f}")
    
    def update_threshold(self, value):
        """Update confidence threshold."""
        threshold = float(value)
        self.threshold_label.config(text=f"{threshold:.2f}")
        self.classifier.ml_threshold = threshold
        self.add_result_message(f"🎯 Confidence threshold updated: {threshold:.2f}")
    
    def add_result_message(self, message):
        """Add a message to the results display."""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_statistics(self):
        """Update session statistics."""
        if not self.classification_results:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "No classifications yet")
            return
        
        # Calculate statistics
        total_samples = len(self.classification_results)
        ml_samples = sum(1 for s in self.classification_results if s.classification_method == "ml")
        rule_samples = total_samples - ml_samples
        
        type_a_samples = sum(1 for s in self.classification_results if s.classification == "Type A")
        type_b_samples = sum(1 for s in self.classification_results if s.classification == "Type B")
        
        avg_confidence = sum(s.confidence for s in self.classification_results) / total_samples
        avg_processing_time = sum(s.processing_time for s in self.classification_results) / total_samples
        
        # Format statistics
        stats_text = f"📊 Session Statistics:\n"
        stats_text += f"   Total Samples: {total_samples}\n"
        stats_text += f"   ML Classifications: {ml_samples} ({ml_samples/total_samples:.1%})\n"
        stats_text += f"   Rule-based Classifications: {rule_samples} ({rule_samples/total_samples:.1%})\n"
        stats_text += f"   Type A: {type_a_samples} ({type_a_samples/total_samples:.1%})\n"
        stats_text += f"   Type B: {type_b_samples} ({type_b_samples/total_samples:.1%})\n"
        stats_text += f"   Avg Confidence: {avg_confidence:.1%}\n"
        stats_text += f"   Avg Processing Time: {avg_processing_time:.2f}s"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats_text)
    
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
                self.add_result_message(f"📊 Results exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {e}")
            self.add_result_message(f"❌ Export error: {e}")
    
    def clear_results(self):
        """Clear all classification results."""
        if messagebox.askyesno("Clear Results", "Are you sure you want to clear all results?"):
            self.classification_results.clear()
            self.results_text.delete(1.0, tk.END)
            self.add_result_message("🗑️ Results cleared")
            self.update_statistics()
    
    def save_session(self):
        """Save current session to file."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Session"
            )
            
            if filename:
                session_data = {
                    'timestamp': datetime.now().isoformat(),
                    'settings': {
                        'ml_enabled': self.ml_enabled.get(),
                        'confidence_threshold': self.confidence_threshold.get(),
                        'auto_classify': self.auto_classify.get(),
                        'auto_classify_interval': self.auto_classify_interval.get(),
                        'selected_camera': self.selected_camera.get()
                    },
                    'results_count': len(self.classification_results)
                }
                
                with open(filename, 'w') as f:
                    json.dump(session_data, f, indent=2)
                
                messagebox.showinfo("Session Saved", f"Session saved to {filename}")
                self.add_result_message(f"💾 Session saved to {filename}")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save session: {e}")
    
    def load_session(self):
        """Load session from file."""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Load Session"
            )
            
            if filename:
                with open(filename, 'r') as f:
                    session_data = json.load(f)
                
                # Apply settings
                settings = session_data.get('settings', {})
                self.ml_enabled.set(settings.get('ml_enabled', True))
                self.confidence_threshold.set(settings.get('confidence_threshold', 0.6))
                self.auto_classify.set(settings.get('auto_classify', False))
                self.auto_classify_interval.set(settings.get('auto_classify_interval', 5))
                
                self.update_classifier_settings()
                
                messagebox.showinfo("Session Loaded", f"Session loaded from {filename}")
                self.add_result_message(f"📁 Session loaded from {filename}")
                
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load session: {e}")
    
    def on_closing(self):
        """Handle application closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.stop_camera()
            self.root.destroy()

def main():
    """Main function to run the enhanced GUI application."""
    root = tk.Tk()
    app = EnhancedSoilClassificationGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()







