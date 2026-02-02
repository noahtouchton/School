#!/usr/bin/env python3
"""
Camera Integration Module for Real-time Soil Classification

This module provides USB camera integration for live soil classification.
This is a stretch goal implementation for future enhancement.
"""

import cv2
import numpy as np
from soil_classifier import SoilClassifier
import time
from typing import Optional, Callable

class CameraSoilClassifier:
    """Real-time soil classification using USB camera."""
    
    def __init__(self, camera_index: int = 0, db_path: str = "soil_samples.db"):
        """
        Initialize camera-based soil classifier.
        
        Args:
            camera_index: USB camera index (usually 0 for default camera)
            db_path: Path to SQLite database for sample storage
        """
        self.classifier = SoilClassifier(db_path)
        self.camera_index = camera_index
        self.camera = None
        self.is_running = False
        
        # Classification parameters
        self.min_confidence = 0.7
        self.frame_skip = 5  # Process every 5th frame for performance
        self.frame_count = 0
        
        # Callbacks
        self.on_classification: Optional[Callable] = None
        self.on_low_lighting: Optional[Callable] = None
        
    def initialize_camera(self) -> bool:
        """
        Initialize the USB camera.
        
        Returns:
            True if camera initialized successfully, False otherwise
        """
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"Camera {self.camera_index} initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.
        
        Returns:
            Captured frame or None if failed
        """
        if not self.camera or not self.camera.isOpened():
            return None
            
        ret, frame = self.camera.read()
        return frame if ret else None
    
    def process_frame(self, frame: np.ndarray) -> Optional[dict]:
        """
        Process a single frame for soil classification.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Classification results or None if processing failed
        """
        try:
            # Save frame temporarily for processing
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Process with soil classifier
            sample = self.classifier.process_image(temp_path, 
                                                 f"camera_{int(time.time())}")
            
            # Clean up temporary file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return {
                'classification': sample.classification,
                'confidence': sample.confidence,
                'lighting_lux': sample.lighting_lux,
                'bin_count': sample.bin_count,
                'processing_time': sample.processing_time,
                'analysis': sample.bin_values
            }
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None
    
    def run_live_classification(self, display_window: bool = True):
        """
        Run live soil classification from camera feed.
        
        Args:
            display_window: Whether to display the camera feed window
        """
        if not self.initialize_camera():
            return
        
        self.is_running = True
        print("Starting live soil classification...")
        print("Press 'q' to quit, 's' to save current frame")
        
        try:
            while self.is_running:
                frame = self.capture_frame()
                if frame is None:
                    continue
                
                # Process every nth frame for performance
                self.frame_count += 1
                if self.frame_count % self.frame_skip == 0:
                    results = self.process_frame(frame)
                    
                    if results:
                        # Check lighting conditions
                        if results['lighting_lux'] < self.classifier.min_lighting_lux:
                            if self.on_low_lighting:
                                self.on_low_lighting(results['lighting_lux'])
                        
                        # Check confidence threshold
                        if results['confidence'] >= self.min_confidence:
                            if self.on_classification:
                                self.on_classification(results)
                            
                            # Display classification on frame
                            text = f"{results['classification']} ({results['confidence']:.1%})"
                            cv2.putText(frame, text, (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "Low confidence", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Display lighting info
                    if results:
                        lighting_text = f"Lighting: {results['lighting_lux']:.0f} lux"
                        cv2.putText(frame, lighting_text, (10, 70), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                if display_window:
                    cv2.imshow('Soil Classification', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save current frame
                        timestamp = int(time.time())
                        filename = f"captured_frame_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"Frame saved as {filename}")
                        
                        # Process saved frame
                        results = self.process_frame(frame)
                        if results:
                            print(f"Classification: {results['classification']} "
                                  f"({results['confidence']:.1%})")
                
        except KeyboardInterrupt:
            print("\nStopping live classification...")
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop the camera and clean up resources."""
        self.is_running = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Camera stopped and resources cleaned up")
    
    def set_classification_callback(self, callback: Callable):
        """Set callback function for when classification is performed."""
        self.on_classification = callback
    
    def set_low_lighting_callback(self, callback: Callable):
        """Set callback function for when lighting is too low."""
        self.on_low_lighting = callback

def demo_live_classification():
    """Demonstrate live soil classification."""
    print("🌱 Live Soil Classification Demo")
    print("=" * 40)
    print("This demo requires a USB camera to be connected.")
    print("Press 'q' to quit, 's' to save current frame")
    print()
    
    # Initialize camera classifier
    camera_classifier = CameraSoilClassifier()
    
    # Set up callbacks
    def on_classification(results):
        print(f"✓ Classification: {results['classification']} "
              f"(confidence: {results['confidence']:.1%})")
    
    def on_low_lighting(lux):
        print(f"⚠ Low lighting warning: {lux:.0f} lux < 1000 lux")
    
    camera_classifier.set_classification_callback(on_classification)
    camera_classifier.set_low_lighting_callback(on_low_lighting)
    
    # Run live classification
    try:
        camera_classifier.run_live_classification()
    except Exception as e:
        print(f"Error running live classification: {e}")
        print("Make sure a USB camera is connected and not being used by another application.")

if __name__ == "__main__":
    demo_live_classification()
