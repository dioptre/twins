#!/usr/bin/env python3
"""
Webcam capture script for PIFuHD with background subtraction and frame quality assessment.
"""

import cv2
import numpy as np
import os
import time
import json
from datetime import datetime

class WebcamCapture:
    def __init__(self, background_frames=30, quality_threshold=0.3):
        self.cap = cv2.VideoCapture(0)
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.background_frames = background_frames
        self.quality_threshold = quality_threshold
        self.best_frame = None
        self.best_score = 0
        self.frame_count = 0
        
        # Create output directory
        self.output_dir = "webcam_captures"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
    def calculate_frame_quality(self, frame, mask):
        """Calculate frame quality based on sharpness and person detection"""
        # Convert to grayscale for sharpness calculation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness using Laplacian variance
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate person area (white pixels in mask)
        person_area = np.sum(mask == 255) / (mask.shape[0] * mask.shape[1])
        
        # Combined quality score
        quality_score = (sharpness / 1000) * person_area
        return quality_score, sharpness, person_area
    
    def remove_background(self, frame, mask):
        """Remove background using the mask"""
        # Create 3-channel mask
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
        
        # Create clean 3-channel mask
        mask_3ch_clean = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Apply mask to frame
        result = frame * mask_3ch_clean
        
        # Add white background
        white_bg = np.ones_like(frame) * 255
        result = result + white_bg * (1 - mask_3ch_clean)
        
        return result.astype(np.uint8)
    
    def create_pose_keypoints(self, frame, mask):
        """Create dummy pose keypoints file for PIFuHD"""
        # Find contours to get bounding box
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Get largest contour (should be the person)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Create basic keypoints based on bounding box
        # This is a simplified version - for best results, use OpenPose
        keypoints = []
        
        # Add basic body keypoints (x, y, confidence)
        center_x, center_y = x + w//2, y + h//2
        
        # Basic skeleton keypoints (simplified)
        skeleton_points = [
            (center_x, y + h//8),  # nose
            (center_x, y + h//6),  # neck
            (center_x - w//4, y + h//3),  # right shoulder
            (center_x + w//4, y + h//3),  # left shoulder
            (center_x - w//3, y + h//2),  # right elbow
            (center_x + w//3, y + h//2),  # left elbow
            (center_x - w//4, y + 2*h//3),  # right wrist
            (center_x + w//4, y + 2*h//3),  # left wrist
            (center_x - w//6, y + h//2),  # right hip
            (center_x + w//6, y + h//2),  # left hip
            (center_x - w//6, y + 3*h//4),  # right knee
            (center_x + w//6, y + 3*h//4),  # left knee
            (center_x - w//8, y + 7*h//8),  # right ankle
            (center_x + w//8, y + 7*h//8),  # left ankle
            (center_x - w//12, y + h//10),  # right eye
            (center_x + w//12, y + h//10),  # left eye
            (center_x - w//20, y + h//8),  # right ear
            (center_x + w//20, y + h//8),  # left ear
        ]
        
        for px, py in skeleton_points:
            keypoints.extend([px, py, 0.8])  # x, y, confidence
        
        # Pad to 75 values (25 keypoints * 3 values each)
        while len(keypoints) < 75:
            keypoints.extend([0, 0, 0])
        
        keypoints_data = {
            "version": 1.3,
            "people": [{
                "pose_keypoints_2d": keypoints[:75]
            }]
        }
        
        return keypoints_data
    
    def capture_and_process(self):
        """Main capture loop"""
        print("Starting webcam capture...")
        print("Press 's' to save best frame, 'r' to reset, 'q' to quit")
        print("Learning background...")
        
        learning_phase = True
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.frame_count += 1
            
            # Learn background for first N frames
            if learning_phase and self.frame_count <= self.background_frames:
                self.background_subtractor.apply(frame, learningRate=0.1)
                cv2.putText(frame, f"Learning background: {self.frame_count}/{self.background_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow('Webcam Capture', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            if learning_phase:
                learning_phase = False
                print("Background learning complete. Starting capture...")
            
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame, learningRate=0.01)
            
            # Remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Calculate quality score
            quality_score, sharpness, person_area = self.calculate_frame_quality(frame, fg_mask)
            
            # Update best frame if this one is better
            if quality_score > self.best_score and person_area > self.quality_threshold:
                self.best_score = quality_score
                self.best_frame = frame.copy()
                self.best_mask = fg_mask.copy()
            
            # Create display frame with info
            display_frame = frame.copy()
            
            # Add quality info
            info_text = [
                f"Quality: {quality_score:.3f}",
                f"Sharpness: {sharpness:.1f}",
                f"Person area: {person_area:.3f}",
                f"Best score: {self.best_score:.3f}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(display_frame, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show mask overlay
            mask_overlay = cv2.applyColorMap(fg_mask, cv2.COLORMAP_JET)
            display_frame = cv2.addWeighted(display_frame, 0.7, mask_overlay, 0.3, 0)
            
            cv2.imshow('Webcam Capture', display_frame)
            cv2.imshow('Foreground Mask', fg_mask)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and self.best_frame is not None:
                self.save_best_frame()
            elif key == ord('r'):
                self.reset_best_frame()
        
        self.cleanup()
    
    def save_best_frame(self):
        """Save the best captured frame and process it"""
        if self.best_frame is None:
            print("No good frame captured yet!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save original frame
        original_path = os.path.join(self.output_dir, f"original_{timestamp}.jpg")
        cv2.imwrite(original_path, self.best_frame)
        
        # Remove background
        processed_frame = self.remove_background(self.best_frame, self.best_mask)
        processed_path = os.path.join(self.output_dir, f"processed_{timestamp}.jpg")
        cv2.imwrite(processed_path, processed_frame)
        
        # Create keypoints file
        keypoints_data = self.create_pose_keypoints(self.best_frame, self.best_mask)
        if keypoints_data:
            keypoints_path = os.path.join(self.output_dir, f"processed_{timestamp}_keypoints.json")
            with open(keypoints_path, 'w') as f:
                json.dump(keypoints_data, f)
        
        print(f"Saved frame with quality score: {self.best_score:.3f}")
        print(f"Files saved:")
        print(f"  Original: {original_path}")
        print(f"  Processed: {processed_path}")
        print(f"  Keypoints: {keypoints_path}")
        
        # Run PIFuHD on the processed frame
        self.run_pifuhd(self.output_dir)
    
    def run_pifuhd(self, input_dir):
        """Run PIFuHD reconstruction on the captured frame"""
        print("Running PIFuHD reconstruction...")
        
        cmd = [
            "python", "-m", "apps.simple_test",
            "--input_path", input_dir,
            "--out_path", "./results/webcam",
            "--ckpt_path", "./checkpoints/pifuhd.pt"
        ]
        
        import subprocess
        try:
            result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
            if result.returncode == 0:
                print("PIFuHD reconstruction completed successfully!")
                print("Check ./results/webcam/ for the 3D mesh")
            else:
                print(f"PIFuHD failed: {result.stderr}")
        except Exception as e:
            print(f"Error running PIFuHD: {e}")
    
    def reset_best_frame(self):
        """Reset the best frame selection"""
        self.best_frame = None
        self.best_score = 0
        print("Reset best frame selection")
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture = WebcamCapture()
    capture.capture_and_process()