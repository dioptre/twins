#!/usr/bin/env python3
"""
Automated webcam capture for PIFuHD that detects new people and automatically generates 3D models.
"""

import cv2
import numpy as np
import os
import time
import json
import hashlib
from datetime import datetime
from collections import deque
import threading
import subprocess

class AutoWebcamCapture:
    def __init__(self, 
                 background_frames=30, 
                 quality_threshold=0.25,
                 capture_cooldown=10,
                 stability_frames=15,
                 auto_capture_delay=3.0):
        
        self.cap = cv2.VideoCapture(0)
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Configuration
        self.background_frames = background_frames
        self.quality_threshold = quality_threshold
        self.capture_cooldown = capture_cooldown  # seconds between auto-captures
        self.stability_frames = stability_frames   # frames to wait for stable person
        self.auto_capture_delay = auto_capture_delay  # delay before auto-capture
        
        # State tracking
        self.frame_count = 0
        self.last_capture_time = 0
        self.person_detected_time = None
        self.person_stable_count = 0
        self.current_best_frame = None
        self.current_best_score = 0
        self.current_best_mask = None
        
        # Person tracking
        self.person_history = deque(maxlen=30)  # Track person presence
        self.person_hash_history = []  # Track different people
        self.processing_queue = []
        self.is_processing = False
        
        # Setup
        self.output_dir = "auto_captures"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("results/auto_webcam", exist_ok=True)
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Person detection model
        self.person_cascade = None
        self.init_person_detector()
        
    def init_person_detector(self):
        """Initialize person detection"""
        try:
            # Try to load a person detector (fallback to background subtraction if not available)
            cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            if os.path.exists(cascade_path):
                self.person_cascade = cv2.CascadeClassifier(cascade_path)
                print("✅ Person cascade detector loaded")
            else:
                print("⚠️ Person cascade not found, using background subtraction only")
        except Exception as e:
            print(f"⚠️ Person detector init failed: {e}")
    
    def detect_person_features(self, frame, mask):
        """Extract features to identify different people"""
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Extract features for person identification
        person_region = frame[y:y+h, x:x+w]
        if person_region.size == 0:
            return None
        
        # Resize for consistent comparison
        person_region_small = cv2.resize(person_region, (64, 128))
        
        # Create simple feature hash (color histogram + shape)
        hist_b = cv2.calcHist([person_region_small], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([person_region_small], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([person_region_small], [2], None, [32], [0, 256])
        
        # Combine histograms and add shape info
        feature_vector = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
        feature_vector = np.append(feature_vector, [w/h, w*h])  # aspect ratio and area
        
        # Create hash
        feature_hash = hashlib.md5(feature_vector.tobytes()).hexdigest()[:12]
        
        return {
            'hash': feature_hash,
            'bbox': (x, y, w, h),
            'area': cv2.contourArea(largest_contour)
        }
    
    def is_new_person(self, person_features):
        """Check if this appears to be a new person"""
        if not person_features:
            return False
            
        current_hash = person_features['hash']
        
        # Check against recent person hashes
        for prev_hash, timestamp in self.person_hash_history:
            # If we've seen this person recently, not new
            if current_hash == prev_hash and (time.time() - timestamp) < 60:
                return False
        
        return True
    
    def calculate_frame_quality(self, frame, mask, person_features):
        """Calculate frame quality with person-specific metrics"""
        if not person_features:
            return 0, 0, 0
        
        # Convert to grayscale for sharpness calculation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness using Laplacian variance in person region
        x, y, w, h = person_features['bbox']
        person_region = gray[y:y+h, x:x+w]
        if person_region.size > 0:
            sharpness = cv2.Laplacian(person_region, cv2.CV_64F).var()
        else:
            sharpness = 0
        
        # Calculate person area ratio
        person_area = person_features['area'] / (frame.shape[0] * frame.shape[1])
        
        # Quality score with person-specific weights
        quality_score = (sharpness / 1000) * person_area
        
        # Bonus for good size person (not too small, not too large)
        size_bonus = 1.0
        if 0.1 < person_area < 0.6:  # Good size range
            size_bonus = 1.3
        
        quality_score *= size_bonus
        
        return quality_score, sharpness, person_area
    
    def should_auto_capture(self, person_features):
        """Determine if we should automatically capture this frame"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_capture_time < self.capture_cooldown:
            return False
        
        # Must have person
        if not person_features:
            self.person_detected_time = None
            self.person_stable_count = 0
            return False
        
        # Check if new person
        if not self.is_new_person(person_features):
            return False
        
        # Check person stability (same person for several frames)
        if self.person_detected_time is None:
            self.person_detected_time = current_time
            self.person_stable_count = 1
            return False
        
        self.person_stable_count += 1
        
        # Need stability and minimum time
        stable_duration = current_time - self.person_detected_time
        
        return (self.person_stable_count >= self.stability_frames and 
                stable_duration >= self.auto_capture_delay)
    
    def auto_save_and_process(self, person_features):
        """Automatically save the best frame and start processing"""
        if not self.current_best_frame is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            person_id = person_features['hash']
            
            print(f"\n🎯 Auto-capturing new person: {person_id}")
            print(f"📊 Quality score: {self.current_best_score:.3f}")
            
            # Save files
            original_path = os.path.join(self.output_dir, f"person_{person_id}_{timestamp}_original.jpg")
            processed_path = os.path.join(self.output_dir, f"person_{person_id}_{timestamp}_processed.jpg")
            
            cv2.imwrite(original_path, self.current_best_frame)
            
            # Remove background
            processed_frame = self.remove_background(self.current_best_frame, self.current_best_mask)
            cv2.imwrite(processed_path, processed_frame)
            
            # Create keypoints
            keypoints_data = self.create_pose_keypoints(self.current_best_frame, self.current_best_mask)
            if keypoints_data:
                keypoints_path = os.path.join(self.output_dir, f"person_{person_id}_{timestamp}_processed_keypoints.json")
                with open(keypoints_path, 'w') as f:
                    json.dump(keypoints_data, f)
                
                # Add to processing queue
                self.processing_queue.append({
                    'person_id': person_id,
                    'timestamp': timestamp,
                    'input_dir': self.output_dir,
                    'files': {
                        'original': original_path,
                        'processed': processed_path,
                        'keypoints': keypoints_path
                    }
                })
                
                print(f"📁 Files saved for person {person_id}")
                
                # Start processing in background
                if not self.is_processing:
                    threading.Thread(target=self.process_queue, daemon=True).start()
            
            # Update tracking
            self.person_hash_history.append((person_features['hash'], time.time()))
            self.last_capture_time = time.time()
            
            # Reset for next person
            self.reset_capture_state()
    
    def process_queue(self):
        """Process PIFuHD reconstructions in background"""
        self.is_processing = True
        
        while self.processing_queue:
            task = self.processing_queue.pop(0)
            person_id = task['person_id']
            
            print(f"\n🔄 Processing 3D reconstruction for person {person_id}...")
            
            try:
                # Run PIFuHD
                cmd = [
                    "python", "-m", "apps.simple_test",
                    "--input_path", task['input_dir'],
                    "--out_path", "./results/auto_webcam",
                    "--ckpt_path", "./checkpoints/pifuhd.pt"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
                
                if result.returncode == 0:
                    print(f"✅ 3D reconstruction completed for person {person_id}")
                    print(f"📁 Results in: ./results/auto_webcam/")
                else:
                    print(f"❌ PIFuHD failed for person {person_id}: {result.stderr}")
                    
            except Exception as e:
                print(f"❌ Processing error for person {person_id}: {e}")
        
        self.is_processing = False
        print("\n🎉 All processing complete!")
    
    def reset_capture_state(self):
        """Reset capture state for next person"""
        self.current_best_frame = None
        self.current_best_score = 0
        self.current_best_mask = None
        self.person_detected_time = None
        self.person_stable_count = 0
    
    def remove_background(self, frame, mask):
        """Remove background using the mask"""
        kernel = np.ones((5,5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
        
        mask_3ch_clean = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR) / 255.0
        result = frame * mask_3ch_clean
        white_bg = np.ones_like(frame) * 255
        result = result + white_bg * (1 - mask_3ch_clean)
        
        return result.astype(np.uint8)
    
    def create_pose_keypoints(self, frame, mask):
        """Create basic pose keypoints from mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        keypoints = []
        center_x, center_y = x + w//2, y + h//2
        
        # Basic skeleton keypoints
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
            keypoints.extend([px, py, 0.8])
        
        while len(keypoints) < 75:
            keypoints.extend([0, 0, 0])
        
        return {
            "version": 1.3,
            "people": [{"pose_keypoints_2d": keypoints[:75]}]
        }
    
    def run(self):
        """Main capture loop with automatic detection"""
        print("🚀 Starting automated webcam capture...")
        print("👁️ The system will automatically detect and capture new people")
        print("Press 'q' to quit, 'p' to pause/resume auto-capture")
        print("Learning background...")
        
        learning_phase = True
        auto_capture_enabled = True
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.frame_count += 1
            
            # Learn background
            if learning_phase and self.frame_count <= self.background_frames:
                self.background_subtractor.apply(frame, learningRate=0.1)
                cv2.putText(frame, f"Learning background: {self.frame_count}/{self.background_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow('Auto Webcam Capture', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            if learning_phase:
                learning_phase = False
                print("✅ Background learning complete. Auto-capture active!")
            
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame, learningRate=0.01)
            
            # Clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Detect person features
            person_features = self.detect_person_features(frame, fg_mask)
            
            # Track person presence
            self.person_history.append(person_features is not None)
            
            # Calculate quality if person detected
            quality_score = 0
            sharpness = 0 
            person_area = 0
            
            if person_features:
                quality_score, sharpness, person_area = self.calculate_frame_quality(frame, fg_mask, person_features)
                
                # Update best frame for current person
                if quality_score > self.current_best_score and person_area > self.quality_threshold:
                    self.current_best_score = quality_score
                    self.current_best_frame = frame.copy()
                    self.current_best_mask = fg_mask.copy()
                
                # Auto-capture logic
                if auto_capture_enabled and self.should_auto_capture(person_features):
                    self.auto_save_and_process(person_features)
            
            # Create display frame
            display_frame = frame.copy()
            
            # Add status info
            status_color = (0, 255, 0) if auto_capture_enabled else (0, 0, 255)
            status_text = "AUTO-CAPTURE ON" if auto_capture_enabled else "AUTO-CAPTURE PAUSED"
            
            info_texts = [
                f"Status: {status_text}",
                f"Quality: {quality_score:.3f} (Best: {self.current_best_score:.3f})",
                f"Person Area: {person_area:.3f}",
                f"Processing Queue: {len(self.processing_queue)}",
                f"People Captured: {len(self.person_hash_history)}"
            ]
            
            for i, text in enumerate(info_texts):
                color = status_color if i == 0 else (255, 255, 255)
                cv2.putText(display_frame, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add person detection indicator
            if person_features:
                x, y, w, h = person_features['bbox']
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"ID: {person_features['hash'][:6]}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Show stability progress
                if self.person_detected_time and auto_capture_enabled:
                    stability_progress = min(self.person_stable_count / self.stability_frames, 1.0)
                    time_progress = min((time.time() - self.person_detected_time) / self.auto_capture_delay, 1.0)
                    
                    # Progress bars
                    bar_width = w
                    cv2.rectangle(display_frame, (x, y+h+5), 
                                (x + int(bar_width * stability_progress), y+h+15), (0, 255, 255), -1)
                    cv2.rectangle(display_frame, (x, y+h+20), 
                                (x + int(bar_width * time_progress), y+h+30), (255, 0, 255), -1)
            
            # Show mask overlay
            if person_features:
                mask_overlay = cv2.applyColorMap(fg_mask, cv2.COLORMAP_JET)
                display_frame = cv2.addWeighted(display_frame, 0.8, mask_overlay, 0.2, 0)
            
            cv2.imshow('Auto Webcam Capture', display_frame)
            cv2.imshow('Person Mask', fg_mask)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                auto_capture_enabled = not auto_capture_enabled
                print(f"Auto-capture {'enabled' if auto_capture_enabled else 'disabled'}")
                self.reset_capture_state()
        
        # Wait for any remaining processing to complete
        print("Waiting for processing to complete...")
        while self.is_processing:
            time.sleep(1)
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture = AutoWebcamCapture(
        capture_cooldown=15,        # 15 seconds between captures
        stability_frames=20,        # 20 frames of stability required
        auto_capture_delay=2.0      # 2 seconds after detection
    )
    capture.run()