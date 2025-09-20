#!/usr/bin/env python3
"""
Test script for automated webcam capture functionality
"""

import cv2
import numpy as np
from auto_webcam_capture import AutoWebcamCapture

def test_person_detection():
    """Test person feature extraction"""
    print("Testing person detection and feature extraction...")
    
    # Create test frame with simulated person
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_mask = np.zeros((480, 640), dtype=np.uint8)
    
    # Draw a person-like shape
    cv2.rectangle(test_frame, (200, 100), (400, 400), (100, 150, 200), -1)
    cv2.rectangle(test_mask, (200, 100), (400, 400), 255, -1)
    
    capture = AutoWebcamCapture()
    features = capture.detect_person_features(test_frame, test_mask)
    
    if features:
        print(f"✅ Person features extracted:")
        print(f"  - Hash: {features['hash']}")
        print(f"  - Bbox: {features['bbox']}")
        print(f"  - Area: {features['area']}")
        return True
    else:
        print("❌ Failed to extract person features")
        return False

def test_quality_calculation():
    """Test frame quality calculation"""
    print("Testing frame quality calculation...")
    
    # Create sharp test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(test_mask, (200, 100), (400, 400), 255, -1)
    
    capture = AutoWebcamCapture()
    person_features = capture.detect_person_features(test_frame, test_mask)
    
    if person_features:
        quality, sharpness, area = capture.calculate_frame_quality(test_frame, test_mask, person_features)
        print(f"✅ Quality metrics calculated:")
        print(f"  - Quality Score: {quality:.3f}")
        print(f"  - Sharpness: {sharpness:.1f}")
        print(f"  - Person Area: {area:.3f}")
        return True
    else:
        print("❌ Failed to calculate quality")
        return False

def test_new_person_detection():
    """Test new person detection logic"""
    print("Testing new person detection...")
    
    import time
    capture = AutoWebcamCapture()
    
    # Create two different person features
    person1 = {'hash': 'person123abc', 'bbox': (100, 100, 200, 300), 'area': 60000}
    person2 = {'hash': 'person456def', 'bbox': (150, 120, 180, 280), 'area': 50400}
    
    # First person should be new
    is_new1 = capture.is_new_person(person1)
    print(f"  - Person 1 is new: {is_new1}")
    
    # Add person1 to history with current time
    capture.person_hash_history.append((person1['hash'], time.time()))
    
    # Same person should not be new (within 60 second window)
    is_new1_again = capture.is_new_person(person1)
    print(f"  - Person 1 again is new: {is_new1_again}")
    
    # Different person should be new
    is_new2 = capture.is_new_person(person2)
    print(f"  - Person 2 is new: {is_new2}")
    
    if is_new1 and not is_new1_again and is_new2:
        print("✅ New person detection working correctly")
        return True
    else:
        print("❌ New person detection failed")
        return False

def test_background_subtraction():
    """Test background subtraction setup"""
    print("Testing background subtraction...")
    
    try:
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mask = bg_subtractor.apply(test_frame)
        
        print(f"✅ Background subtraction working - Mask shape: {mask.shape}")
        return True
    except Exception as e:
        print(f"❌ Background subtraction failed: {e}")
        return False

def main():
    print("🔍 Testing automated webcam capture system...")
    print("="*60)
    
    tests = [
        ("Background Subtraction", test_background_subtraction),
        ("Person Detection", test_person_detection),
        ("Quality Calculation", test_quality_calculation),
        ("New Person Detection", test_new_person_detection)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        if test_func():
            passed += 1
        print()
    
    print("="*60)
    print(f"📊 Test Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Automated capture system is ready.")
        print("\nTo start automated capture:")
        print("  python auto_webcam_capture.py")
        print("\nSystem features:")
        print("  🎯 Automatic person detection")
        print("  🔄 Background processing queue")
        print("  ⏱️  Configurable capture cooldowns")
        print("  👥 Multiple person tracking")
        print("  📊 Quality-based frame selection")
        print("\nControls:")
        print("  'q' - Quit")
        print("  'p' - Pause/Resume auto-capture")
    else:
        print("⚠️  Some tests failed. Please check your setup.")

if __name__ == "__main__":
    main()