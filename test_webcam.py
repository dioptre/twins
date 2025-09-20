#!/usr/bin/env python3
"""
Simple test script for webcam capture functionality
"""

import cv2
import numpy as np

def test_webcam():
    """Test if webcam is accessible"""
    print("Testing webcam access...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read frame from webcam")
        cap.release()
        return False
    
    print(f"✅ Webcam working - Resolution: {frame.shape[1]}x{frame.shape[0]}")
    cap.release()
    return True

def test_background_subtractor():
    """Test background subtraction"""
    print("Testing background subtraction...")
    
    try:
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Create a test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mask = bg_subtractor.apply(test_frame)
        
        print(f"✅ Background subtraction working - Mask shape: {mask.shape}")
        return True
    except Exception as e:
        print(f"❌ Background subtraction failed: {e}")
        return False

def test_opencv_features():
    """Test required OpenCV features"""
    print("Testing OpenCV features...")
    
    # Test morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    test_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    result = cv2.morphologyEx(test_img, cv2.MORPH_OPEN, kernel)
    
    print("✅ Morphological operations working")
    
    # Test colormap
    colored = cv2.applyColorMap(test_img, cv2.COLORMAP_JET)
    print("✅ Color mapping working")
    
    return True

def main():
    print("🔍 Testing webcam capture prerequisites...")
    print("="*50)
    
    tests = [
        ("Webcam Access", test_webcam),
        ("Background Subtraction", test_background_subtractor),
        ("OpenCV Features", test_opencv_features)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        if test_func():
            passed += 1
    
    print("\n" + "="*50)
    print(f"📊 Test Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! You can run the webcam capture script.")
        print("\nTo start capturing:")
        print("  python webcam_capture.py")
        print("\nControls:")
        print("  's' - Save best frame and run PIFuHD")
        print("  'r' - Reset best frame")
        print("  'q' - Quit")
    else:
        print("⚠️  Some tests failed. Please check your setup.")

if __name__ == "__main__":
    main()