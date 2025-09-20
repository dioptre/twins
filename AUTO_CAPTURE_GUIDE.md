# PIFuHD Automated Webcam Capture

This system automatically detects new people entering the webcam view and generates 3D models without manual intervention.

## 🚀 Quick Start

```bash
source .venv/bin/activate
python auto_webcam_capture.py
```

## 🧠 How It Works

### 1. **Background Learning Phase**
- First 30 frames learn the empty background
- **Important:** Stay out of camera view during this phase

### 2. **Automated Detection**
- System continuously monitors for people
- Creates unique "fingerprints" for each person using:
  - Color histograms
  - Body shape/size
  - Position characteristics

### 3. **Quality Assessment**
- Tracks best frame for each detected person
- Considers sharpness, pose, and visibility
- Waits for stable, high-quality captures

### 4. **Automatic Processing**
- When a new person is detected and stable:
  - Captures best frame automatically
  - Removes background
  - Generates pose keypoints
  - Runs PIFuHD reconstruction in background
  - Saves all results with person ID

## ⚙️ Configuration

Default settings (modify in script):

```python
AutoWebcamCapture(
    capture_cooldown=15,        # Seconds between captures
    stability_frames=20,        # Frames person must be stable
    auto_capture_delay=2.0      # Seconds after detection
)
```

## 🎮 Controls

- **'q'** - Quit application
- **'p'** - Pause/Resume auto-capture mode

## 📊 Visual Interface

The system displays:

- **Status**: AUTO-CAPTURE ON/PAUSED
- **Quality metrics**: Current and best scores
- **Person detection**: Bounding boxes with IDs
- **Progress bars**: 
  - Yellow: Frame stability progress
  - Magenta: Time delay progress
- **Processing queue**: Number of pending reconstructions

## 🎯 Person Detection Features

### **Unique Person Identification**
- Each person gets a unique 12-character ID
- Based on appearance features (color, shape, size)
- Prevents duplicate processing of same person

### **Smart Cooldown System**
- 15-second minimum between captures
- 60-second memory window for person recognition
- Prevents spam captures of same individual

### **Quality-Based Selection**
- Monitors sharpness and person visibility
- Automatically selects best frame over time
- Requires minimum quality thresholds

## 📁 Output Structure

```
auto_captures/
├── person_a1b2c3d4_20250920_143052_original.jpg     # Raw capture
├── person_a1b2c3d4_20250920_143052_processed.jpg    # Background removed
└── person_a1b2c3d4_20250920_143052_processed_keypoints.json

results/auto_webcam/pifuhd_final/recon/
├── result_person_a1b2c3d4_20250920_143052_processed_512.obj
└── result_person_a1b2c3d4_20250920_143052_processed_512.png
```

## 🔧 Troubleshooting

### **No Auto-Captures Happening**
- Check person area > 0.25 (25% of frame)
- Ensure good lighting and contrast
- Try moving closer to camera
- Check if cooldown period is active

### **Same Person Captured Multiple Times**
- Person recognition failed due to:
  - Large appearance changes (clothing, pose)
  - Poor lighting conditions
  - Camera angle changes
- Consider increasing cooldown period

### **Poor Quality Captures**
- Improve lighting setup
- Reduce camera motion/shake
- Ask subjects to stand still briefly
- Adjust quality thresholds in code

### **Processing Queue Backing Up**
- PIFuHD processing is CPU/GPU intensive
- Each reconstruction takes 1-2 minutes
- Consider reducing capture frequency
- Monitor system resources

## 🎨 Advanced Customization

### **Adjust Detection Sensitivity**
```python
# In auto_webcam_capture.py
quality_threshold=0.15      # Lower = more sensitive
stability_frames=10         # Lower = faster triggers
```

### **Modify Person Recognition**
```python
# Change recognition window
if (current_hash == prev_hash and (time.time() - timestamp) < 30):  # 30 seconds instead of 60
```

### **Custom Quality Scoring**
```python
# Modify calculate_frame_quality() method
# Adjust weights for sharpness vs. person area
quality_score = (sharpness / 500) * (person_area * 2)  # Favor person area more
```

## 🏢 Production Setup Tips

### **Optimal Camera Placement**
- Mount 6-8 feet from expected person positions
- Chest/waist height for full body capture
- Good uniform lighting
- Plain, contrasting background

### **Performance Optimization**
- Use dedicated GPU for PIFuHD processing
- Consider lower resolution for faster processing
- Implement frame skipping for high FPS cameras
- Add disk space monitoring for output files

### **Multi-Camera Support**
Run multiple instances with different camera indices:
```python
self.cap = cv2.VideoCapture(1)  # Camera index 1, 2, etc.
```

## 📈 Monitoring & Analytics

The system tracks:
- Total people captured
- Processing queue length
- Average quality scores
- Capture success rates
- Processing completion times

Consider adding logging for production deployments.

---

**Ready for hands-free 3D human capture!** 🎥➡️👥➡️🤖➡️🎯