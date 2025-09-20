# PIFuHD Webcam Capture Guide

This guide will help you capture high-quality frames from your webcam, automatically remove backgrounds, and generate 3D human models using PIFuHD.

## Quick Start

1. **Activate the environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Test your setup:**
   ```bash
   python test_webcam.py
   ```

3. **Start capturing:**
   ```bash
   python webcam_capture.py
   ```

## How It Works

### Phase 1: Background Learning (30 frames)
- The system learns what your background looks like
- **Important:** Stay out of the camera view during this phase
- You'll see "Learning background: X/30" on screen

### Phase 2: Capture and Quality Assessment
- The system continuously evaluates frame quality based on:
  - **Sharpness:** How clear the image is (higher is better)
  - **Person Area:** How much of the frame contains a person (0.3+ recommended)
  - **Quality Score:** Combined metric (sharpness × person_area)

### Phase 3: Processing
- When you press 's', the best frame is processed:
  - Background is removed automatically
  - Basic pose keypoints are generated
  - PIFuHD reconstruction runs automatically
  - Results saved to `./results/webcam/`

## Controls

- **'s'** - Save best frame and run PIFuHD reconstruction
- **'r'** - Reset best frame selection (start fresh)
- **'q'** - Quit the application

## Tips for Best Results

### Camera Setup
- Use good lighting (avoid backlighting)
- Position camera at chest/waist height
- Ensure full body is visible in frame
- Stand 6-8 feet from camera

### Capture Tips
- Stand still for a few seconds to get sharp images
- Wear form-fitting clothes for better reconstruction
- Avoid complex backgrounds during background learning
- T-pose or A-pose works well for reconstruction

### Quality Metrics
- **Sharpness > 500:** Good image clarity
- **Person Area > 0.3:** Sufficient person detection
- **Quality Score > 0.15:** Recommended for good results

## Output Files

When you save a frame, these files are created:

1. **`webcam_captures/original_TIMESTAMP.jpg`** - Raw webcam frame
2. **`webcam_captures/processed_TIMESTAMP.jpg`** - Background removed
3. **`webcam_captures/processed_TIMESTAMP_keypoints.json`** - Pose data
4. **`results/webcam/`** - PIFuHD 3D reconstruction results

## Troubleshooting

### Common Issues

**"Could not open webcam"**
- Check if another app is using the camera
- Try unplugging/reconnecting USB cameras
- Check camera permissions

**Poor quality scores**
- Improve lighting conditions
- Move closer to camera
- Reduce motion blur by staying still

**Background not removed properly**
- Ensure you weren't in frame during background learning
- Try resetting ('r') and learning background again
- Use a simpler, static background

**PIFuHD reconstruction fails**
- Check that keypoints file was created
- Ensure processed image has good person detection
- Try with a clearer, higher quality frame

### Performance Notes

- PIFuHD reconstruction takes 1-2 minutes per frame
- Requires at least 8GB GPU memory (recommended)
- CPU-only mode is much slower but possible

## Advanced Usage

### Custom Quality Thresholds
```python
capture = WebcamCapture(
    background_frames=30,      # Frames to learn background
    quality_threshold=0.3      # Minimum person area
)
```

### Integration with OpenPose
For better pose estimation, replace the basic keypoint generation with OpenPose:

1. Install OpenPose
2. Modify `create_pose_keypoints()` method
3. Use real pose detection instead of bounding box estimation

## File Structure
```
pifuhd/
├── webcam_capture.py          # Main capture script
├── test_webcam.py            # Setup testing
├── webcam_captures/          # Captured frames
└── results/webcam/           # PIFuHD outputs
    └── pifuhd_final/recon/   # 3D mesh files (.obj)
```

Ready to capture some 3D humans! 🎥➡️🧍‍♂️➡️🎯