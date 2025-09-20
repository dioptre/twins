#!/bin/bash

# PIFuHD Animation Setup Script
# Sets up the complete animation pipeline with UniRig and DigiHuman integration

set -e

echo "🚀 Setting up PIFuHD Animation Pipeline..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "README.md" ]] || [[ ! -d "lib" ]]; then
    print_error "Please run this script from the PIFuHD root directory"
    exit 1
fi

# 1. Initialize and update submodules
print_status "Initializing Git submodules..."
git submodule init
git submodule update --recursive

# 2. Check if submodules exist
if [[ ! -d "external/UniRig" ]]; then
    print_error "UniRig submodule not found. Make sure it was added correctly."
    exit 1
fi

if [[ ! -d "external/DigiHuman" ]]; then
    print_error "DigiHuman submodule not found. Make sure it was added correctly."
    exit 1
fi

print_success "Submodules initialized successfully"

# 3. Set up UniRig dependencies
print_status "Setting up UniRig dependencies..."
cd external/UniRig

# Check if requirements.txt exists
if [[ -f "requirements.txt" ]]; then
    print_status "Installing UniRig Python dependencies..."
    pip install -r requirements.txt
else
    print_warning "UniRig requirements.txt not found. Installing common dependencies..."
    pip install torch torchvision torchaudio
    pip install trimesh
    pip install numpy scipy
    pip install transformers
    pip install omegaconf
fi

# Download UniRig models if available
if [[ -f "src/inference/download.py" ]]; then
    print_status "Downloading UniRig pre-trained models..."
    python src/inference/download.py || print_warning "Model download failed - may need manual setup"
fi

cd ../..

# 4. Set up DigiHuman dependencies  
print_status "Setting up DigiHuman backend dependencies..."
cd external/DigiHuman/Backend

if [[ -f "requirements.txt" ]]; then
    print_status "Installing DigiHuman Python dependencies..."
    pip install -r requirements.txt
else
    print_warning "DigiHuman requirements.txt not found. Installing common dependencies..."
    pip install mediapipe
    pip install opencv-python
    pip install numpy
fi

cd ../../..

# 5. Set up animation pipeline dependencies
print_status "Installing additional animation pipeline dependencies..."
pip install trimesh[easy]  # Full trimesh with all optional dependencies
pip install pymeshlab      # For mesh processing
pip install open3d        # For 3D operations

# 6. Create output directories
print_status "Creating output directories..."
mkdir -p animations
mkdir -p temp_animation
mkdir -p results

# 7. Make scripts executable
print_status "Setting up script permissions..."
chmod +x apps/animate_model.py
chmod +x scripts/setup_animation.sh

if [[ -f "external/UniRig/launch/inference/generate_skeleton.sh" ]]; then
    chmod +x external/UniRig/launch/inference/generate_skeleton.sh
fi

if [[ -f "external/UniRig/launch/inference/generate_skin.sh" ]]; then
    chmod +x external/UniRig/launch/inference/generate_skin.sh
fi

# 8. Test basic imports
print_status "Testing Python imports..."
python -c "
try:
    import trimesh
    import numpy as np
    import cv2
    print('✅ Core dependencies working')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
" || exit 1

# 9. Create example usage script
print_status "Creating example usage scripts..."

cat > scripts/demo_animation.sh << 'EOF'
#!/bin/bash

# Demo script for PIFuHD Animation Pipeline
# Make sure you have a PIFuHD generated OBJ file to test with

echo "🎬 PIFuHD Animation Pipeline Demo"
echo "================================="

# Check if sample exists
if [[ ! -f "sample_images/test.png" ]]; then
    echo "❌ No sample image found. Please run PIFuHD reconstruction first:"
    echo "   sh scripts/demo.sh"
    exit 1
fi

# Run PIFuHD reconstruction if results don't exist
if [[ ! -f "results/test_256.obj" ]]; then
    echo "🔄 Running PIFuHD reconstruction first..."
    sh scripts/demo.sh
fi

# Run animation pipeline
if [[ -f "results/test_256.obj" ]]; then
    echo "🎭 Starting animation pipeline..."
    python -m apps.animate_model \
        --input results/test_256.obj \
        --output animations/test_character \
        --cleanup
    
    echo "✅ Demo completed! Check animations/test_character/ for results"
else
    echo "❌ No OBJ file found. Please run PIFuHD reconstruction first."
fi
EOF

chmod +x scripts/demo_animation.sh

# 10. Create Unity integration guide
cat > ANIMATION_SETUP.md << 'EOF'
# PIFuHD Animation Setup Guide

## Quick Start

1. **Generate 3D Model with PIFuHD:**
   ```bash
   sh scripts/demo.sh
   ```

2. **Create Rigged Character:**
   ```bash
   python -m apps.animate_model --input results/your_model.obj --output animations/character_name
   ```

3. **Unity Integration:**
   - Follow the instructions in `animations/character_name/digihuman_setup_instructions.json`
   - Import the rigged FBX into your DigiHuman Unity project
   - Configure character components as specified

## Pipeline Components

- **PIFuHD**: Generates high-quality 3D human models from single images
- **UniRig**: AI-powered automated rigging system
- **DigiHuman**: Real-time animation system with MediaPipe integration

## Troubleshooting

### UniRig Issues:
- Ensure CUDA is available (8GB+ GPU recommended)
- Check model downloads in `external/UniRig/`

### DigiHuman Issues:
- Verify Unity 2020.3.25f1+ is installed
- Ensure HTTP connections are enabled in Unity Player Settings
- Check MediaPipe installation: `pip install mediapipe`

### Common Problems:
- **Import errors**: Run `pip install -r requirements.txt`
- **GPU memory**: Reduce batch size or use CPU mode
- **File format issues**: Ensure OBJ files are valid and manifold

## Animation Features

✅ Full body pose estimation  
✅ Hand gesture recognition  
✅ Facial landmark detection  
✅ Real-time motion capture  
✅ Video export capabilities  
✅ Multiple character support  

EOF

# 11. Final checks and summary
print_success "PIFuHD Animation Pipeline setup completed!"
echo ""
echo "📋 Setup Summary:"
echo "=================="
echo "✅ Git submodules initialized (UniRig + DigiHuman)"
echo "✅ Python dependencies installed"
echo "✅ Output directories created"  
echo "✅ Scripts configured and executable"
echo "✅ Documentation generated"
echo ""
echo "🚀 Next Steps:"
echo "1. Generate a 3D model:     sh scripts/demo.sh"
echo "2. Test animation pipeline: sh scripts/demo_animation.sh"
echo "3. Read the guide:          cat ANIMATION_SETUP.md"
echo ""
echo "💡 For Unity integration, see external/DigiHuman/README.md"
print_success "Ready to animate your 3D models! 🎭"