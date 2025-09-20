#!/usr/bin/env python3
"""
PIFuHD Rigging Preparation
Prepares PIFuHD models for rigging and animation

This script:
1. Preprocesses PIFuHD OBJ files for optimal rigging
2. Converts to various formats (GLB, FBX)
3. Provides setup instructions for manual or automated rigging
4. Creates DigiHuman integration guide

Usage:
    python -m apps.prepare_for_rigging --input results/your_model.obj --output rigging_prep/
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root to path
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_PATH)

import trimesh
import numpy as np
from lib.mesh_util import save_obj_mesh


class PIFuHDRiggingPrep:
    """Prepare PIFuHD models for rigging and animation"""
    
    def __init__(self):
        self.temp_dir = Path("temp_rigging")
        self.temp_dir.mkdir(exist_ok=True)
        
    def preprocess_mesh(self, obj_path, output_dir):
        """
        Preprocess PIFuHD mesh for optimal rigging
        """
        print(f"🔄 Processing PIFuHD mesh: {obj_path}")
        
        # Load mesh
        mesh = trimesh.load(obj_path)
        vertices = mesh.vertices
        faces = mesh.faces
        
        print(f"   Original vertices: {len(vertices)}")
        print(f"   Original faces: {len(faces)}")
        
        # Center mesh at origin
        bbox_center = (vertices.max(axis=0) + vertices.min(axis=0)) * 0.5
        vertices -= bbox_center
        
        # Scale to standard human size (approximately 1.8m tall)
        bbox_size = vertices.max(axis=0) - vertices.min(axis=0)
        scale_factor = 1.8 / bbox_size[1]
        vertices *= scale_factor
        
        print(f"   Scaled by factor: {scale_factor:.3f}")
        print(f"   New height: {(vertices.max(axis=0) - vertices.min(axis=0))[1]:.3f}m")
        
        # Update mesh
        mesh.vertices = vertices
        
        # Repair mesh if needed
        if not mesh.is_watertight:
            print("   ⚠️  Mesh is not watertight, attempting repair...")
            mesh.fill_holes()
            if mesh.is_watertight:
                print("   ✅ Mesh repaired successfully")
            else:
                print("   ⚠️  Mesh repair incomplete - may need manual cleanup")
        
        # Check mesh quality
        print(f"   Watertight: {mesh.is_watertight}")
        print(f"   Volume: {mesh.volume:.6f}")
        
        # Save in multiple formats
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned OBJ
        clean_obj = output_dir / "clean_mesh.obj"
        save_obj_mesh(str(clean_obj), vertices, faces)
        
        # Save GLB for modern workflows
        clean_glb = output_dir / "clean_mesh.glb"
        mesh.export(str(clean_glb))
        
        # Try to save FBX if possible
        clean_fbx = output_dir / "clean_mesh.fbx"
        try:
            mesh.export(str(clean_fbx))
            print("   ✅ FBX export successful")
        except Exception as e:
            print(f"   ⚠️  FBX export failed: {e}")
            print("   💡 Consider using Blender for FBX conversion")
        
        print(f"   📁 Processed files saved to: {output_dir}")
        return output_dir, clean_obj, clean_glb
    
    def create_rigging_instructions(self, output_dir):
        """
        Create comprehensive rigging instructions
        """
        output_dir = Path(output_dir)
        
        # UniRig instructions
        unirig_guide = {
            "title": "UniRig Automated Rigging",
            "description": "AI-powered automatic skeleton generation and skinning",
            "requirements": [
                "CUDA-enabled GPU with 8GB+ VRAM",
                "PyTorch with CUDA support",
                "UniRig repository cloned and set up"
            ],
            "steps": [
                "1. Ensure UniRig is properly installed",
                "2. Download pre-trained models",
                "3. Run skeleton generation: bash external/UniRig/launch/inference/generate_skeleton.sh --input clean_mesh.glb --output skeleton.fbx",
                "4. Run skinning: bash external/UniRig/launch/inference/generate_skin.sh --input skeleton.fbx --output rigged_character.fbx"
            ],
            "expected_output": "Fully rigged FBX file with humanoid skeleton and skinning weights",
            "troubleshooting": [
                "If GPU memory insufficient, try CPU mode or reduce mesh resolution",
                "If model download fails, check network and storage space",
                "For best results, ensure input mesh is clean and manifold"
            ]
        }
        
        # Blender manual rigging instructions
        blender_guide = {
            "title": "Blender Manual/Semi-Automated Rigging",
            "description": "Professional rigging using Blender's Rigify addon",
            "requirements": [
                "Blender 3.0+",
                "Rigify addon enabled",
                "Basic Blender rigging knowledge"
            ],
            "steps": [
                "1. Import clean_mesh.obj or clean_mesh.glb into Blender",
                "2. Scale and position mesh appropriately",
                "3. Add Rigify Human armature (Add > Armature > Human Metarig)",
                "4. Scale and align metarig to match your character",
                "5. Generate final rig (Rigify buttons panel)",
                "6. Bind mesh to rig using Automatic Weights",
                "7. Test and refine weight painting as needed",
                "8. Export as FBX with rig and animations"
            ],
            "expected_output": "Professional-quality rigged character ready for animation",
            "tips": [
                "Use T-pose for best automatic weights",
                "Clean up weight painting for better deformation",
                "Add facial bones for expression animation",
                "Export with 'Apply Transform' enabled"
            ]
        }
        
        # DigiHuman integration guide
        digihuman_guide = {
            "title": "DigiHuman Unity Integration",
            "description": "Real-time animation with MediaPipe pose estimation",
            "requirements": [
                "Unity 2020.3.25f1 or newer",
                "DigiHuman project imported",
                "MediaPipe backend running"
            ],
            "character_setup": [
                "1. Import rigged FBX into Unity DigiHuman project",
                "2. In Import Settings, set Animation Type to 'Humanoid'",
                "3. Configure Avatar Definition as 'Create from this model'",
                "4. Apply and ensure T-pose is detected",
                "5. Drag character to CharacterChooser/CharacterSlideshow/Parent",
                "6. Add BlendShapeController component to character",
                "7. Add QualityData component",
                "8. Configure SkinnedMeshRenderer references",
                "9. Set up blendshape indices for facial animation",
                "10. Add character to CharacterSlideshow nodes array"
            ],
            "animation_features": [
                "Full body pose tracking via MediaPipe",
                "Hand gesture recognition",
                "Facial expression mapping",
                "Real-time motion capture",
                "Video recording capabilities"
            ],
            "backend_setup": [
                "1. Navigate to external/DigiHuman/Backend",
                "2. Install requirements: pip install -r requirements.txt",
                "3. Run server: python server.py",
                "4. Ensure Unity HTTP connections are enabled"
            ]
        }
        
        # Comparison guide
        comparison = {
            "rigging_methods": {
                "UniRig": {
                    "pros": ["Fully automated", "AI-powered", "Fast processing", "Consistent results"],
                    "cons": ["Requires powerful GPU", "Less control", "May need fine-tuning"],
                    "best_for": "Quick prototyping, batch processing, users new to rigging"
                },
                "Blender": {
                    "pros": ["Full control", "Professional quality", "Facial rigging", "Industry standard"],
                    "cons": ["Manual work required", "Learning curve", "Time-intensive"],
                    "best_for": "Production work, custom requirements, detailed characters"
                }
            },
            "recommended_workflow": [
                "1. Start with UniRig for base skeleton and skinning",
                "2. Import into Blender for refinements and facial rigging",
                "3. Export optimized FBX for DigiHuman Unity integration",
                "4. Test real-time animation and adjust as needed"
            ]
        }
        
        # Save all guides
        guides_file = output_dir / "rigging_guide.json"
        all_guides = {
            "unirig": unirig_guide,
            "blender": blender_guide,
            "digihuman": digihuman_guide,
            "comparison": comparison
        }
        
        with open(guides_file, 'w') as f:
            json.dump(all_guides, f, indent=2)
        
        # Create markdown summary
        readme_content = f"""# PIFuHD Rigging & Animation Guide

Your PIFuHD model has been preprocessed and is ready for rigging!

## 📁 Generated Files
- `clean_mesh.obj` - Cleaned and scaled OBJ file
- `clean_mesh.glb` - Modern GLB format for web/engines
- `clean_mesh.fbx` - FBX format (if export succeeded)
- `rigging_guide.json` - Detailed technical instructions

## 🎯 Quick Start Options

### Option 1: UniRig (Automated)
```bash
# From PIFuHD root directory
cd external/UniRig
bash launch/inference/generate_skeleton.sh --input ../../{output_dir.name}/clean_mesh.glb --output ../../{output_dir.name}/rigged_auto.fbx
```

### Option 2: Blender (Manual)
1. Open Blender
2. Import `clean_mesh.obj`
3. Add Rigify Human metarig
4. Align rig to character
5. Generate and bind rig
6. Export as FBX

### Option 3: Hybrid Approach
1. Use UniRig for base rigging
2. Import into Blender for refinements
3. Add facial bones and expressions
4. Export final character

## 🎮 DigiHuman Unity Integration

Once rigged:
1. Import FBX into DigiHuman Unity project
2. Set Animation Type to "Humanoid"
3. Add character components (see detailed guide)
4. Configure for real-time animation

## 🚀 Animation Features
- ✅ Full body pose estimation
- ✅ Hand gesture recognition  
- ✅ Facial expression mapping
- ✅ Real-time motion capture
- ✅ Video recording

## 📚 Need Help?
- Check `rigging_guide.json` for detailed instructions
- UniRig docs: https://github.com/VAST-AI-Research/UniRig
- DigiHuman setup: external/DigiHuman/README.md
- Blender Rigify: https://docs.blender.org/manual/en/latest/addons/rigging/rigify/

Happy animating! 🎭
"""
        
        readme_file = output_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print(f"📖 Rigging guides created:")
        print(f"   - {guides_file}")
        print(f"   - {readme_file}")
        
        return guides_file
    
    def create_test_scripts(self, output_dir):
        """
        Create test scripts for different rigging approaches
        """
        output_dir = Path(output_dir)
        
        # UniRig test script
        unirig_script = output_dir / "test_unirig.sh"
        with open(unirig_script, 'w') as f:
            f.write(f"""#!/bin/bash
# Test UniRig automated rigging
echo "🤖 Testing UniRig automated rigging..."

# Check if UniRig exists
if [ ! -d "../external/UniRig" ]; then
    echo "❌ UniRig not found. Run: git submodule update --init --recursive"
    exit 1
fi

# Test skeleton generation
echo "🦴 Generating skeleton..."
cd ../external/UniRig
python run.py \\
    --config configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml \\
    --input ../../{output_dir.name}/clean_mesh.glb \\
    --output ../../{output_dir.name}/test_skeleton.fbx

if [ $? -eq 0 ]; then
    echo "✅ Skeleton generation successful"
    
    # Test skinning
    echo "🎨 Generating skinning weights..."
    python run.py \\
        --config configs/task/quick_inference_unirig_skin.yaml \\
        --input ../../{output_dir.name}/test_skeleton.fbx \\
        --output ../../{output_dir.name}/test_rigged.fbx
    
    if [ $? -eq 0 ]; then
        echo "✅ Full rigging pipeline successful!"
        echo "📁 Check {output_dir.name}/test_rigged.fbx"
    else
        echo "❌ Skinning failed"
    fi
else
    echo "❌ Skeleton generation failed"
fi
""")
        
        os.chmod(unirig_script, 0o755)
        
        # Blender automation script
        blender_script = output_dir / "test_blender.py"
        with open(blender_script, 'w') as f:
            f.write(f"""import bpy
import bmesh
import os

# Blender script for semi-automated rigging
print("🎨 Starting Blender rigging automation...")

# Clear existing mesh
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import mesh
mesh_path = "{Path.cwd()}/{output_dir.name}/clean_mesh.obj"
bpy.ops.import_scene.obj(filepath=mesh_path)

# Get imported object
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# Add Rigify metarig
bpy.ops.object.armature_human_metarig_add()
armature = bpy.context.selected_objects[0]

# Scale armature to match character
armature.scale = (1.0, 1.0, 1.0)  # Adjust as needed

print("✅ Basic setup complete. Manual alignment and generation needed.")
print("💡 Next steps:")
print("1. Scale and align the metarig to your character")
print("2. Use Rigify 'Generate Rig' button")
print("3. Bind mesh with Automatic Weights")
print("4. Export as FBX")
""")
        
        print(f"🧪 Test scripts created:")
        print(f"   - {unirig_script}")
        print(f"   - {blender_script}")
        
        return unirig_script, blender_script
    
    def prepare_model(self, obj_path, output_dir):
        """
        Complete preparation workflow
        """
        print("🎭 PIFuHD Rigging Preparation Started")
        print("=" * 50)
        
        try:
            # Step 1: Preprocess mesh
            prep_dir, clean_obj, clean_glb = self.preprocess_mesh(obj_path, output_dir)
            
            # Step 2: Create instructions
            self.create_rigging_instructions(output_dir)
            
            # Step 3: Create test scripts
            self.create_test_scripts(output_dir)
            
            print("\n🎉 Preparation Complete!")
            print("=" * 50)
            print(f"📁 Output directory: {output_dir}")
            print("\n🚀 Next Steps:")
            print("1. Choose your rigging approach (see README.md)")
            print("2. Run test scripts to verify setup")
            print("3. Follow integration guide for DigiHuman")
            
            return output_dir
            
        except Exception as e:
            print(f"❌ Preparation failed: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Prepare PIFuHD model for rigging')
    parser.add_argument('--input', '-i', required=True, help='Input OBJ file from PIFuHD')
    parser.add_argument('--output', '-o', required=True, help='Output directory for prepared files')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return 1
    
    # Create preparation workflow
    prep = PIFuHDRiggingPrep()
    
    # Run preparation
    result = prep.prepare_model(args.input, args.output)
    
    if result:
        print(f"\n✅ Preparation completed successfully!")
        print(f"📖 Read {result}/README.md for next steps")
        return 0
    else:
        print("❌ Preparation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())