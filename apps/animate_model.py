#!/usr/bin/env python3
"""
PIFuHD Animation Pipeline
Integrates PIFuHD, UniRig, and DigiHuman for automated 3D model rigging and animation

This script provides a complete pipeline:
1. Take PIFuHD generated OBJ files
2. Use UniRig for automated rigging with humanoid skeleton
3. Export in format compatible with DigiHuman animation system
4. Prepare for real-time animation with MediaPipe pose estimation

Usage:
    python -m apps.animate_model --input results/sample.obj --output animations/
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

# Add project root to path
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_PATH)

import trimesh
import numpy as np
from lib.mesh_util import save_obj_mesh


class PIFuHDAnimationPipeline:
    """Complete animation pipeline for PIFuHD generated meshes"""
    
    def __init__(self, unirig_path="external/UniRig", digihuman_path="external/DigiHuman"):
        self.unirig_path = Path(ROOT_PATH) / unirig_path
        self.digihuman_path = Path(ROOT_PATH) / digihuman_path
        self.temp_dir = Path("temp_animation")
        self.temp_dir.mkdir(exist_ok=True)
        
    def validate_dependencies(self):
        """Check if UniRig and DigiHuman are properly set up"""
        if not self.unirig_path.exists():
            raise FileNotFoundError(f"UniRig not found at {self.unirig_path}")
        if not self.digihuman_path.exists():
            raise FileNotFoundError(f"DigiHuman not found at {self.digihuman_path}")
        
        # Check UniRig requirements
        unirig_requirements = self.unirig_path / "requirements.txt"
        if not unirig_requirements.exists():
            print(f"Warning: UniRig requirements.txt not found")
            
        return True
    
    def preprocess_pifuhd_mesh(self, obj_path, output_path):
        """
        Preprocess PIFuHD mesh for rigging
        - Clean mesh geometry
        - Ensure proper orientation (T-pose if possible)
        - Scale normalization
        """
        print(f"Preprocessing PIFuHD mesh: {obj_path}")
        
        # Load mesh using trimesh
        mesh = trimesh.load(obj_path)
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Center mesh
        bbox_center = (vertices.max(axis=0) + vertices.min(axis=0)) * 0.5
        vertices -= bbox_center
        
        # Scale to standard size (DigiHuman expects ~2m height)
        bbox_size = vertices.max(axis=0) - vertices.min(axis=0)
        scale_factor = 2.0 / bbox_size[1]  # Scale to 2m height
        vertices *= scale_factor
        
        # Ensure mesh is manifold and watertight
        mesh.vertices = vertices
        if not mesh.is_watertight:
            print("Warning: Mesh is not watertight, attempting repair...")
            mesh.fill_holes()
        
        # Export as both OBJ and GLB for UniRig
        obj_output = self.temp_dir / "preprocessed.obj"
        glb_output = self.temp_dir / "preprocessed.glb"
        
        # Save OBJ
        save_obj_mesh(str(obj_output), vertices, faces)
        
        # Save GLB using trimesh
        mesh.export(str(glb_output))
        
        print(f"Preprocessed mesh saved to {obj_output} and {glb_output}")
        return str(glb_output), str(obj_output)
    
    def generate_skeleton_with_unirig(self, glb_path, output_path):
        """
        Generate skeleton using UniRig's AI-powered rigging
        """
        print(f"Generating skeleton with UniRig for: {glb_path}")
        
        # Prepare UniRig command
        generate_script = self.unirig_path / "launch/inference/generate_skeleton.sh"
        
        if not generate_script.exists():
            # Use Python interface instead
            cmd = [
                sys.executable, 
                str(self.unirig_path / "run.py"),
                "--config", str(self.unirig_path / "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"),
                "--input", glb_path,
                "--output", output_path
            ]
        else:
            # Use bash script
            cmd = ["bash", str(generate_script), "--input", glb_path, "--output", output_path]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.unirig_path))
            if result.returncode != 0:
                print(f"UniRig error: {result.stderr}")
                return None
            print(f"Skeleton generated successfully: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error running UniRig: {e}")
            return None
    
    def generate_skinning_weights(self, skeleton_path, output_path):
        """
        Generate skinning weights using UniRig
        """
        print(f"Generating skinning weights for: {skeleton_path}")
        
        # Use UniRig skin generation
        generate_script = self.unirig_path / "launch/inference/generate_skin.sh"
        
        if not generate_script.exists():
            cmd = [
                sys.executable,
                str(self.unirig_path / "run.py"),
                "--config", str(self.unirig_path / "configs/task/quick_inference_unirig_skin.yaml"),
                "--input", skeleton_path,
                "--output", output_path
            ]
        else:
            cmd = ["bash", str(generate_script), "--input", skeleton_path, "--output", output_path]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.unirig_path))
            if result.returncode != 0:
                print(f"UniRig skinning error: {result.stderr}")
                return None
            print(f"Skinning weights generated: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error generating skinning weights: {e}")
            return None
    
    def convert_to_digihuman_format(self, rigged_model_path, output_dir):
        """
        Convert rigged model to DigiHuman compatible format
        DigiHuman expects:
        - Humanoid T-Pose rig
        - FBX format with proper bone hierarchy
        - BlendShape support for facial animation
        """
        print(f"Converting to DigiHuman format: {rigged_model_path}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy rigged model to output directory
        output_file = output_dir / "rigged_character.fbx"
        
        if Path(rigged_model_path).suffix.lower() == '.fbx':
            import shutil
            shutil.copy2(rigged_model_path, output_file)
        else:
            # Convert GLB to FBX if needed (requires Blender or similar)
            print("Note: Manual conversion to FBX may be required for full DigiHuman compatibility")
            print("Consider using Blender to convert GLB to FBX with proper humanoid rig")
        
        # Generate DigiHuman integration instructions
        instructions = {
            "character_setup": {
                "model_path": str(output_file),
                "rig_type": "Humanoid",
                "pose_type": "T-Pose",
                "requirements": [
                    "Set rig to Humanoid in Unity Import Settings",
                    "Ensure T-Pose configuration",
                    "Add BlendShapeController component",
                    "Add QualityData component",
                    "Configure SkinnedMeshRenderer references"
                ]
            },
            "integration_steps": [
                "1. Import FBX into Unity DigiHuman project",
                "2. Set Animation Type to 'Humanoid' in Import Settings",
                "3. Apply T-Pose configuration",
                "4. Drag model to CharacterChooser/CharacterSlideshow/Parent",
                "5. Add required components (BlendShapeController, QualityData)",
                "6. Configure blendshape indices for facial animation",
                "7. Add character reference to CharacterSlideshow nodes array"
            ],
            "animation_features": [
                "Full body pose animation via MediaPipe",
                "Hand gesture recognition",
                "Facial landmark animation (if blendshapes available)",
                "Real-time motion capture integration"
            ]
        }
        
        # Save instructions
        instructions_file = output_dir / "digihuman_setup_instructions.json"
        with open(instructions_file, 'w') as f:
            json.dump(instructions, f, indent=2)
        
        print(f"DigiHuman setup instructions saved to: {instructions_file}")
        return output_dir
    
    def create_animation_workflow(self, obj_path, output_dir):
        """
        Complete workflow: PIFuHD OBJ -> Rigged Model -> DigiHuman Ready
        """
        print("=== PIFuHD Animation Pipeline Started ===")
        
        try:
            # Step 1: Validate dependencies
            self.validate_dependencies()
            
            # Step 2: Preprocess PIFuHD mesh
            glb_path, obj_path_processed = self.preprocess_pifuhd_mesh(obj_path, self.temp_dir)
            
            # Step 3: Generate skeleton with UniRig
            skeleton_output = self.temp_dir / "skeleton.fbx"
            skeleton_path = self.generate_skeleton_with_unirig(glb_path, str(skeleton_output))
            
            if not skeleton_path:
                print("Failed to generate skeleton. Check UniRig installation.")
                return None
            
            # Step 4: Generate skinning weights
            skinned_output = self.temp_dir / "rigged_model.fbx"
            rigged_model = self.generate_skinning_weights(skeleton_path, str(skinned_output))
            
            if not rigged_model:
                print("Failed to generate skinning weights.")
                return None
            
            # Step 5: Convert to DigiHuman format
            final_output = self.convert_to_digihuman_format(rigged_model, output_dir)
            
            print("=== PIFuHD Animation Pipeline Completed Successfully ===")
            print(f"Output directory: {final_output}")
            print("Next steps:")
            print("1. Follow the instructions in digihuman_setup_instructions.json")
            print("2. Import the rigged model into Unity DigiHuman project")
            print("3. Configure the character following DigiHuman guidelines")
            print("4. Test real-time animation with MediaPipe pose estimation")
            
            return final_output
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            return None
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print("Temporary files cleaned up")


def main():
    parser = argparse.ArgumentParser(description='PIFuHD Animation Pipeline')
    parser.add_argument('--input', '-i', required=True, help='Input OBJ file from PIFuHD')
    parser.add_argument('--output', '-o', required=True, help='Output directory for animated model')
    parser.add_argument('--cleanup', action='store_true', help='Clean up temporary files after processing')
    parser.add_argument('--unirig_path', default='external/UniRig', help='Path to UniRig installation')
    parser.add_argument('--digihuman_path', default='external/DigiHuman', help='Path to DigiHuman installation')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Create pipeline
    pipeline = PIFuHDAnimationPipeline(args.unirig_path, args.digihuman_path)
    
    # Run workflow
    result = pipeline.create_animation_workflow(args.input, args.output)
    
    if result:
        print(f"✅ Animation pipeline completed successfully!")
        print(f"📁 Output: {result}")
        
        if args.cleanup:
            pipeline.cleanup_temp_files()
        
        return 0
    else:
        print("❌ Animation pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())