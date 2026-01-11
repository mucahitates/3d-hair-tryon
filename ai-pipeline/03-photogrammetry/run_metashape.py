"""
Metashape CLI Wrapper - FIXED for Metashape 2.x API
Runs Metashape via command-line or GUI Console
"""

import os
import subprocess
from pathlib import Path
import time


METASHAPE_SCRIPT_TEMPLATE = """
import Metashape

# Project paths
photos_dir = r"{photos_dir}"
project_path = r"{project_path}"
output_path = r"{output_path}"

print("[Metashape Script] Starting reconstruction...")

# Create document
doc = Metashape.Document()
doc.save(project_path)

# Add chunk
chunk = doc.addChunk()
chunk.label = "Head Reconstruction"

# Collect photos
import os
photos = [os.path.join(photos_dir, f) for f in os.listdir(photos_dir) 
          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"[Metashape Script] Adding {{len(photos)}} photos...")
chunk.addPhotos(photos)
doc.save()

print(f"[Metashape Script] {{len(chunk.cameras)}} cameras added")

# Match photos
print("[Metashape Script] Matching photos...")
chunk.matchPhotos(
    downscale=1,
    generic_preselection=True,
    reference_preselection=False,
    filter_mask=False,
    keypoint_limit=40000,
    tiepoint_limit=4000
)

# Align cameras
print("[Metashape Script] Aligning cameras...")
chunk.alignCameras()
doc.save()

aligned = sum([1 for camera in chunk.cameras if camera.transform])
print(f"[Metashape Script] {{aligned}}/{{len(chunk.cameras)}} cameras aligned")

if aligned < len(chunk.cameras) * 0.5:
    print("[Metashape Script] ERROR: Alignment failed")
    exit(1)

# Build depth maps
print("[Metashape Script] Building depth maps...")
chunk.buildDepthMaps(downscale=2, filter_mode=Metashape.MildFiltering)

# Build point cloud (Metashape 2.x API)
print("[Metashape Script] Building point cloud...")
chunk.buildPointCloud(point_colors=True, point_confidence=True)
doc.save()

# Get point count
print(f"[Metashape Script] Point cloud generated")

# Build mesh (using PointCloudData for Metashape 2.x)
print("[Metashape Script] Building mesh...")
chunk.buildModel(
    surface_type=Metashape.Arbitrary,
    interpolation=Metashape.EnabledInterpolation,
    face_count=Metashape.HighFaceCount,
    source_data=Metashape.PointCloudData,
    vertex_colors=True
)
doc.save()

print(f"[Metashape Script] Mesh: {{len(chunk.model.faces)}} faces")

# Build texture
print("[Metashape Script] Building texture...")
chunk.buildUV(mapping_mode=Metashape.GenericMapping, page_count=1)
chunk.buildTexture(blending_mode=Metashape.MosaicBlending, texture_size=4096)
doc.save()

# Export
print("[Metashape Script] Exporting model...")
chunk.exportModel(
    path=output_path,
    format=Metashape.ModelFormatOBJ,
    texture_format=Metashape.ImageFormatJPEG,
    save_texture=True,
    save_normals=True,
    save_colors=True,
    embed_texture=False
)

print(f"[Metashape Script] COMPLETE: {{output_path}}")
"""


def run_metashape_reconstruction(photos_dir, project_dir, output_dir, 
                                 metashape_exe="C:/Program Files/Agisoft/Metashape Pro/metashape.exe"):
    """
    Run Metashape reconstruction via command-line
    
    Args:
        photos_dir: Directory with masked photos
        project_dir: Directory for Metashape project
        output_dir: Directory for output model
        metashape_exe: Path to metashape.exe
    
    Returns:
        output_model_path or None
    """
    
    photos_dir = Path(photos_dir).resolve()
    project_dir = Path(project_dir).resolve()
    output_dir = Path(output_dir).resolve()
    
    # Create directories
    project_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths
    project_path = project_dir / "reconstruction.psx"
    output_path = output_dir / "bald_head_model.obj"
    script_path = project_dir / "metashape_script.py"
    log_path = project_dir / "metashape_log.txt"
    
    # Check Metashape
    if not Path(metashape_exe).exists():
        print(f"[ERROR] Metashape not found: {metashape_exe}")
        return None
    
    # Check photos
    photos = list(photos_dir.glob("*.jpg")) + list(photos_dir.glob("*.jpeg")) + list(photos_dir.glob("*.png"))
    if len(photos) < 20:
        print(f"[ERROR] Insufficient photos: {len(photos)} < 20")
        return None
    
    print(f"[MetashapeWrapper] Photos: {len(photos)}")
    print(f"[MetashapeWrapper] Project: {project_path}")
    print(f"[MetashapeWrapper] Output: {output_path}")
    
    # Generate script
    script_content = METASHAPE_SCRIPT_TEMPLATE.format(
        photos_dir=str(photos_dir),
        project_path=str(project_path),
        output_path=str(output_path)
    )
    
    script_path.write_text(script_content, encoding='utf-8')
    print(f"[MetashapeWrapper] Script created: {script_path}")
    
    # Run Metashape
    print("[MetashapeWrapper] Starting Metashape...")
    print("[MetashapeWrapper] This may take 10-30 minutes...")
    
    cmd = [
        str(metashape_exe),
        "-r", str(script_path),
        "--log", str(log_path)
    ]
    
    try:
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n[MetashapeWrapper] Metashape finished in {elapsed:.1f} seconds")
        
        # Check output
        if output_path.exists():
            print(f"[MetashapeWrapper] ✅ Model created: {output_path}")
            print(f"[MetashapeWrapper] Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
            return str(output_path)
        else:
            print("[MetashapeWrapper] ❌ Model not created")
            print(f"[MetashapeWrapper] Check log: {log_path}")
            
            if log_path.exists():
                print("\n=== METASHAPE LOG (last 2000 chars) ===")
                print(log_path.read_text()[-2000:])
                print("=" * 60)
            
            return None
    
    except subprocess.TimeoutExpired:
        print("[MetashapeWrapper] ❌ Metashape timeout (1 hour)")
        return None
    
    except Exception as e:
        print(f"[MetashapeWrapper] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Metashape CLI Wrapper')
    parser.add_argument('--photos', '-p', required=True, help='Photos directory')
    parser.add_argument('--project', '-pr', required=True, help='Project directory')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--metashape-exe', default="C:/Program Files/Agisoft/Metashape Pro/metashape.exe")
    
    args = parser.parse_args()
    
    result = run_metashape_reconstruction(
        photos_dir=args.photos,
        project_dir=args.project,
        output_dir=args.output,
        metashape_exe=args.metashape_exe
    )
    
    if result:
        print(f"\n✅ SUCCESS: {result}")
        exit(0)
    else:
        print("\n❌ FAILED")
        exit(1)


if __name__ == "__main__":
    main()