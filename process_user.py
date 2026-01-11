"""
User Processing Pipeline
Handles complete workflow for a single user:
1. Create user directory structure
2. Process all photos (hair segmentation)
3. Run photogrammetry (Metashape)
4. Generate 3D model
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json
import logging


# Setup logging
def setup_logging(user_id):
    """Setup logging for user processing"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{user_id}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class UserProcessor:
    """
    Main processing pipeline for user photos
    """
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.logger = setup_logging(user_id)
        
        # Paths
        self.data_dir = Path("data/users") / user_id
        self.raw_photos_dir = self.data_dir / "raw_photos"
        self.masked_photos_dir = self.data_dir / "masked_photos"
        self.masks_dir = self.data_dir / "masks"
        self.metashape_dir = self.data_dir / "metashape_project"
        self.output_dir = self.data_dir / "output"
        
        # Status tracking
        self.status = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "steps": {
                "1_structure_created": False,
                "2_photos_validated": False,
                "3_segmentation_complete": False,
                "4_photogrammetry_complete": False,
                "5_model_ready": False
            },
            "photo_count": 0,
            "errors": []
        }
    
    def create_structure(self):
        """Create user directory structure"""
        self.logger.info(f"Creating directory structure for {self.user_id}")
        
        try:
            # Create all directories
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.raw_photos_dir.mkdir(exist_ok=True)
            self.masked_photos_dir.mkdir(exist_ok=True)
            self.masks_dir.mkdir(exist_ok=True)
            self.metashape_dir.mkdir(exist_ok=True)
            self.output_dir.mkdir(exist_ok=True)
            
            # Create README
            readme_content = f"""
# User: {self.user_id}
Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Directory Structure:
- raw_photos/       : Original photos (30-35 images)
- masked_photos/    : Processed photos (hair removed)
- masks/            : Face and hair masks
- metashape_project/: Metashape working files
- output/           : Final 3D model

## Workflow:
1. Upload photos to raw_photos/
2. Run: python process_user.py --user {self.user_id} --step segment
3. Run: python process_user.py --user {self.user_id} --step photogrammetry
4. Check output/ for final model
"""
            (self.data_dir / "README.md").write_text(readme_content)
            
            self.status["steps"]["1_structure_created"] = True
            self.save_status()
            
            self.logger.info(f"Directory structure created: {self.data_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create structure: {e}")
            self.status["errors"].append(str(e))
            self.save_status()
            return False
    
    def validate_photos(self, min_photos=30, max_photos=50):
        """Validate that sufficient photos exist"""
        self.logger.info("Validating photos...")
        
        try:
            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png'}
            photos = [f for f in self.raw_photos_dir.iterdir() 
                     if f.suffix.lower() in image_extensions]
            
            photo_count = len(photos)
            self.status["photo_count"] = photo_count
            
            self.logger.info(f"Found {photo_count} photos")
            
            if photo_count < min_photos:
                error_msg = f"Insufficient photos: {photo_count} < {min_photos}"
                self.logger.error(f"{error_msg}")
                self.status["errors"].append(error_msg)
                self.save_status()
                return False
            
            if photo_count > max_photos:
                self.logger.warning(f"More than {max_photos} photos, using first {max_photos}")
                photos = photos[:max_photos]
            
            # Check photo quality (resolution, corruption)
            valid_photos = []
            for photo in photos:
                try:
                    import cv2
                    img = cv2.imread(str(photo))
                    if img is None:
                        self.logger.warning(f"Corrupted image: {photo.name}")
                        continue
                    
                    h, w = img.shape[:2]
                    if w < 800 or h < 800:
                        self.logger.warning(f"Low resolution: {photo.name} ({w}x{h})")
                        continue
                    
                    valid_photos.append(photo)
                    
                except Exception as e:
                    self.logger.warning(f"Cannot read {photo.name}: {e}")
            
            if len(valid_photos) < min_photos:
                error_msg = f"Only {len(valid_photos)} valid photos after quality check"
                self.logger.error(f"{error_msg}")
                self.status["errors"].append(error_msg)
                self.save_status()
                return False
            
            self.status["photo_count"] = len(valid_photos)
            self.status["steps"]["2_photos_validated"] = True
            self.save_status()
            
            self.logger.info(f"Validation passed: {len(valid_photos)} valid photos")
            return True
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            self.status["errors"].append(str(e))
            self.save_status()
            return False
    
    def run_segmentation(self):
        """Run hair segmentation on all photos"""
        self.logger.info("Starting hair segmentation...")
        
        try:
            # Import segmentation module
            sys.path.insert(0, str(Path("ai-pipeline/02-hair-segmentation")))
            from segment_hair import HairSegmenter
            
            # Initialize segmenter
            segmenter = HairSegmenter()
            
            # Get all photos
            image_extensions = {'.jpg', '.jpeg', '.png'}
            photos = [f for f in self.raw_photos_dir.iterdir() 
                     if f.suffix.lower() in image_extensions]
            
            self.logger.info(f"Processing {len(photos)} photos...")
            
            # Process each photo
            processed_count = 0
            failed_count = 0
            
            for i, photo in enumerate(photos, 1):
                try:
                    self.logger.info(f"[{i}/{len(photos)}] Processing {photo.name}")
                    
                    # Segment
                    face_mask_path, hair_mask_path = segmenter.segment(
                        photo,
                        output_dir=self.masks_dir,
                        save_visualization=False
                    )
                    
                    # Create masked image (only face, hair removed)
                    import cv2
                    img = cv2.imread(str(photo))
                    face_mask = cv2.imread(face_mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    # Apply mask
                    masked_img = img.copy()
                    masked_img[face_mask == 0] = 0  # Black out non-face regions
                    
                    # Save masked image
                    masked_path = self.masked_photos_dir / f"{photo.stem}_masked{photo.suffix}"
                    cv2.imwrite(str(masked_path), masked_img)
                    
                    processed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {photo.name}: {e}")
                    failed_count += 1
            
            if processed_count == 0:
                error_msg = "No photos were successfully processed"
                self.logger.error(f"{error_msg}")
                self.status["errors"].append(error_msg)
                self.save_status()
                return False
            
            self.status["steps"]["3_segmentation_complete"] = True
            self.save_status()
            
            self.logger.info(f"Segmentation complete: {processed_count} processed, {failed_count} failed")
            return True
            
        except Exception as e:
            self.logger.error(f"Segmentation failed: {e}")
            self.status["errors"].append(str(e))
            self.save_status()
            import traceback
            traceback.print_exc()
            return False
    
    def run_photogrammetry(self):
        """Run Metashape photogrammetry"""
        self.logger.info("Starting photogrammetry...")
        
        try:
            # Import Metashape wrapper
            sys.path.insert(0, str(Path("ai-pipeline/03-photogrammetry")))
            from run_metashape import run_metashape_reconstruction
            
            # Run reconstruction
            self.logger.info("Running Metashape reconstruction...")
            output_model = run_metashape_reconstruction(
                photos_dir=str(self.masked_photos_dir),
                project_dir=str(self.metashape_dir),
                output_dir=str(self.output_dir)
            )
            
            if output_model is None:
                error_msg = "Photogrammetry failed"
                self.logger.error(f"{error_msg}")
                self.status["errors"].append(error_msg)
                self.save_status()
                return False
            
            # Success
            self.status["steps"]["4_photogrammetry_complete"] = True
            self.status["steps"]["5_model_ready"] = True
            self.save_status()
            
            self.logger.info(f"Model ready: {output_model}")
            return True
            
        except Exception as e:
            self.logger.error(f"Photogrammetry failed: {e}")
            self.status["errors"].append(str(e))
            self.save_status()
            import traceback
            traceback.print_exc()
            return False
    
    def save_status(self):
        """Save processing status to JSON"""
        status_file = self.data_dir / "status.json"
        try:
            with open(status_file, 'w') as f:
                json.dump(self.status, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save status: {e}")
    
    def load_status(self):
        """Load processing status from JSON"""
        status_file = self.data_dir / "status.json"
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    self.status = json.load(f)
                return True
            except Exception as e:
                self.logger.error(f"Failed to load status: {e}")
        return False
    
    def get_status(self):
        """Get current processing status"""
        return self.status


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process user photos for 3D reconstruction')
    parser.add_argument('--user', '-u', required=True, help='User ID (e.g., user_001)')
    parser.add_argument('--step', '-s', 
                       choices=['init', 'validate', 'segment', 'photogrammetry', 'all'],
                       default='all',
                       help='Processing step to run')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = UserProcessor(args.user)
    
    # Load existing status if available
    processor.load_status()
    
    print("\n" + "=" * 60)
    print(f"PROCESSING USER: {args.user}")
    print("=" * 60 + "\n")
    
    # Execute requested step
    if args.step == 'init' or args.step == 'all':
        if not processor.create_structure():
            print("\nFailed at: Structure creation")
            sys.exit(1)
    
    if args.step == 'validate' or args.step == 'all':
        if not processor.validate_photos():
            print("\nFailed at: Photo validation")
            sys.exit(1)
    
    if args.step == 'segment' or args.step == 'all':
        if not processor.run_segmentation():
            print("\nFailed at: Segmentation")
            sys.exit(1)
    
    if args.step == 'photogrammetry' or args.step == 'all':
        if not processor.run_photogrammetry():
            print("\nFailed at: Photogrammetry")
            sys.exit(1)
    
    # Print final status
    print("\n" + "=" * 60)
    print("PROCESSING STATUS")
    print("=" * 60)
    
    status = processor.get_status()
    for step, completed in status["steps"].items():
        icon = "OK" if completed else "PENDING"
        print(f"[{icon}] {step}")
    
    if status["errors"]:
        print(f"\nErrors: {len(status['errors'])}")
        for error in status["errors"]:
            print(f"   - {error}")
    
    print()


if __name__ == "__main__":
    main()