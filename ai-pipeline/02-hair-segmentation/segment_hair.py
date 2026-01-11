"""
Production-Grade Hair Segmentation Module
Uses color-based segmentation (BiSeNet model can be added later)

This module creates:
- face_mask.png (only face, no hair, no background)
- hair_mask.png (only hair)
- Masked photos ready for photogrammetry
"""

import os
import cv2
import numpy as np
from pathlib import Path


class HairSegmenter:
    """
    Hair Segmentation using color-based approach
    
    Output:
    - Face mask: skin + facial features (no hair)
    - Hair mask: hair regions only
    """
    
    def __init__(self):
        print("[HairSegmenter] Initialized (color-based mode)")
    
    def segment_color_based(self, image):
        """
        Color-based segmentation
        Separates face from hair using skin detection
        """
        h, w = image.shape[:2]
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # ===== SKIN DETECTION =====
        # YCrCb is better for skin detection
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Clean up skin mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Fill holes in skin mask
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Keep only largest contour (main face)
            largest_contour = max(contours, key=cv2.contourArea)
            skin_mask_filled = np.zeros_like(skin_mask)
            cv2.drawContours(skin_mask_filled, [largest_contour], -1, 255, -1)
            skin_mask = skin_mask_filled
        
        # ===== HAIR DETECTION =====
        # Hair is typically dark
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, dark_mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        
        # Hair = dark regions AND not skin
        hair_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(skin_mask))
        
        # Clean up hair mask
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
        
        # ===== FACE MASK REFINEMENT =====
        # Expand face mask slightly to include edges
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        face_mask = cv2.dilate(skin_mask, kernel_dilate, iterations=2)
        
        # Ensure hair doesn't overlap with face
        hair_mask = cv2.bitwise_and(hair_mask, cv2.bitwise_not(face_mask))
        
        return face_mask, hair_mask
    
    def segment(self, image_path, output_dir=None, save_visualization=True):
        """
        Main segmentation function
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save masks (default: same as input)
            save_visualization: Whether to save visualization images
        
        Returns:
            face_mask_path, hair_mask_path
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"[HairSegmenter] Processing: {Path(image_path).name}")
        print(f"[HairSegmenter] Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Segment
        face_mask, hair_mask = self.segment_color_based(image)
        
        # Prepare output paths
        if output_dir is None:
            output_dir = Path(image_path).parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(image_path).stem
        
        face_mask_path = output_dir / f"{base_name}_face_mask.png"
        hair_mask_path = output_dir / f"{base_name}_hair_mask.png"
        
        # Save masks
        cv2.imwrite(str(face_mask_path), face_mask)
        cv2.imwrite(str(hair_mask_path), hair_mask)
        
        print(f"[HairSegmenter] ✅ Face mask: {face_mask_path.name}")
        print(f"[HairSegmenter] ✅ Hair mask: {hair_mask_path.name}")
        
        # Save visualizations
        if save_visualization:
            # Masked image (only face visible, rest black)
            masked_image = image.copy()
            masked_image[face_mask == 0] = 0
            masked_path = output_dir / f"{base_name}_face_only.jpg"
            cv2.imwrite(str(masked_path), masked_image)
            
            # Overlay visualization
            overlay = image.copy()
            # Green = face, Red = hair
            overlay[face_mask > 0] = cv2.addWeighted(
                overlay[face_mask > 0], 0.7, 
                np.array([0, 255, 0], dtype=np.uint8), 0.3, 0
            )
            overlay[hair_mask > 0] = cv2.addWeighted(
                overlay[hair_mask > 0], 0.7, 
                np.array([0, 0, 255], dtype=np.uint8), 0.3, 0
            )
            overlay_path = output_dir / f"{base_name}_overlay.jpg"
            cv2.imwrite(str(overlay_path), overlay)
        
        return str(face_mask_path), str(hair_mask_path)


def main():
    """CLI interface for hair segmentation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hair Segmentation for 3D Reconstruction')
    parser.add_argument('--input', '-i', required=True, help='Input image or directory')
    parser.add_argument('--output', '-o', default=None, help='Output directory')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization images')
    
    args = parser.parse_args()
    
    # Initialize segmenter
    segmenter = HairSegmenter()
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        segmenter.segment(
            input_path, 
            output_dir=args.output,
            save_visualization=not args.no_viz
        )
    elif input_path.is_dir():
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
        
        print(f"[INFO] Found {len(images)} images in {input_path}")
        
        for i, img_path in enumerate(images, 1):
            print(f"\n[{i}/{len(images)}]")
            try:
                segmenter.segment(
                    img_path,
                    output_dir=args.output,
                    save_visualization=not args.no_viz
                )
            except Exception as e:
                print(f"[ERROR] Failed to process {img_path}: {e}")
    else:
        print(f"[ERROR] Invalid input path: {input_path}")
        return 1
    
    print("\n[INFO] ✅ Segmentation complete!")
    return 0


if __name__ == "__main__":
    exit(main())