"""
Production-Grade Hair Segmentation Module
Uses BiSeNet for accurate face parsing and hair extraction
Fallback to color-based segmentation if model not available
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path


class HairSegmenter:
    """
    Hair Segmentation using BiSeNet Face Parsing
    
    Face Parsing Labels (BiSeNet):
    0: background, 1: skin, 2-5: eyes/eyebrows, 7-8: ears, 
    10: nose, 11-13: mouth, 14: neck, 17: hair
    
    Output:
    - Face mask: Only face parts (no hair, no background)
    - Hair mask: Only hair regions
    """
    
    # Face parts (no hair, no background)
    FACE_LABELS = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]
    HAIR_LABELS = [17]
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.use_model = False
        print(f"[HairSegmenter] Device: {device}")
        
        # Try to load BiSeNet model
        self.load_model()
    
    def load_model(self):
        """Load BiSeNet model if weights exist"""
        model_path = Path(__file__).parent / "models" / "weights" / "79999_iter.pth"
        
        if not model_path.exists():
            print(f"[HairSegmenter] Model not found: {model_path}")
            print("[HairSegmenter] Using color-based segmentation")
            return False
        
        try:
            # Import BiSeNet
            sys.path.insert(0, str(Path(__file__).parent / "models"))
            from bisenet import BiSeNet
            
            # Load model
            self.model = BiSeNet(n_classes=19)
            self.model.load_state_dict(torch.load(str(model_path), map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.use_model = True
            
            print(f"[HairSegmenter] ✅ BiSeNet model loaded")
            return True
            
        except Exception as e:
            print(f"[HairSegmenter] ⚠️ Could not load BiSeNet: {e}")
            print("[HairSegmenter] Using color-based segmentation")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for BiSeNet"""
        # Resize to 512x512
        img_resized = cv2.resize(image, (512, 512))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img_normalized = img_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_normalized - mean) / std
        
        # Convert to tensor [1, 3, 512, 512]
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def segment_with_model(self, image):
        """Segment using BiSeNet model"""
        h, w = image.shape[:2]
        
        # Preprocess
        img_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            output = self.model(img_tensor)[0]  # BiSeNet returns (out, out16, out32)
            parsing = output.squeeze(0).cpu().numpy().argmax(0)
        
        # Resize back to original size
        parsing = cv2.resize(parsing.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Create face mask (only face parts, no hair)
        face_mask = np.isin(parsing, self.FACE_LABELS).astype(np.uint8) * 255
        
        # Create hair mask
        hair_mask = np.isin(parsing, self.HAIR_LABELS).astype(np.uint8) * 255
        
        return face_mask, hair_mask
    
    def segment_color_based(self, image):
        """
        Fallback: Color-based segmentation
        Less accurate but works without model
        """
        h, w = image.shape[:2]
        
        # Convert to different color spaces
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Skin detection
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Clean up skin mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            skin_mask_filled = np.zeros_like(skin_mask)
            cv2.drawContours(skin_mask_filled, [largest_contour], -1, 255, -1)
            skin_mask = skin_mask_filled
        
        # Hair detection (dark regions)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, dark_mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        
        # Hair = dark AND not skin
        hair_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(skin_mask))
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
        
        # Face mask = skin expanded
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        face_mask = cv2.dilate(skin_mask, kernel_dilate, iterations=2)
        
        # Ensure no overlap
        hair_mask = cv2.bitwise_and(hair_mask, cv2.bitwise_not(face_mask))
        
        return face_mask, hair_mask
    
    def segment(self, image_path, output_dir=None, save_visualization=True):
        """
        Main segmentation function
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save masks
            save_visualization: Whether to save visualization
        
        Returns:
            face_mask_path, hair_mask_path
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"[HairSegmenter] Processing: {Path(image_path).name}")
        
        # Segment (model or fallback)
        if self.use_model:
            face_mask, hair_mask = self.segment_with_model(image)
        else:
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
        
        # Save visualizations
        if save_visualization:
            # Masked image (only face, rest black)
            masked_image = image.copy()
            masked_image[face_mask >0] = 0
            masked_path = output_dir / f"{base_name}_masked.jpg"
            cv2.imwrite(str(masked_path), masked_image)
            
            # Overlay (green=face, red=hair)
            overlay = image.copy()
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
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hair Segmentation with BiSeNet')
    parser.add_argument('--input', '-i', required=True, help='Input image or directory')
    parser.add_argument('--output', '-o', default=None, help='Output directory')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualizations')
    
    args = parser.parse_args()
    
    # Initialize segmenter
    segmenter = HairSegmenter()
    
    # Process
    input_path = Path(args.input)
    
    if input_path.is_file():
        segmenter.segment(input_path, output_dir=args.output, save_visualization=not args.no_viz)
    elif input_path.is_dir():
        images = [f for f in input_path.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        print(f"[INFO] Found {len(images)} images")
        
        for i, img in enumerate(images, 1):
            print(f"\n[{i}/{len(images)}]")
            try:
                segmenter.segment(img, output_dir=args.output, save_visualization=not args.no_viz)
            except Exception as e:
                print(f"[ERROR] {img.name}: {e}")
    
    print("\n[INFO] ✅ Complete!")


if __name__ == "__main__":
    main()