"""
Hair Segmentation Test Script
Tests if PyTorch + CUDA + basic image processing works
"""

import torch
import cv2
import numpy as np
from PIL import Image
import sys

def test_cuda():
    """Test CUDA availability"""
    print("=" * 60)
    print("CUDA TEST")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  WARNING: CUDA not available. Will use CPU (slow).")
    
    return cuda_available

def test_pytorch():
    """Test PyTorch basic operations"""
    print("\n" + "=" * 60)
    print("PYTORCH TEST")
    print("=" * 60)
    
    # Create random tensor
    x = torch.rand(3, 224, 224)
    print(f"Tensor shape: {x.shape}")
    print(f"Tensor device: {x.device}")
    
    # Test GPU transfer
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        print(f"GPU tensor device: {x_gpu.device}")
        
        # Simple operation on GPU
        result = x_gpu * 2 + 1
        print(f"GPU operation successful: {result.shape}")
        
        return True
    else:
        print("Skipping GPU test (CUDA not available)")
        return False

def test_opencv():
    """Test OpenCV installation"""
    print("\n" + "=" * 60)
    print("OPENCV TEST")
    print("=" * 60)
    
    try:
        # Create test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (100, 150, 200)  # BGR
        
        # Draw circle
        cv2.circle(img, (320, 240), 100, (0, 255, 0), -1)
        
        # Save test image
        output_path = "test_opencv_output.jpg"
        cv2.imwrite(output_path, img)
        print(f"✅ OpenCV test image saved: {output_path}")
        
        # Read back
        img_read = cv2.imread(output_path)
        print(f"✅ Image read back successfully: {img_read.shape}")
        
        return True
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def create_simple_hair_mask(image_path, output_path):
    """
    Simple hair mask demo using color-based segmentation
    (This is NOT production code, just a quick test)
    """
    print("\n" + "=" * 60)
    print("SIMPLE HAIR MASK TEST")
    print("=" * 60)
    
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not read image: {image_path}")
            print("💡 Place a test image named 'test_face.jpg' in the current directory")
            return False
        
        print(f"✅ Image loaded: {img.shape}")
        
        # Convert to HSV for better hair detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Simple dark hair detection (this is very basic!)
        # Hair is typically darker than skin
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 80])
        
        hair_mask = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
        
        # Save mask
        cv2.imwrite(output_path, hair_mask)
        print(f"✅ Hair mask saved: {output_path}")
        
        # Create visualization
        vis = img.copy()
        vis[hair_mask > 0] = [0, 0, 255]  # Red overlay on hair
        vis_path = output_path.replace('.png', '_visualization.jpg')
        cv2.imwrite(vis_path, vis)
        print(f"✅ Visualization saved: {vis_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hair mask test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "3D HAIR TRY-ON PLATFORM - SYSTEM TEST" + " " * 10 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    results = {}
    
    # Test 1: CUDA
    results['cuda'] = test_cuda()
    
    # Test 2: PyTorch
    results['pytorch'] = test_pytorch()
    
    # Test 3: OpenCV
    results['opencv'] = test_opencv()
    
    # Test 4: Simple hair mask (optional - needs test image)
    test_image = "test_face.jpg"
    import os
    if os.path.exists(test_image):
        results['hair_mask'] = create_simple_hair_mask(
            test_image, 
            "test_hair_mask.png"
        )
    else:
        print("\n" + "=" * 60)
        print("HAIR MASK TEST SKIPPED")
        print("=" * 60)
        print(f"💡 To test hair segmentation:")
        print(f"   1. Place a face photo in current directory")
        print(f"   2. Name it: test_face.jpg")
        print(f"   3. Run this script again")
        results['hair_mask'] = None
    
    # Summary
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 20 + "TEST SUMMARY" + " " * 26 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    for test_name, result in results.items():
        if result is None:
            status = "⏭️  SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        
        print(f"{test_name.upper():20s}: {status}")
    
    print()
    
    # Next steps
    if results['cuda'] and results['pytorch'] and results['opencv']:
        print("╔" + "=" * 58 + "╗")
        print("║" + " " * 15 + "🎉 ALL TESTS PASSED! 🎉" + " " * 17 + "║")
        print("╚" + "=" * 58 + "╝")
        print()
        print("Next steps:")
        print("1. Download BiSeNet hair segmentation model weights")
        print("2. Create production hair segmentation module")
        print("3. Test with Metashape photogrammetry")
        print()
    else:
        print("╔" + "=" * 58 + "╗")
        print("║" + " " * 10 + "⚠️  SOME TESTS FAILED - CHECK ABOVE" + " " * 11 + "║")
        print("╚" + "=" * 58 + "╝")
        print()
        sys.exit(1)

if __name__ == "__main__":
    main()