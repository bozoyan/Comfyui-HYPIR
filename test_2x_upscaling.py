#!/usr/bin/env python3
"""
Test the specific 2x upscaling scenario that the user encountered
"""

import torch
import sys
import os
from PIL import Image
import numpy as np

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def create_test_image_for_upscaling(size=(1152, 896)):
    """Create a test image similar to user's scenario"""
    width, height = size
    img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img_array)

def test_2x_upscaling_scenario():
    """Test the exact 2x upscaling scenario that caused the error"""
    print("🔍 Testing 2x upscaling scenario...")
    
    try:
        # Import the node
        from hypir_advanced_node import HYPIRAdvancedRestorationWithDevice
        
        # Create test image similar to user's case
        # User's processing resulted in: input_size: torch.Size([1, 3, 2304, 1792])
        # This suggests original image was around 1152x896 before 2x upscaling
        print("📸 Creating test image (1152x896)...")
        test_image = create_test_image_for_upscaling((1152, 896))
        
        # Create node instance
        print("🔧 Creating HYPIR node...")
        node = HYPIRAdvancedRestorationWithDevice()
        
        # Test parameters matching user's scenario
        upscale_factor = 2  # This is what caused the large image
        device = "mps"
        tile_size = 1024  # User's tile_size from logs
        
        print(f"🎯 Testing with 2x upscaling parameters:")
        print(f"  - Original image size: {test_image.size}")
        print(f"  - Upscale factor: {upscale_factor}x") 
        print(f"  - Expected output size: {test_image.size[0]*upscale_factor}x{test_image.size[1]*upscale_factor}")
        print(f"  - Device: {device}")
        print(f"  - Tile size: {tile_size}")
        
        # Convert PIL image to ComfyUI tensor format
        img_array = np.array(test_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)  # (H, W, C)
        
        print(f"  - Input tensor shape: {img_tensor.shape}")
        
        # Run restoration with 2x upscaling
        print("🚀 Running HYPIR 2x upscaling...")
        result = node.restore_image_with_device(
            image=img_tensor,
            prompt="high quality, detailed, 2x upscaling",
            upscale_factor=upscale_factor,
            seed=-1,
            model_name="HYPIR_sd2",
            base_model_path="stable-diffusion-2-1-base",
            device=device,
            model_t=200,
            coeff_t=120,  # User's coeff_t from logs
            lora_rank=256,
            patch_size=512,
            encode_patch_size=tile_size,
            decode_patch_size=tile_size,
            batch_size=1,
            unload_model_after=True
        )
        
        if result and len(result) == 2:  # Should return (IMAGE, STRING)
            restored_image_tensor, status_msg = result
            print(f"✅ Success! Status: {status_msg[:200]}...")  # Truncate long status
            print(f"Output tensor shape: {restored_image_tensor.shape}")
            
            # Verify the upscaling worked
            expected_height = img_tensor.shape[0] * upscale_factor
            expected_width = img_tensor.shape[1] * upscale_factor
            actual_height, actual_width = restored_image_tensor.shape[0], restored_image_tensor.shape[1]
            
            if actual_height == expected_height and actual_width == expected_width:
                print(f"✅ Upscaling verified: {img_tensor.shape[1]}x{img_tensor.shape[0]} → {actual_width}x{actual_height}")
            else:
                print(f"⚠️  Upscaling mismatch: expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}")
            
            # Convert and save result
            if len(restored_image_tensor.shape) == 3:  # (H, W, C)
                img_array = (restored_image_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                restored_image = Image.fromarray(img_array)
                
                # Save test result
                output_path = os.path.join(current_dir, "test_2x_upscaling_result.png")
                restored_image.save(output_path)
                print(f"💾 Test result saved to: {output_path}")
            
            return True
        else:
            print("❌ No valid result returned from restoration")
            return False
            
    except Exception as e:
        print(f"❌ Error during 2x upscaling test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing 2x Upscaling Scenario (User's Issue) ===")
    
    # Check PyTorch and MPS
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("⚠️  MPS not available, exiting...")
        sys.exit(1)
    
    print("\n" + "="*60)
    
    # Run the test
    success = test_2x_upscaling_scenario()
    
    print("\n" + "="*60)
    print("🎯 Test Result:")
    if success:
        print("🎉 2x Upscaling test PASSED! The shape error should be fixed.")
        print("The user should now be able to do 2x upscaling without the shape error.")
    else:
        print("❌ 2x Upscaling test FAILED. Check the error messages above.")
    
    sys.exit(0 if success else 1)