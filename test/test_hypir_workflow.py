#!/usr/bin/env python3
"""
Test HYPIR end-to-end functionality with MPS device
"""

import torch
import sys
import os
from PIL import Image
import numpy as np

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def create_test_image(size=(512, 512)):
    """Create a simple test image"""
    # Create a simple gradient image
    width, height = size
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            img_array[i, j] = [
                int(255 * i / height),  # Red gradient
                int(255 * j / width),   # Green gradient  
                128  # Constant blue
            ]
    
    return Image.fromarray(img_array)

def test_hypir_workflow():
    """Test complete HYPIR workflow"""
    print("🧪 Testing HYPIR end-to-end workflow with MPS...")
    
    try:
        # Import the node
        from hypir_advanced_node import HYPIRAdvancedRestorationWithDevice
        
        # Create test image
        print("📸 Creating test image...")
        test_image = create_test_image((512, 512))
        
        # Create node instance
        print("🔧 Creating HYPIR node...")
        node = HYPIRAdvancedRestorationWithDevice()
        
        # Test parameters
        steps = 5  # Use fewer steps for testing
        cfg = 7.5
        tile_size = 512  # Smaller tile size for testing
        device = "mps"
        
        print(f"🎯 Testing with parameters:")
        print(f"  - Device: {device}")
        print(f"  - Steps: {steps}")
        print(f"  - CFG: {cfg}")
        print(f"  - Tile size: {tile_size}")
        
        # Run restoration
        print("🚀 Running HYPIR restoration...")
        result = node.restore_image_with_device(
            image=torch.from_numpy(np.array(test_image)).float() / 255.0,  # Convert PIL to tensor
            prompt="high quality, detailed",
            upscale_factor=1,
            seed=-1,
            model_name="HYPIR_sd2",
            base_model_path="stable-diffusion-2-1-base",
            device=device,
            model_t=200,
            coeff_t=100,
            lora_rank=256,
            patch_size=tile_size,
            encode_patch_size=tile_size,
            decode_patch_size=tile_size,
            batch_size=1,
            unload_model_after=True
        )
        
        if result and len(result) == 2:  # Should return (IMAGE, STRING)
            restored_image_tensor, status_msg = result
            print(f"✅ Success! Status: {status_msg}")
            print(f"Restored tensor shape: {restored_image_tensor.shape}")
            
            # Convert tensor back to PIL image for saving
            if len(restored_image_tensor.shape) == 3:  # (H, W, C)
                img_array = (restored_image_tensor.cpu().numpy() * 255).astype(np.uint8)
                restored_image = Image.fromarray(img_array)
            else:
                print(f"Unexpected tensor shape: {restored_image_tensor.shape}")
                return False
            
            # Save test result
            output_path = os.path.join(current_dir, "test_result.png")
            restored_image.save(output_path)
            print(f"💾 Test result saved to: {output_path}")
            
            return True
        else:
            print("❌ No result returned from restoration")
            return False
            
    except Exception as e:
        print(f"❌ Error during HYPIR workflow test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_selection():
    """Test device selection logic"""
    print("\n🔍 Testing device selection...")
    
    try:
        from hypir_advanced_node import get_optimal_device, check_mps_compatibility
        
        # Test automatic device selection
        auto_device = get_optimal_device("auto")
        print(f"Auto-selected device: {auto_device}")
        
        # Test MPS compatibility check
        if auto_device == "mps":
            compatible_device, warnings = check_mps_compatibility("mps", (512, 512))
            print(f"MPS compatibility check: {compatible_device}")
            if warnings:
                print("Warnings:")
                for warning in warnings[:3]:  # Show first 3 warnings
                    print(f"  - {warning}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during device selection test: {e}")
        return False

if __name__ == "__main__":
    print("=== HYPIR End-to-End MPS Compatibility Test ===")
    
    # Check PyTorch and MPS
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("⚠️  MPS not available, exiting...")
        sys.exit(1)
    
    print("\n" + "="*60)
    
    # Run tests
    device_test = test_device_selection()
    workflow_test = test_hypir_workflow()
    
    print("\n" + "="*60)
    print("🎯 Test Summary:")
    print(f"Device Selection: {'✅ PASS' if device_test else '❌ FAIL'}")
    print(f"HYPIR Workflow: {'✅ PASS' if workflow_test else '❌ FAIL'}")
    
    if device_test and workflow_test:
        print("\n🎉 All tests passed! HYPIR should work correctly with MPS.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    sys.exit(0 if (device_test and workflow_test) else 1)