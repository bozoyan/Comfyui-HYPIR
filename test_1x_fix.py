#!/usr/bin/env python3
"""
Test HYPIR 1x processing to verify black image fix
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
    """Create a test image with clear patterns"""
    width, height = size
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a colorful test pattern
    for i in range(height):
        for j in range(width):
            img_array[i, j] = [
                int(128 + 127 * np.sin(2 * np.pi * i / height)),  # Red sine wave
                int(128 + 127 * np.cos(2 * np.pi * j / width)),   # Green cosine wave
                int(255 * (i + j) / (height + width))             # Blue gradient
            ]
    
    return Image.fromarray(img_array)

def test_1x_processing():
    """Test 1x upscaling that user reported as producing black images"""
    print("🔍 测试 1倍放大处理（用户报告的黑色图像问题）...")
    
    try:
        # Import the node
        from hypir_advanced_node import HYPIRAdvancedRestorationWithDevice
        
        # Create test image
        print("📸 创建测试图像...")
        test_image = create_test_image((512, 512))
        
        # Create node instance
        print("🔧 创建 HYPIR 节点...")
        node = HYPIRAdvancedRestorationWithDevice()
        
        # Test parameters similar to user's case
        upscale_factor = 1  # This is the problematic case
        device = "mps"
        tile_size = 512
        
        print(f"🎯 测试参数:")
        print(f"  - 图像尺寸: {test_image.size}")
        print(f"  - 放大倍数: {upscale_factor}x") 
        print(f"  - 设备: {device}")
        print(f"  - Tile 大小: {tile_size}")
        
        # Convert PIL image to ComfyUI tensor format
        img_array = np.array(test_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)  # (H, W, C)
        
        print(f"  - 输入张量形状: {img_tensor.shape}")
        print(f"  - 输入张量范围: min={img_tensor.min():.4f}, max={img_tensor.max():.4f}, mean={img_tensor.mean():.4f}")
        
        # Run restoration
        print("🚀 运行 HYPIR 1倍处理...")
        result = node.restore_image_with_device(
            image=img_tensor,
            prompt="high quality, detailed",
            upscale_factor=upscale_factor,
            seed=42,  # Fixed seed for reproducibility
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
            print(f"✅ 处理成功！")
            print(f"输出张量形状: {restored_image_tensor.shape}")
            
            # Analyze output
            output_min = restored_image_tensor.min().item()
            output_max = restored_image_tensor.max().item()
            output_mean = restored_image_tensor.mean().item()
            output_std = restored_image_tensor.std().item()
            
            print(f"📊 输出分析:")
            print(f"  - 范围: min={output_min:.6f}, max={output_max:.6f}")
            print(f"  - 统计: mean={output_mean:.6f}, std={output_std:.6f}")
            
            # Check for black image issue
            if output_max < 0.01:
                print("❌ 仍然是黑色图像问题！")
                return False
            elif output_max < 0.1:
                print("⚠️  图像过暗，可能仍有问题")
                return False
            else:
                print("✅ 图像输出正常")
                
                # Save result for visual inspection
                if len(restored_image_tensor.shape) == 3:  # (H, W, C)
                    # Handle potential NaN/inf values before conversion
                    clean_tensor = torch.nan_to_num(restored_image_tensor, nan=0.0, posinf=1.0, neginf=0.0)
                    clean_tensor = clean_tensor.clamp(0, 1)
                    
                    img_array = (clean_tensor.cpu().numpy() * 255).astype(np.uint8)
                    restored_image = Image.fromarray(img_array)
                    
                    # Save test result
                    output_path = os.path.join(current_dir, "test_1x_result.png")
                    restored_image.save(output_path)
                    print(f"💾 测试结果已保存: {output_path}")
                
                return True
        else:
            print("❌ 处理失败，未返回有效结果")
            if result:
                print(f"返回的结果: {result}")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== HYPIR 1倍放大黑色图像修复验证 ===")
    
    # Check environment
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"MPS 可用: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("⚠️  MPS 不可用，跳过测试...")
        sys.exit(1)
    
    print("\n" + "="*60)
    
    # Run the test
    success = test_1x_processing()
    
    print("\n" + "="*60)
    print("🎯 测试结果:")
    if success:
        print("🎉 1倍放大测试通过！黑色图像问题应该已修复。")
        print("用户现在应该能够正常使用 1倍放大功能了。")
    else:
        print("❌ 1倍放大测试失败，黑色图像问题可能仍然存在。")
        print("建议进一步调试或使用 CPU 设备作为替代方案。")
    
    sys.exit(0 if success else 1)