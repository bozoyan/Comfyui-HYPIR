#!/usr/bin/env python3
"""
测试 2 倍放大的修复效果
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
    """创建测试图像"""
    width, height = size
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 创建彩色测试图案
    for i in range(height):
        for j in range(width):
            img_array[i, j] = [
                int(128 + 127 * np.sin(2 * np.pi * i / height)),  # Red sine wave
                int(128 + 127 * np.cos(2 * np.pi * j / width)),   # Green cosine wave
                int(255 * (i + j) / (height + width))             # Blue gradient
            ]
    
    return Image.fromarray(img_array)

def test_2x_upscale():
    """测试 2 倍放大功能"""
    print("🔍 测试 2 倍放大修复效果...")
    
    try:
        # Import the node
        from hypir_advanced_node import HYPIRAdvancedRestorationWithDevice
        
        # Create test image - 使用原始用户的输入尺寸
        print("📸 创建测试图像...")
        test_image = create_test_image((864, 1184))  # 原始用户输入的一半尺寸
        
        # Create node instance
        print("🔧 创建 HYPIR 节点...")
        node = HYPIRAdvancedRestorationWithDevice()
        
        # 2倍放大参数
        upscale_factor = 2  # 这是用户遇到问题的场景
        device = "mps"
        tile_size = 512
        
        print(f"🎯 测试参数:")
        print(f"  - 输入图像尺寸: {test_image.size}")
        print(f"  - 放大倍数: {upscale_factor}x") 
        print(f"  - 期望输出尺寸: {test_image.size[0] * upscale_factor}x{test_image.size[1] * upscale_factor}")
        print(f"  - 设备: {device}")
        print(f"  - Tile 大小: {tile_size}")
        
        # Convert PIL image to ComfyUI tensor format
        img_array = np.array(test_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)  # (H, W, C)
        
        print(f"  - 输入张量形状: {img_tensor.shape}")
        print(f"  - 输入张量范围: min={img_tensor.min():.4f}, max={img_tensor.max():.4f}")
        
        # Run restoration
        print("🚀 运行 HYPIR 2倍放大...")
        result = node.restore_image_with_device(
            image=img_tensor,
            prompt="high quality, detailed",
            upscale_factor=upscale_factor,
            seed=42,
            model_name="HYPIR_sd2",
            base_model_path="stable-diffusion-2-1-base",
            device=device,
            model_t=200,
            coeff_t=120,
            lora_rank=256,
            patch_size=512,
            encode_patch_size=tile_size,
            decode_patch_size=tile_size,
            batch_size=1,
            unload_model_after=True
        )
        
        if result and len(result) == 2:
            restored_image_tensor, status_msg = result
            print(f"✅ 处理成功！")
            print(f"状态信息: {status_msg}")
            print(f"输出张量形状: {restored_image_tensor.shape}")
            
            # 检查输出尺寸是否正确
            expected_height = test_image.size[1] * upscale_factor
            expected_width = test_image.size[0] * upscale_factor
            actual_height, actual_width = restored_image_tensor.shape[0], restored_image_tensor.shape[1]
            
            print(f"📏 尺寸检查:")
            print(f"  - 期望尺寸: {expected_width}x{expected_height}")
            print(f"  - 实际尺寸: {actual_width}x{actual_height}")
            
            if (actual_width, actual_height) == (expected_width, expected_height):
                print("✅ 输出尺寸正确！2倍放大成功")
                size_correct = True
            else:
                print("❌ 输出尺寸不正确！")
                size_correct = False
            
            # 分析输出质量
            output_min = restored_image_tensor.min().item()
            output_max = restored_image_tensor.max().item()
            output_mean = restored_image_tensor.mean().item()
            
            print(f"📊 输出质量分析:")
            print(f"  - 范围: min={output_min:.6f}, max={output_max:.6f}")
            print(f"  - 平均值: {output_mean:.6f}")
            
            if output_max < 0.01:
                print("❌ 输出是黑色图像")
                quality_good = False
            elif output_max < 0.1:
                print("⚠️  输出过暗，可能有问题")
                quality_good = False
            else:
                print("✅ 输出质量正常")
                quality_good = True
                
                # 保存结果用于检查
                if len(restored_image_tensor.shape) == 3:  # (H, W, C)
                    clean_tensor = torch.nan_to_num(restored_image_tensor, nan=0.0, posinf=1.0, neginf=0.0)
                    clean_tensor = clean_tensor.clamp(0, 1)
                    
                    img_array = (clean_tensor.cpu().numpy() * 255).astype(np.uint8)
                    restored_image = Image.fromarray(img_array)
                    
                    output_path = os.path.join(current_dir, "test_2x_result.png")
                    restored_image.save(output_path)
                    print(f"💾 2倍放大结果已保存: {output_path}")
            
            return size_correct and quality_good
            
        else:
            print("❌ 处理失败，未返回有效结果")
            if result:
                print(f"返回结果: {result}")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== HYPIR 2倍放大修复验证 ===")
    
    # Check environment
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"MPS 可用: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("⚠️  MPS 不可用，跳过测试...")
        sys.exit(1)
    
    print("\n" + "="*60)
    
    # Run the test
    success = test_2x_upscale()
    
    print("\n" + "="*60)
    print("🎯 测试结果:")
    if success:
        print("🎉 2倍放大测试通过！")
        print("✅ 输出尺寸正确")
        print("✅ 输出质量正常")
        print("✅ 张量尺寸匹配问题已解决")
        print("\n用户现在应该能够正常使用 2倍放大功能了！")
    else:
        print("❌ 2倍放大测试失败")
        print("🔍 可能的问题:")
        print("  - 张量尺寸仍然不匹配")
        print("  - Wavelet reconstruction 仍有 NaN 问题")
        print("  - 输出质量不佳")
        print("\n建议进一步调试或使用 CPU 设备。")
    
    sys.exit(0 if success else 1)