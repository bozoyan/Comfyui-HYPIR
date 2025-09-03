#!/usr/bin/env python3
"""
诊断 HYPIR 插件输出黑色图像的问题
"""

import torch
import sys
import os
from PIL import Image
import numpy as np

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_black_image_issue():
    """测试黑色图像问题的诊断"""
    print("🔍 诊断 HYPIR 黑色图像问题...")
    
    try:
        # Import the node
        from hypir_advanced_node import HYPIRAdvancedRestorationWithDevice
        
        # 创建一个小的测试图像（避免大图像的复杂性）
        print("📸 创建测试图像...")
        test_image = create_simple_test_image((256, 256))
        
        # Create node instance
        print("🔧 创建 HYPIR 节点...")
        node = HYPIRAdvancedRestorationWithDevice()
        
        # 使用1倍放大（用户遇到黑色图像的场景）
        upscale_factor = 1
        device = "mps"
        tile_size = 512  # 相对较小的tile size以减少复杂性
        
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
        
        # 运行HYPIR处理
        print("🚀 运行 HYPIR 处理...")
        result = node.restore_image_with_device(
            image=img_tensor,
            prompt="high quality, detailed",
            upscale_factor=upscale_factor,
            seed=42,  # 固定种子以便重现
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
            output_tensor, status_msg = result
            print(f"✅ 处理完成！")
            print(f"输出张量形状: {output_tensor.shape}")
            
            # 详细分析输出
            output_min = output_tensor.min().item()
            output_max = output_tensor.max().item()
            output_mean = output_tensor.mean().item()
            output_std = output_tensor.std().item()
            
            print(f"📊 输出张量分析:")
            print(f"  - 范围: min={output_min:.6f}, max={output_max:.6f}")
            print(f"  - 统计: mean={output_mean:.6f}, std={output_std:.6f}")
            
            # 检查是否为黑色图像
            if output_max < 0.01:
                print("❌ 检测到黑色图像问题！")
                print("🔍 可能的原因:")
                print("   1. MPS 设备数值计算精度问题")
                print("   2. 数据类型转换问题")
                print("   3. Tiled VAE 处理中的数值溢出")
                print("   4. 模型权重加载问题")
                
                # 建议的解决方案
                print("💡 建议的解决方案:")
                print("   1. 尝试更小的 tile 大小 (256, 384)")
                print("   2. 切换到 CPU 设备进行对比测试")
                print("   3. 检查输入图像是否正确加载")
                print("   4. 验证模型权重是否正确")
                
                return False
            elif output_max < 0.1:
                print("⚠️  图像过暗，可能存在问题")
                return False
            else:
                print("✅ 输出图像正常")
                
                # 保存输出图像进行检查
                output_array = (output_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                output_image = Image.fromarray(output_array)
                
                output_path = os.path.join(current_dir, "debug_output.png")
                output_image.save(output_path)
                print(f"💾 输出图像已保存: {output_path}")
                
                return True
        else:
            print("❌ 处理失败，未返回有效结果")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_test_image(size=(256, 256)):
    """创建简单的测试图像"""
    width, height = size
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 创建一个简单的渐变图案，确保有明显的对比
    for i in range(height):
        for j in range(width):
            # 创建彩色渐变
            img_array[i, j] = [
                min(255, int(255 * (i + j) / (height + width))),  # Red
                min(255, int(255 * i / height)),                  # Green
                min(255, int(255 * j / width))                    # Blue
            ]
    
    return Image.fromarray(img_array)

def test_cpu_comparison():
    """对比 CPU 和 MPS 的处理结果"""
    print("\n🔄 对比 CPU 和 MPS 处理结果...")
    
    try:
        from hypir_advanced_node import HYPIRAdvancedRestorationWithDevice
        
        # 创建相同的测试图像
        test_image = create_simple_test_image((256, 256))
        img_array = np.array(test_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)
        
        node = HYPIRAdvancedRestorationWithDevice()
        
        # 测试参数
        common_params = {
            "image": img_tensor,
            "prompt": "high quality, detailed",
            "upscale_factor": 1,
            "seed": 42,
            "model_name": "HYPIR_sd2",
            "base_model_path": "stable-diffusion-2-1-base",
            "model_t": 200,
            "coeff_t": 120,
            "lora_rank": 256,
            "patch_size": 512,
            "encode_patch_size": 512,
            "decode_patch_size": 512,
            "batch_size": 1,
            "unload_model_after": True
        }
        
        results = {}
        
        # 测试 CPU
        print("🖥️  测试 CPU 设备...")
        try:
            cpu_result = node.restore_image_with_device(device="cpu", **common_params)
            if cpu_result and len(cpu_result) == 2:
                cpu_tensor = cpu_result[0]
                cpu_stats = {
                    'min': cpu_tensor.min().item(),
                    'max': cpu_tensor.max().item(),
                    'mean': cpu_tensor.mean().item()
                }
                results['cpu'] = cpu_stats
                print(f"   CPU 结果: min={cpu_stats['min']:.6f}, max={cpu_stats['max']:.6f}, mean={cpu_stats['mean']:.6f}")
            else:
                print("   CPU 处理失败")
        except Exception as e:
            print(f"   CPU 处理出错: {e}")
        
        # 测试 MPS
        print("🍎 测试 MPS 设备...")
        try:
            mps_result = node.restore_image_with_device(device="mps", **common_params)
            if mps_result and len(mps_result) == 2:
                mps_tensor = mps_result[0]
                mps_stats = {
                    'min': mps_tensor.min().item(),
                    'max': mps_tensor.max().item(),
                    'mean': mps_tensor.mean().item()
                }
                results['mps'] = mps_stats
                print(f"   MPS 结果: min={mps_stats['min']:.6f}, max={mps_stats['max']:.6f}, mean={mps_stats['mean']:.6f}")
            else:
                print("   MPS 处理失败")
        except Exception as e:
            print(f"   MPS 处理出错: {e}")
        
        # 比较结果
        if 'cpu' in results and 'mps' in results:
            print("📊 设备对比结果:")
            cpu_stats = results['cpu']
            mps_stats = results['mps']
            
            print(f"   CPU: {cpu_stats}")
            print(f"   MPS: {mps_stats}")
            
            # 检查差异
            max_diff = abs(cpu_stats['max'] - mps_stats['max'])
            mean_diff = abs(cpu_stats['mean'] - mps_stats['mean'])
            
            if mps_stats['max'] < 0.01 and cpu_stats['max'] > 0.1:
                print("❌ MPS 输出明显异常（黑色图像），而 CPU 正常")
                print("🔍 这表明问题确实与 MPS 设备兼容性有关")
            elif max_diff < 0.05 and mean_diff < 0.05:
                print("✅ CPU 和 MPS 结果基本一致")
            else:
                print("⚠️  CPU 和 MPS 结果存在差异，但可能在可接受范围内")
        
        return results
        
    except Exception as e:
        print(f"❌ 设备对比测试出错: {e}")
        return {}

if __name__ == "__main__":
    print("=== HYPIR 黑色图像问题诊断工具 ===")
    
    # 检查环境
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"MPS 可用: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("⚠️  MPS 不可用，无法进行完整测试")
        sys.exit(1)
    
    print("\n" + "="*60)
    
    # 运行主要诊断测试
    main_test_passed = test_black_image_issue()
    
    # 运行设备对比测试
    comparison_results = test_cpu_comparison()
    
    print("\n" + "="*60)
    print("🎯 诊断总结:")
    
    if main_test_passed:
        print("✅ 主要测试通过，未检测到黑色图像问题")
    else:
        print("❌ 主要测试失败，检测到黑色图像问题")
    
    if comparison_results:
        print("✅ 设备对比测试完成")
    else:
        print("❌ 设备对比测试失败")
    
    if not main_test_passed:
        print("\n💡 建议的下一步:")
        print("1. 检查我们添加的 MPS 数值稳定性修复是否生效")
        print("2. 尝试更小的 tile 大小")
        print("3. 检查模型权重是否正确加载")
        print("4. 考虑在关键处理步骤添加更多调试信息")
    
    sys.exit(0 if main_test_passed else 1)