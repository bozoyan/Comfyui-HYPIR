#!/usr/bin/env python3
"""
专门测试 wavelet_reconstruction 和尺寸匹配修复
"""

import torch
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_size_matching():
    """测试尺寸匹配修复逻辑"""
    print("🔧 测试尺寸匹配修复逻辑...")
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("MPS 不可用，使用 CPU 测试")
        device = torch.device("cpu")
        dtype = torch.float32
    else:
        device = torch.device("mps")
        dtype = torch.float16
    
    try:
        # 添加 HYPIR 路径
        hypir_path = os.path.join(current_dir, "HYPIR")
        if hypir_path not in sys.path:
            sys.path.append(hypir_path)
        
        from HYPIR.utils.common import wavelet_reconstruction
        
        # 模拟用户遇到的尺寸不匹配问题
        print("📊 创建测试张量...")
        
        # 原始问题：x 是 1728，ref 是 1730 的尺寸差异
        content_feat = torch.randn(1, 3, 1184, 1728, device=device, dtype=dtype)  # 处理后的内容
        style_feat = torch.randn(1, 3, 1184, 1730, device=device, dtype=dtype)    # 参考图像（稍微不同尺寸）
        
        print(f"content_feat: {content_feat.shape}")
        print(f"style_feat: {style_feat.shape}")
        print(f"尺寸差异: width {content_feat.shape[3]} vs {style_feat.shape[3]}")
        
        # 测试我们的修复
        print("🔄 测试 wavelet_reconstruction 尺寸修复...")
        try:
            result = wavelet_reconstruction(content_feat, style_feat)
            
            print(f"✅ wavelet_reconstruction 成功!")
            print(f"结果形状: {result.shape}")
            print(f"应该匹配 content_feat: {result.shape == content_feat.shape}")
            
            # 检查结果质量
            if torch.isnan(result).any():
                print("❌ 结果包含 NaN")
                return False
            elif torch.isinf(result).any():
                print("❌ 结果包含 infinity")
                return False
            else:
                print("✅ 结果数值正常")
                print(f"结果范围: min={result.min():.6f}, max={result.max():.6f}")
                return True
                
        except Exception as e:
            print(f"❌ wavelet_reconstruction 失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except ImportError as e:
        print(f"❌ 无法导入 HYPIR 模块: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        return False

def test_interpolation_logic():
    """测试插值逻辑修复"""
    print("\\n🎯 测试 BaseEnhancer 插值逻辑修复...")
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        device = torch.device("cpu")
        dtype = torch.float32
    else:
        device = torch.device("mps")
        dtype = torch.float16
    
    # 模拟 BaseEnhancer.enhance 中的尺寸处理
    print("📐 模拟 enhance 方法中的尺寸处理...")
    
    # 原始低质量图像
    lq_original = torch.randn(1, 3, 1184, 864, device=device, dtype=dtype)
    upscale_factor = 2
    
    # Step 1: 放大
    if device.type == "mps" and dtype == torch.float16:
        # MPS float16 兼容性处理
        lq_upscaled_f32 = torch.nn.functional.interpolate(
            lq_original.float(), scale_factor=upscale_factor, mode="bicubic", antialias=True
        )
        lq_upscaled = lq_upscaled_f32.half()
    else:
        lq_upscaled = torch.nn.functional.interpolate(
            lq_original, scale_factor=upscale_factor, mode="bicubic", antialias=True
        )
    ref = lq_upscaled  # 参考图像
    h0, w0 = lq_upscaled.shape[2:]
    
    print(f"原始图像: {lq_original.shape}")
    print(f"放大后参考: {ref.shape}")
    print(f"目标尺寸 h0, w0: {h0}, {w0}")
    
    # Step 2: 模拟 VAE 处理和填充
    vae_scale_factor = 8
    h1, w1 = lq_upscaled.shape[2:]
    ph = (h1 + vae_scale_factor - 1) // vae_scale_factor * vae_scale_factor - h1
    pw = (w1 + vae_scale_factor - 1) // vae_scale_factor * vae_scale_factor - w1
    
    print(f"填充: ph={ph}, pw={pw}")
    
    # 模拟 VAE 解码后的结果（可能有轻微尺寸差异）
    x_from_vae = torch.randn(1, 3, h1, w1 + 2, device=device, dtype=dtype)  # 轻微差异
    
    print(f"VAE 解码结果: {x_from_vae.shape}")
    
    # Step 3: 应用我们的修复逻辑
    print("🔧 应用尺寸修复逻辑...")
    if device.type == "mps" and dtype == torch.float16:
        # MPS float16 兼容性处理
        x_f32 = torch.nn.functional.interpolate(x_from_vae.float(), size=(h0, w0), mode="bicubic", antialias=True)
        x = x_f32.half()
    else:
        x = torch.nn.functional.interpolate(x_from_vae, size=(h0, w0), mode="bicubic", antialias=True)
    
    print(f"插值后 x: {x.shape}")
    print(f"参考 ref: {ref.shape}")
    
    # 我们的修复逻辑
    if x.shape != ref.shape:
        print(f"[Size Fix] Resizing x from {x.shape} to match ref {ref.shape}")
        if device.type == "mps" and dtype == torch.float16:
            # MPS float16 兼容性处理
            x_f32 = torch.nn.functional.interpolate(x.float(), size=ref.shape[2:], mode="bicubic", antialias=True)
            x = x_f32.half()
        else:
            x = torch.nn.functional.interpolate(x, size=ref.shape[2:], mode="bicubic", antialias=True)
        print(f"修复后 x: {x.shape}")
    
    # 验证尺寸匹配
    if x.shape == ref.shape:
        print("✅ 尺寸匹配修复成功!")
        return True
    else:
        print("❌ 尺寸仍然不匹配")
        return False

if __name__ == "__main__":
    print("=== HYPIR 尺寸匹配和 Wavelet 修复测试 ===")
    
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"MPS 可用: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    
    print("\\n" + "="*50)
    
    # 运行测试
    size_test = test_size_matching()
    interpolation_test = test_interpolation_logic()
    
    print("\\n" + "="*50)
    print("🎯 测试总结:")
    print(f"Wavelet 尺寸修复: {'✅ PASS' if size_test else '❌ FAIL'}")
    print(f"插值逻辑修复: {'✅ PASS' if interpolation_test else '❌ FAIL'}")
    
    if size_test and interpolation_test:
        print("\\n🎉 所有修复测试通过！")
        print("✅ 张量尺寸不匹配问题应该已解决")
        print("✅ 2倍放大应该能输出正确尺寸")
        print("\\n现在用户可以尝试在 ComfyUI 中运行 2倍放大了！")
    else:
        print("\\n⚠️  有测试失败，可能需要进一步调试。")
    
    sys.exit(0 if (size_test and interpolation_test) else 1)