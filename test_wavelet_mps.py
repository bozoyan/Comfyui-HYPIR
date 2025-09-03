#!/usr/bin/env python3
"""
专门测试 wavelet reconstruction 在 MPS 设备上的问题
"""

import torch
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_wavelet_reconstruction_mps():
    """测试 wavelet reconstruction 在 MPS 设备上的表现"""
    print("🔍 测试 wavelet reconstruction MPS 兼容性...")
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("MPS 不可用")
        return False
    
    try:
        # 添加 HYPIR 路径
        hypir_path = os.path.join(current_dir, "HYPIR")
        if hypir_path not in sys.path:
            sys.path.append(hypir_path)
        
        from HYPIR.utils.common import wavelet_reconstruction, wavelet_decomposition
        
        device = torch.device("mps")
        dtype = torch.float16
        
        # 创建测试数据
        print("📊 创建测试张量...")
        content_feat = torch.randn(1, 3, 64, 64, device=device, dtype=dtype)
        style_feat = torch.randn(1, 3, 64, 64, device=device, dtype=dtype)
        
        print(f"content_feat: shape={content_feat.shape}, device={content_feat.device}, dtype={content_feat.dtype}")
        print(f"style_feat: shape={style_feat.shape}, device={style_feat.device}, dtype={style_feat.dtype}")
        
        # 检查输入数据
        print(f"content_feat 范围: min={content_feat.min():.6f}, max={content_feat.max():.6f}")
        print(f"style_feat 范围: min={style_feat.min():.6f}, max={style_feat.max():.6f}")
        
        # 先单独测试 wavelet_decomposition
        print("\\n🧪 测试 wavelet_decomposition...")
        try:
            content_high, content_low = wavelet_decomposition(content_feat)
            style_high, style_low = wavelet_decomposition(style_feat)
            
            print(f"content_high: min={content_high.min():.6f}, max={content_high.max():.6f}")
            print(f"content_low: min={content_low.min():.6f}, max={content_low.max():.6f}")
            print(f"style_high: min={style_high.min():.6f}, max={style_high.max():.6f}")
            print(f"style_low: min={style_low.min():.6f}, max={style_low.max():.6f}")
            
            # 检查是否有 NaN 或 inf
            if torch.isnan(content_high).any() or torch.isnan(content_low).any():
                print("❌ content wavelet decomposition 产生了 NaN")
            if torch.isnan(style_high).any() or torch.isnan(style_low).any():
                print("❌ style wavelet decomposition 产生了 NaN")
            if torch.isinf(content_high).any() or torch.isinf(content_low).any():
                print("❌ content wavelet decomposition 产生了 inf")
            if torch.isinf(style_high).any() or torch.isinf(style_low).any():
                print("❌ style wavelet decomposition 产生了 inf")
                
        except Exception as e:
            print(f"❌ wavelet_decomposition 失败: {e}")
            return False
        
        # 测试 wavelet_reconstruction
        print("\\n🔄 测试 wavelet_reconstruction...")
        try:
            result = wavelet_reconstruction(content_feat, style_feat)
            
            print(f"reconstruction result: shape={result.shape}, device={result.device}, dtype={result.dtype}")
            print(f"result 范围: min={result.min():.6f}, max={result.max():.6f}, mean={result.mean():.6f}")
            
            # 检查结果
            if torch.isnan(result).any():
                print("❌ wavelet_reconstruction 结果包含 NaN")
                return False
            
            if torch.isinf(result).any():
                print("❌ wavelet_reconstruction 结果包含 inf")
                return False
            
            if result.abs().max() < 1e-6:
                print("❌ wavelet_reconstruction 结果全为零或接近零")
                return False
            
            print("✅ wavelet_reconstruction 测试通过")
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
        import traceback
        traceback.print_exc()
        return False

def test_float16_operations():
    """测试 MPS 设备上的 float16 数学运算"""
    print("\\n🧮 测试 MPS float16 数学运算...")
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("MPS 不可用")
        return False
    
    device = torch.device("mps")
    dtype = torch.float16
    
    # 测试基本运算
    print("📐 测试基本数学运算...")
    try:
        a = torch.randn(100, 100, device=device, dtype=dtype)
        b = torch.randn(100, 100, device=device, dtype=dtype)
        
        # 加法
        c = a + b
        print(f"加法: {c.shape}, 无 NaN: {not torch.isnan(c).any()}")
        
        # 乘法
        d = a * b
        print(f"乘法: {d.shape}, 无 NaN: {not torch.isnan(d).any()}")
        
        # 卷积操作（类似 wavelet blur）
        kernel = torch.tensor([
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625]
        ], dtype=dtype, device=device)
        
        x = torch.randn(1, 3, 32, 32, device=device, dtype=dtype)
        kernel_expanded = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
        conv_result = torch.nn.functional.conv2d(x, kernel_expanded, padding=1, groups=3)
        print(f"卷积: {conv_result.shape}, 无 NaN: {not torch.isnan(conv_result).any()}")
        
        return True
        
    except Exception as e:
        print(f"❌ float16 运算测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=== MPS Wavelet Reconstruction 专项测试 ===")
    
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"MPS 可用: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("⚠️  MPS 不可用")
        sys.exit(1)
    
    print("\\n" + "="*50)
    
    # 运行测试
    float16_test = test_float16_operations()
    wavelet_test = test_wavelet_reconstruction_mps()
    
    print("\\n" + "="*50)
    print("🎯 测试总结:")
    print(f"Float16 运算: {'✅ PASS' if float16_test else '❌ FAIL'}")
    print(f"Wavelet Reconstruction: {'✅ PASS' if wavelet_test else '❌ FAIL'}")
    
    if float16_test and wavelet_test:
        print("\\n🎉 所有测试通过！MPS wavelet reconstruction 应该正常工作。")
    else:
        print("\\n⚠️  有测试失败，MPS 可能存在兼容性问题。")
        print("建议:")
        print("1. 检查 PyTorch MPS 后端版本")
        print("2. 尝试使用 float32 而非 float16")
        print("3. 考虑使用 CPU 作为备选方案")
    
    sys.exit(0 if (float16_test and wavelet_test) else 1)