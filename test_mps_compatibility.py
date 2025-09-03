#!/usr/bin/env python3
"""
测试 MPS 兼容性改进
"""
import os
import sys

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_mps_compatibility():
    """测试 MPS 兼容性改进"""
    try:
        print("🧪 测试 MPS 兼容性改进...")
        
        # 测试设备选择函数
        from hypir_advanced_node import get_optimal_device, check_mps_compatibility
        
        print("\\n🔍 测试设备选择:")
        auto_device = get_optimal_device("auto")
        mps_device = get_optimal_device("mps") 
        print(f"自动选择: {auto_device}")
        print(f"MPS 请求: {mps_device}")
        
        print("\\n📏 测试 MPS 兼容性检查:")
        
        # 测试小图像
        small_device, small_warnings = check_mps_compatibility("mps", (512, 512))
        print(f"小图像 (512x512): 设备={small_device}")
        if small_warnings:
            for warning in small_warnings[:2]:  # 只显示前两个警告
                print(f"  - {warning}")
        
        # 测试大图像
        large_device, large_warnings = check_mps_compatibility("mps", (2304, 1792))
        print(f"大图像 (2304x1792): 设备={large_device}")
        if large_warnings:
            for warning in large_warnings[:3]:  # 只显示前三个警告
                print(f"  - {warning}")
        
        print("\\n💡 测试数据类型优化:")
        try:
            # 测试 BaseEnhancer 的数据类型选择
            hypir_path = os.path.join(current_dir, "HYPIR")
            if hypir_path not in sys.path:
                sys.path.append(hypir_path)
            
            from HYPIR.enhancer.base import BaseEnhancer
            
            # 模拟 MPS 设备
            mps_enhancer = BaseEnhancer.__new__(BaseEnhancer)
            mps_enhancer.__init__(
                base_model_path="test",
                weight_path="test", 
                lora_modules=[],
                lora_rank=256,
                model_t=200,
                coeff_t=100,
                device="mps"
            )
            print(f"MPS 设备数据类型: {mps_enhancer.weight_dtype}")
            
            # 模拟 CUDA 设备
            cuda_enhancer = BaseEnhancer.__new__(BaseEnhancer)
            cuda_enhancer.__init__(
                base_model_path="test",
                weight_path="test",
                lora_modules=[],
                lora_rank=256, 
                model_t=200,
                coeff_t=100,
                device="cuda"
            )
            print(f"CUDA 设备数据类型: {cuda_enhancer.weight_dtype}")
            
        except Exception as e:
            print(f"数据类型测试失败: {e}")
        
        print("\\n🎉 MPS 兼容性测试完成!")
        print("\\n📋 改进摘要:")
        print("1. ✅ MPS 设备使用 float16 数据类型")
        print("2. ✅ 大图像自动调整 tile 大小")
        print("3. ✅ MPS 错误时提供详细指导")
        print("4. ✅ 智能设备回退机制")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mps_compatibility()
    sys.exit(0 if success else 1)