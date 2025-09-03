#!/usr/bin/env python3
"""
测试设备选择功能
"""
import os
import sys

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_device_selection():
    """测试设备选择功能"""
    try:
        # 测试设备检测函数
        from hypir_advanced_node import get_available_devices, get_optimal_device
        
        print("🔍 检测可用设备...")
        available_devices = get_available_devices()
        print(f"可用设备: {available_devices}")
        
        print("\\n🎯 测试设备选择...")
        
        # 测试自动选择
        auto_device = get_optimal_device("auto")
        print(f"自动选择设备: {auto_device}")
        
        # 测试各种设备选择
        test_devices = ["cuda", "mps", "cpu", "dml", "invalid_device"]
        
        for device in test_devices:
            selected = get_optimal_device(device)
            print(f"请求 '{device}' -> 实际使用 '{selected}'")
        
        print("\\n📱 测试 ComfyUI 节点配置...")
        from hypir_advanced_node import HYPIRAdvancedRestoration
        
        # 获取输入类型
        input_types = HYPIRAdvancedRestoration.INPUT_TYPES()
        device_options = input_types["required"]["device"][0]
        device_default = input_types["required"]["device"][1]["default"]
        
        print(f"设备选项: {device_options}")
        print(f"默认设备: {device_default}")
        
        print("\\n🎉 设备选择功能测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_device_selection()
    sys.exit(0 if success else 1)