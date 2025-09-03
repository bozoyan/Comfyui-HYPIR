#!/usr/bin/env python3
"""
测试带设备选择的新节点
"""
import os
import sys

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_new_node():
    """测试新的带设备选择的节点"""
    try:
        print("🧪 测试新节点导入...")
        from hypir_advanced_node import HYPIRAdvancedRestorationWithDevice, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        
        print("✅ 节点导入成功")
        
        print("\\n📋 节点信息:")
        print(f"可用节点: {list(NODE_CLASS_MAPPINGS.keys())}")
        print(f"显示名称: {NODE_DISPLAY_NAME_MAPPINGS}")
        
        print("\\n🔧 测试新节点配置...")
        input_types = HYPIRAdvancedRestorationWithDevice.INPUT_TYPES()
        
        print(f"节点函数名: {HYPIRAdvancedRestorationWithDevice.FUNCTION}")
        print(f"节点类别: {HYPIRAdvancedRestorationWithDevice.CATEGORY}")
        print(f"返回类型: {HYPIRAdvancedRestorationWithDevice.RETURN_TYPES}")
        
        # 检查设备选项
        device_config = input_types['required']['device']
        print(f"设备选项: {device_config[0]}")
        print(f"默认设备: {device_config[1]['default']}")
        
        # 验证参数顺序
        required_params = list(input_types['required'].keys())
        print(f"\\n📝 参数顺序: {required_params}")
        
        # 验证设备参数位置
        device_index = required_params.index('device')
        print(f"设备参数位置: 第 {device_index + 1} 个参数")
        
        print("\\n🎉 新节点测试完成！现在您可以:")
        print("1. 重启 ComfyUI")
        print("2. 查找 'HYPIR Advanced Restoration (Device Select)' 节点")
        print("3. 使用新的设备选择功能")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_new_node()
    sys.exit(0 if success else 1)