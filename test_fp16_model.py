#!/usr/bin/env python3
"""
测试 FP16 模型是否能正确加载
"""
import os
import sys

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_fp16_model_loading():
    """测试 FP16 模型加载"""
    try:
        from hypir_config import get_base_model_path
        
        # 获取基础模型路径
        base_model_path = get_base_model_path('stable-diffusion-2-1-base')
        print(f"Base model path: {base_model_path}")
        
        # 验证路径存在
        if not os.path.exists(base_model_path):
            print(f"❌ 路径不存在: {base_model_path}")
            return False
            
        # 检查关键文件
        unet_path = os.path.join(base_model_path, "unet", "diffusion_pytorch_model.safetensors")
        vae_path = os.path.join(base_model_path, "vae", "diffusion_pytorch_model.safetensors")
        
        print(f"UNet 文件存在: {os.path.exists(unet_path)}")
        print(f"VAE 文件存在: {os.path.exists(vae_path)}")
        
        if not os.path.exists(unet_path):
            print(f"❌ UNet 文件不存在: {unet_path}")
            return False
            
        if not os.path.exists(vae_path):
            print(f"❌ VAE 文件不存在: {vae_path}")
            return False
        
        # 尝试使用 diffusers 加载
        print("\\n测试 diffusers 加载...")
        try:
            from diffusers import UNet2DConditionModel, AutoencoderKL
            
            # 测试加载 UNet
            print("加载 UNet...")
            unet = UNet2DConditionModel.from_pretrained(
                base_model_path, 
                subfolder="unet",
                torch_dtype=None  # 让它自动检测
            )
            print(f"✓ UNet 加载成功，参数数量: {sum(p.numel() for p in unet.parameters()):,}")
            
            # 测试加载 VAE
            print("加载 VAE...")
            vae = AutoencoderKL.from_pretrained(
                base_model_path,
                subfolder="vae",
                torch_dtype=None
            )
            print(f"✓ VAE 加载成功，参数数量: {sum(p.numel() for p in vae.parameters()):,}")
            
            print("\\n🎉 所有组件加载成功！您的 FP16 模型配置正确。")
            return True
            
        except Exception as e:
            print(f"❌ diffusers 加载失败: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fp16_model_loading()
    sys.exit(0 if success else 1)