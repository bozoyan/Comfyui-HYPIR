#!/usr/bin/env python3
"""
下载完整的 Stable Diffusion 2.1 基础模型
"""
import os
from diffusers import StableDiffusionPipeline

def download_sd21_base():
    """下载 Stable Diffusion 2.1 基础模型"""
    model_id = "stabilityai/stable-diffusion-2-1-base"
    save_path = "/Volumes/BO/AI/models/HYPIR/stable-diffusion-2-1-base-complete"
    
    print(f"正在下载 {model_id} 到 {save_path}")
    print("这可能需要几分钟时间...")
    
    try:
        # 使用 diffusers 下载并保存模型
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=None,  # 使用原始精度
        )
        
        # 保存模型到指定路径
        pipeline.save_pretrained(save_path)
        print(f"模型下载完成！保存在: {save_path}")
        
        # 验证下载的文件
        print("\\n验证下载的文件结构:")
        for root, dirs, files in os.walk(save_path):
            level = root.replace(save_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
                
        return save_path
        
    except Exception as e:
        print(f"下载失败: {e}")
        return None

if __name__ == "__main__":
    download_sd21_base()