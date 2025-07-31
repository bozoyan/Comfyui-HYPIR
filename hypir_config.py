import os

# Try to import folder_paths (ComfyUI specific)
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

from model_downloader import get_hypir_model_path

HYPIR_CONFIG = {
    "default_weight_path": "HYPIR_sd2.pth",
    "default_base_model_path": "stable-diffusion-2-1-base",
    "available_base_models": ["stable-diffusion-2-1-base"],
    "model_t": 200,
    "coeff_t": 100,
    "lora_rank": 256,
    "lora_modules": [
        "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
        "conv_shortcut", "conv_out", "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
    ],
    "patch_size": 512,
    "vae_scale_factor": 8,
}

def get_default_weight_path():
    """Get the default weight path without downloading"""
    return HYPIR_CONFIG["default_weight_path"]

def get_base_model_path(base_model_name):
    """获取基础模型路径，只有在ComfyUI\models\HYPIR没有基础模型文件夹时才自动下载"""
    # 检查ComfyUI models目录下的HYPIR文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for i in range(5):  # 最多向上5层
        parent = os.path.dirname(current_dir)
        if os.path.exists(os.path.join(parent, "models")):
            models_dir = os.path.join(parent, "models")
            hypir_dir = os.path.join(models_dir, "HYPIR")
            
            # 检查HYPIR目录是否存在
            if os.path.exists(hypir_dir):
                # 如果HYPIR目录存在，检查是否有基础模型文件夹
                # 对于stable-diffusion-2-1-base，检查是否有对应的文件夹
                base_model_folder = os.path.join(hypir_dir, base_model_name)
                if os.path.exists(base_model_folder):
                    print(f"找到本地基础模型: {base_model_folder}")
                    return base_model_folder
                else:
                    print(f"本地没有基础模型，将自动下载: stabilityai/{base_model_name}")
                    return f"stabilityai/{base_model_name}"
            else:
                # 如果HYPIR目录不存在，创建目录并返回原始路径（让diffusers自动下载）
                os.makedirs(hypir_dir, exist_ok=True)
                print(f"创建HYPIR目录: {hypir_dir}")
                print(f"将自动下载基础模型: stabilityai/{base_model_name}")
                return f"stabilityai/{base_model_name}"
            break
        current_dir = parent
    
    # 如果找不到ComfyUI目录，返回原始路径
    print(f"无法找到ComfyUI目录，使用原始路径: stabilityai/{base_model_name}")
    return f"stabilityai/{base_model_name}" 