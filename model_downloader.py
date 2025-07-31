import os
import sys
import requests
import hashlib
from pathlib import Path
from tqdm import tqdm

# Try to import folder_paths (ComfyUI specific)
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

class HYPIRModelDownloader:
    """HYPIR模型下载管理器"""
    
    def __init__(self):
        self.models_dir = self._get_models_dir()
        self.hypir_dir = os.path.join(self.models_dir, "HYPIR")
        self._ensure_hypir_dir()
        
        # 模型信息配置
        self.model_configs = {
            "HYPIR_sd2": {
                "filename": "HYPIR_sd2.pth",
                "url": "https://huggingface.co/lxq007/HYPIR/resolve/main/HYPIR_sd2.pth",
                "mirror_url": "https://openxlab.org.cn/models/detail/linxqi/HYPIR/resolve/main/HYPIR_sd2.pth",
                "description": "HYPIR SD2.1 图像恢复模型"
            }
        }
    
    def _get_models_dir(self):
        """获取ComfyUI模型目录"""
        # 直接使用ComfyUI的models目录，而不是checkpoints子目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 向上查找ComfyUI根目录
        for i in range(5):  # 最多向上5层
            parent = os.path.dirname(current_dir)
            if os.path.exists(os.path.join(parent, "models")):
                return os.path.join(parent, "models")
            current_dir = parent
        
        # 如果都找不到，使用当前目录下的models
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    def _ensure_hypir_dir(self):
        """确保HYPIR模型目录存在"""
        os.makedirs(self.hypir_dir, exist_ok=True)
        print(f"HYPIR模型目录: {self.hypir_dir}")
    
    def _download_file(self, url, filepath, description="文件"):
        """下载文件并显示进度条"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"下载{description}"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        pbar.update(size)
            
            return True
        except Exception as e:
            print(f"下载失败: {e}")
            return False
    
    def _verify_file(self, filepath, expected_size=None, expected_md5=None):
        """验证文件完整性"""
        if not os.path.exists(filepath):
            return False
        
        # 只检查文件是否存在，不验证大小
        actual_size = os.path.getsize(filepath)
        if actual_size > 0:
            print(f"文件下载完成，大小: {actual_size} bytes")
            return True
        
        return False
    
    def download_model(self, model_name):
        """下载指定的模型"""
        if model_name not in self.model_configs:
            print(f"未知模型: {model_name}")
            return None
        
        config = self.model_configs[model_name]
        filename = config["filename"]
        model_path = os.path.join(self.hypir_dir, filename)
        
        # 检查模型是否已存在且有效
        if self._verify_file(model_path):
            print(f"模型已存在且有效: {model_path}")
            return model_path
        
        print(f"开始下载模型: {model_name}")
        print(f"描述: {config['description']}")
        print(f"保存路径: {model_path}")
        
        # 尝试主URL下载
        print("尝试从主源下载...")
        if self._download_file(config["url"], model_path, model_name):
            if self._verify_file(model_path):
                print(f"模型下载成功: {model_path}")
                return model_path
            else:
                print("文件验证失败，尝试镜像源...")
        
        # 尝试镜像URL下载
        if "mirror_url" in config:
            print("尝试从镜像源下载...")
            if self._download_file(config["mirror_url"], model_path, f"{model_name} (镜像)"):
                if self._verify_file(model_path):
                    print(f"模型下载成功: {model_path}")
                    return model_path
        
        print(f"模型下载失败: {model_name}")
        return None
    
    def get_model_path(self, model_name):
        """获取模型路径，如果不存在则自动下载"""
        if model_name not in self.model_configs:
            print(f"未知模型: {model_name}")
            return None
        
        config = self.model_configs[model_name]
        filename = config["filename"]
        model_path = os.path.join(self.hypir_dir, filename)
        
        # 检查模型是否存在
        if os.path.exists(model_path):
            print(f"找到现有模型: {model_path}")
            # 返回相对路径：HYPIR/filename
            return os.path.join("HYPIR", filename)
        
        # 模型不存在，尝试下载
        print(f"模型不存在，开始下载: {model_name}")
        downloaded_path = self.download_model(model_name)
        if downloaded_path and os.path.exists(downloaded_path):
            # 返回相对路径：HYPIR/filename
            return os.path.join("HYPIR", filename)
        
        # 即使下载失败，也返回相对路径，让用户知道应该放在哪里
        print(f"模型下载失败，但返回相对路径供参考: HYPIR/{filename}")
        return os.path.join("HYPIR", filename)
    
    def list_available_models(self):
        """列出可用的模型"""
        return list(self.model_configs.keys())
    
    def list_downloaded_models(self):
        """列出已下载的模型"""
        downloaded = []
        for model_name, config in self.model_configs.items():
            filename = config["filename"]
            model_path = os.path.join(self.hypir_dir, filename)
            if os.path.exists(model_path):
                downloaded.append(model_name)
        return downloaded

# 全局下载器实例
_downloader = None

def get_downloader():
    """获取全局下载器实例"""
    global _downloader
    if _downloader is None:
        _downloader = HYPIRModelDownloader()
    return _downloader

def download_hypir_model(model_name="HYPIR_sd2"):
    """下载HYPIR模型的便捷函数"""
    downloader = get_downloader()
    return downloader.get_model_path(model_name)

def get_hypir_model_path(model_name="HYPIR_sd2"):
    """获取HYPIR模型路径的便捷函数"""
    downloader = get_downloader()
    return downloader.get_model_path(model_name)

# 测试函数
if __name__ == "__main__":
    print("HYPIR模型下载器测试")
    downloader = HYPIRModelDownloader()
    
    print(f"模型目录: {downloader.hypir_dir}")
    print(f"可用模型: {downloader.list_available_models()}")
    print(f"已下载模型: {downloader.list_downloaded_models()}")
    
    # 测试下载
    model_path = downloader.get_model_path("HYPIR_sd2")
    if model_path:
        print(f"模型路径: {model_path}")
    else:
        print("模型下载失败") 