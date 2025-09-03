import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import sys

# Try to import folder_paths (ComfyUI specific)
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

# Add HYPIR to path
HYPIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HYPIR")
if HYPIR_PATH not in sys.path:
    sys.path.append(HYPIR_PATH)

# Import SD2Enhancer with better error handling
SD2Enhancer = None
try:
    from HYPIR.enhancer.sd2 import SD2Enhancer
    print("Successfully imported SD2Enhancer")
except ImportError as e:
    print(f"Error importing HYPIR.enhancer.sd2: {e}")
    print("Please make sure HYPIR is properly installed in the HYPIR folder")
    print("Trying alternative import...")
    try:
        # Try direct import from sd2 module
        import sys
        import os
        sd2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HYPIR", "enhancer", "sd2.py")
        if os.path.exists(sd2_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("sd2", sd2_path)
            if spec is not None and spec.loader is not None:
                sd2_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(sd2_module)
                SD2Enhancer = sd2_module.SD2Enhancer
                print("Successfully imported SD2Enhancer via direct module loading")
            else:
                print(f"Could not create module spec for {sd2_path}")
        else:
            print(f"sd2.py not found at {sd2_path}")
    except Exception as e2:
        print(f"Alternative import also failed: {e2}")
        SD2Enhancer = None

from hypir_config import HYPIR_CONFIG, get_default_weight_path, get_base_model_path
from model_downloader import get_hypir_model_path

def get_available_devices():
    """检测可用的设备类型"""
    devices = ["auto"]
    
    # 检查 CUDA
    if torch.cuda.is_available():
        devices.append("cuda")
    
    # 检查 MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append("mps")
    
    # CPU 总是可用
    devices.append("cpu")
    
    # 检查 DirectML (Windows)
    try:
        import torch_directml
        if torch_directml.is_available():
            devices.append("dml")
    except ImportError:
        pass
    
    return devices

def get_optimal_device(preferred_device="auto"):
    """根据偏好和可用性选择最优设备"""
    if preferred_device == "auto":
        # 按优先级自动选择：mps > cuda > cpu
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    # 验证用户选择的设备是否可用
    if preferred_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif preferred_device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    elif preferred_device == "cpu":
        return "cpu"
    elif preferred_device == "dml":
        try:
            import torch_directml
            if torch_directml.is_available():
                return "dml"
        except ImportError:
            pass
    
    # 如果首选设备不可用，回退到自动选择
    print(f"Preferred device '{preferred_device}' not available, falling back to auto selection")
    return get_optimal_device("auto")

class HYPIRAdvancedRestoration:
    """Advanced HYPIR Image Restoration Node for ComfyUI with more control options"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "high quality, detailed", "multiline": True}),
                "upscale_factor": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "model_name": (["HYPIR_sd2"],),
                "base_model_path": (HYPIR_CONFIG["available_base_models"],),
                "model_t": ("INT", {"default": HYPIR_CONFIG["model_t"], "min": 1, "max": 1000}),
                "coeff_t": ("INT", {"default": HYPIR_CONFIG["coeff_t"], "min": 1, "max": 1000}),
                "lora_rank": ("INT", {"default": HYPIR_CONFIG["lora_rank"], "min": 1, "max": 512}),
                "patch_size": ("INT", {"default": HYPIR_CONFIG["patch_size"], "min": 256, "max": 1024, "step": 64}),
                "encode_patch_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "decode_patch_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "unload_model_after": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "restore_image_advanced"
    CATEGORY = "HYPIR"
    
    def __init__(self):
        self.hypir = None
        self.current_config = None
    
    def create_enhancer(self, model_name, base_model_path, model_t, coeff_t, lora_rank):
        """Create HYPIR enhancer with custom parameters"""
        # Check if SD2Enhancer is available
        if SD2Enhancer is None:
            raise ValueError("SD2Enhancer is not available. Please check the HYPIR installation and imports.")
        
        # 使用自动设备选择
        selected_device = get_optimal_device("auto")
        print(f"Auto-selected device: {selected_device}")
        
        # Get model path from model name
        weight_path = get_hypir_model_path(model_name)
        if not weight_path:
            raise ValueError(f"Could not get model path for {model_name}")
        
        # 处理相对路径，转换为绝对路径
        if not os.path.isabs(weight_path):
            # 尝试从 ComfyUI models 目录查找
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 向上查找ComfyUI根目录
            for i in range(5):  # 最多向上5层
                parent = os.path.dirname(current_dir)
                if os.path.exists(os.path.join(parent, "models")):
                    models_dir = os.path.join(parent, "models")
                    absolute_path = os.path.join(models_dir, weight_path)
                    if os.path.exists(absolute_path):
                        weight_path = absolute_path
                        print(f"Found model: {weight_path}")
                    else:
                        print(f"Model file not found: {absolute_path}")
                        raise ValueError(f"Model file not found: {absolute_path}")
                    break
                current_dir = parent
            else:
                print("Could not find ComfyUI models directory")
                raise ValueError("Could not find ComfyUI models directory")
        else:
            # 已经是绝对路径，检查是否存在
            if not os.path.exists(weight_path):
                print(f"Model file not found: {weight_path}")
                raise ValueError(f"Model file not found: {weight_path}")
        
        try:
            enhancer = SD2Enhancer(
                base_model_path=base_model_path,
                weight_path=weight_path,
                lora_modules=HYPIR_CONFIG["lora_modules"],
                lora_rank=lora_rank,
                model_t=model_t,
                coeff_t=coeff_t,
                device=selected_device,
            )
            enhancer.init_models()
            return enhancer
        except Exception as e:
            print(f"Error creating HYPIR enhancer: {e}")
            raise e
    
    def restore_image_advanced(self, image, prompt, upscale_factor, seed, model_name, 
                             base_model_path, model_t, coeff_t, lora_rank, patch_size,
                             encode_patch_size, decode_patch_size, batch_size, unload_model_after=False):
        # Set seed if provided
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Check if we need to create a new enhancer
        # 使用新的基础模型路径获取函数
        actual_base_model_path = get_base_model_path(base_model_path)
        current_config = (model_name, actual_base_model_path, model_t, coeff_t, lora_rank)
        if self.hypir is None or self.current_config != current_config:
            try:
                self.hypir = self.create_enhancer(model_name, actual_base_model_path, model_t, coeff_t, lora_rank)
                self.current_config = current_config
                print(f"HYPIR model loaded with parameters: model_t={model_t}, coeff_t={coeff_t}, lora_rank={lora_rank}")
            except Exception as e:
                return (image, f"Error loading model: {str(e)}")
        
        try:
            # Convert image to format expected by HYPIR
            if len(image.shape) == 4:
                # Batch of images - process in smaller batches
                results = []
                total_images = image.shape[0]
                
                for batch_start in range(0, total_images, batch_size):
                    batch_end = min(batch_start + batch_size, total_images)
                    batch_images = image[batch_start:batch_end]
                    
                    batch_results = []
                    for i in range(batch_images.shape[0]):
                        # Convert to PyTorch tensor with correct format
                        # ComfyUI format: (H, W, C) torch.Tensor, 0-1 values
                        # Convert from (H, W, C) to (C, H, W) format for HYPIR
                        img_tensor = batch_images[i].permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
                        # Ensure tensor is contiguous and on the correct device
                        img_tensor = img_tensor.contiguous()
                        
                        result = self.hypir.enhance(
                            lq=img_tensor,
                            prompt=prompt,
                            upscale=upscale_factor,
                            encode_patch_size=encode_patch_size,
                            decode_patch_size=decode_patch_size,
                            return_type="pt"
                        )
                        # Convert back to ComfyUI format
                        result = result.squeeze(0).permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
                        batch_results.append(result)
                    
                    results.extend(batch_results)
                    print(f"Processed batch {batch_start//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}")
                
                output_image = torch.stack(results, dim=0)
            else:
                # Single image
                # Convert to PyTorch tensor with correct format
                # ComfyUI format: (H, W, C) torch.Tensor, 0-1 values
                # Convert from (H, W, C) to (C, H, W) format for HYPIR
                img_tensor = image.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
                # Ensure tensor is contiguous and on the correct device
                img_tensor = img_tensor.contiguous()
                
                result = self.hypir.enhance(
                    lq=img_tensor,
                    prompt=prompt,
                    upscale=upscale_factor,
                    encode_patch_size=encode_patch_size,
                    decode_patch_size=decode_patch_size,
                    return_type="pt"
                )
                # Convert back to ComfyUI format
                output_image = result.squeeze(0).permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            
            status_msg = f"Success! Used prompt: {prompt}\nDevice: {get_optimal_device('auto')}\nParameters: model_t={model_t}, coeff_t={coeff_t}, lora_rank={lora_rank}, patch_size={patch_size}\nEncode patch: {encode_patch_size}, Decode patch: {decode_patch_size}, Batch size: {batch_size}"
            return_value = (output_image, status_msg)
        except Exception as e:
            print(f"HYPIR advanced restoration error: {e}")
            return_value = (image, f"Error during restoration: {str(e)}")

        # 卸载模型逻辑
        if unload_model_after:
            if self.hypir is not None:
                try:
                    del self.hypir
                    self.hypir = None
                    self.current_config = None
                    
                    # 根据设备类型清理内存
                    selected_device = get_optimal_device("auto")
                    if selected_device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif selected_device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception as e:
                    print(f"Error unloading HYPIR model: {e}")

        return return_value


class HYPIRAdvancedRestorationWithDevice:
    """HYPIR Image Restoration Node with Device Selection for ComfyUI"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "high quality, detailed", "multiline": True}),
                "upscale_factor": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "model_name": (["HYPIR_sd2"],),
                "base_model_path": (HYPIR_CONFIG["available_base_models"],),
                "device": (get_available_devices(), {"default": "auto"}),
                "model_t": ("INT", {"default": HYPIR_CONFIG["model_t"], "min": 1, "max": 1000}),
                "coeff_t": ("INT", {"default": HYPIR_CONFIG["coeff_t"], "min": 1, "max": 1000}),
                "lora_rank": ("INT", {"default": HYPIR_CONFIG["lora_rank"], "min": 1, "max": 512}),
                "patch_size": ("INT", {"default": HYPIR_CONFIG["patch_size"], "min": 256, "max": 1024, "step": 64}),
                "encode_patch_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "decode_patch_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "unload_model_after": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "restore_image_with_device"
    CATEGORY = "HYPIR"
    
    def __init__(self):
        self.hypir = None
        self.current_config = None
    
    def create_enhancer(self, model_name, base_model_path, model_t, coeff_t, lora_rank, device="auto"):
        """Create HYPIR enhancer with custom parameters"""
        # Check if SD2Enhancer is available
        if SD2Enhancer is None:
            raise ValueError("SD2Enhancer is not available. Please check the HYPIR installation and imports.")
        
        # 获取最优设备
        selected_device = get_optimal_device(device)
        print(f"Selected device: {selected_device} (requested: {device})")
        
        # Get model path from model name
        weight_path = get_hypir_model_path(model_name)
        if not weight_path:
            raise ValueError(f"Could not get model path for {model_name}")
        
        # 处理相对路径，转换为绝对路径
        if not os.path.isabs(weight_path):
            # 尝试从 ComfyUI models 目录查找
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 向上查找ComfyUI根目录
            for i in range(5):  # 最多向上5层
                parent = os.path.dirname(current_dir)
                if os.path.exists(os.path.join(parent, "models")):
                    models_dir = os.path.join(parent, "models")
                    absolute_path = os.path.join(models_dir, weight_path)
                    if os.path.exists(absolute_path):
                        weight_path = absolute_path
                        print(f"Found model: {weight_path}")
                    else:
                        print(f"Model file not found: {absolute_path}")
                        raise ValueError(f"Model file not found: {absolute_path}")
                    break
                current_dir = parent
            else:
                print("Could not find ComfyUI models directory")
                raise ValueError("Could not find ComfyUI models directory")
        else:
            # 已经是绝对路径，检查是否存在
            if not os.path.exists(weight_path):
                print(f"Model file not found: {weight_path}")
                raise ValueError(f"Model file not found: {weight_path}")
        
        try:
            enhancer = SD2Enhancer(
                base_model_path=base_model_path,
                weight_path=weight_path,
                lora_modules=HYPIR_CONFIG["lora_modules"],
                lora_rank=lora_rank,
                model_t=model_t,
                coeff_t=coeff_t,
                device=selected_device,
            )
            enhancer.init_models()
            return enhancer
        except Exception as e:
            print(f"Error creating HYPIR enhancer: {e}")
            raise e
    
    def restore_image_with_device(self, image, prompt, upscale_factor, seed, model_name, 
                                base_model_path, device, model_t, coeff_t, lora_rank, patch_size,
                                encode_patch_size, decode_patch_size, batch_size, unload_model_after=False):
        # Set seed if provided
        if seed != -1:
            torch.manual_seed(seed)
            selected_device = get_optimal_device(device)
            if selected_device == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Check if we need to create a new enhancer
        # 使用新的基础模型路径获取函数
        actual_base_model_path = get_base_model_path(base_model_path)
        current_config = (model_name, actual_base_model_path, model_t, coeff_t, lora_rank, device)
        if self.hypir is None or self.current_config != current_config:
            try:
                self.hypir = self.create_enhancer(model_name, actual_base_model_path, model_t, coeff_t, lora_rank, device)
                self.current_config = current_config
                print(f"HYPIR model loaded with device: {get_optimal_device(device)}, parameters: model_t={model_t}, coeff_t={coeff_t}, lora_rank={lora_rank}")
            except Exception as e:
                return (image, f"Error loading model: {str(e)}")
        
        try:
            # Convert image to format expected by HYPIR
            if len(image.shape) == 4:
                # Batch of images - process in smaller batches
                results = []
                total_images = image.shape[0]
                
                for batch_start in range(0, total_images, batch_size):
                    batch_end = min(batch_start + batch_size, total_images)
                    batch_images = image[batch_start:batch_end]
                    
                    batch_results = []
                    for i in range(batch_images.shape[0]):
                        # Convert to PyTorch tensor with correct format
                        # ComfyUI format: (H, W, C) torch.Tensor, 0-1 values
                        # Convert from (H, W, C) to (C, H, W) format for HYPIR
                        img_tensor = batch_images[i].permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
                        # Ensure tensor is contiguous and on the correct device
                        img_tensor = img_tensor.contiguous()
                        
                        result = self.hypir.enhance(
                            lq=img_tensor,
                            prompt=prompt,
                            upscale=upscale_factor,
                            encode_patch_size=encode_patch_size,
                            decode_patch_size=decode_patch_size,
                            return_type="pt"
                        )
                        # Convert back to ComfyUI format
                        result = result.squeeze(0).permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
                        batch_results.append(result)
                    
                    results.extend(batch_results)
                    print(f"Processed batch {batch_start//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}")
                
                output_image = torch.stack(results, dim=0)
            else:
                # Single image
                # Convert to PyTorch tensor with correct format
                # ComfyUI format: (H, W, C) torch.Tensor, 0-1 values
                # Convert from (H, W, C) to (C, H, W) format for HYPIR
                img_tensor = image.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
                # Ensure tensor is contiguous and on the correct device
                img_tensor = img_tensor.contiguous()
                
                result = self.hypir.enhance(
                    lq=img_tensor,
                    prompt=prompt,
                    upscale=upscale_factor,
                    encode_patch_size=encode_patch_size,
                    decode_patch_size=decode_patch_size,
                    return_type="pt"
                )
                # Convert back to ComfyUI format
                output_image = result.squeeze(0).permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            
            status_msg = f"Success! Used prompt: {prompt}\nDevice: {get_optimal_device(device)}\nParameters: model_t={model_t}, coeff_t={coeff_t}, lora_rank={lora_rank}, patch_size={patch_size}\nEncode patch: {encode_patch_size}, Decode patch: {decode_patch_size}, Batch size: {batch_size}"
            return_value = (output_image, status_msg)
        except Exception as e:
            print(f"HYPIR advanced restoration error: {e}")
            return_value = (image, f"Error during restoration: {str(e)}")

        # 卸载模型逻辑
        if unload_model_after:
            if self.hypir is not None:
                try:
                    del self.hypir
                    self.hypir = None
                    self.current_config = None
                    
                    # 根据设备类型清理内存
                    selected_device = get_optimal_device(device)
                    if selected_device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif selected_device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    elif selected_device == "dml":
                        try:
                            import torch_directml
                            torch_directml.empty_cache()
                        except:
                            pass
                except Exception as e:
                    print(f"Error unloading HYPIR model: {e}")

        return return_value

# Node mappings
NODE_CLASS_MAPPINGS = {
    "HYPIRAdvancedRestoration": HYPIRAdvancedRestoration,
    "HYPIRAdvancedRestorationWithDevice": HYPIRAdvancedRestorationWithDevice,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYPIRAdvancedRestoration": "HYPIR Advanced Restoration",
    "HYPIRAdvancedRestorationWithDevice": "HYPIR Advanced Restoration (Device Select)",
} 