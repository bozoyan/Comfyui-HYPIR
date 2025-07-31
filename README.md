# HYPIR ComfyUI Plugin

This is a ComfyUI plugin for [HYPIR (Harnessing Diffusion-Yielded Score Priors for Image Restoration)](https://github.com/XPixelGroup/HYPIR), a state-of-the-art image restoration model based on Stable Diffusion 2.1.

## Features

- **Image Restoration**: Restore and enhance low-quality images using diffusion priors
- **Batch Processing**: Process multiple images at once
- **Advanced Controls**: Fine-tune model parameters for optimal results
- **Model Management**: Load and reuse HYPIR models efficiently
- **Upscaling**: Built-in upscaling capabilities (1x to 8x)

## Installation

### 1. Install the Plugin

Place this folder in your ComfyUI `custom_nodes` directory:
```
ComfyUI/custom_nodes/Comfyui-HYPIR/
```

### 2. Install HYPIR Dependencies

Navigate to the HYPIR folder and install the required dependencies:

```bash
cd ComfyUI/custom_nodes/Comfyui-HYPIR/HYPIR
pip install -r requirements.txt
```

### 3. Model Download (Automatic)

The plugin will automatically download the required models on first use:

#### HYPIR Model
The HYPIR restoration model will be downloaded to:
```
ComfyUI/models/HYPIR/HYPIR_sd2.pth
```

#### Base Model (Stable Diffusion 2.1)
The base Stable Diffusion 2.1 model will be automatically downloaded when needed to:
```
ComfyUI/models/HYPIR/stable-diffusion-2-1-base/
```

**Manual Download (Optional):**

**HYPIR Model:**
If you prefer to download manually, you can get the HYPIR model from:
- **HuggingFace**: [HYPIR_sd2.pth](https://huggingface.co/lxq007/HYPIR/tree/main)
- **OpenXLab**: [HYPIR_sd2.pth](https://openxlab.org.cn/models/detail/linxqi/HYPIR/tree/main)

Place the `HYPIR_sd2.pth` file in:
- Plugin directory: `ComfyUI/custom_nodes/Comfyui-HYPIR/`
- ComfyUI models directory: `ComfyUI/models/checkpoints/`
- Or let the plugin automatically manage it in `ComfyUI/models/HYPIR/`

**Base Model:**
The base Stable Diffusion 2.1 model can be downloaded manually from:
- **HuggingFace**: [stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)

Place the base model in:
```
ComfyUI/models/HYPIR/stable-diffusion-2-1-base/
```

**Note:** The plugin will automatically check for the base model in the HYPIR directory first. If not found, it will automatically download it from HuggingFace.

## Usage

### Basic Image Restoration

1. Add the **HYPIR Image Restoration** node to your workflow
2. Connect an image to the `image` input
3. Set your prompt (e.g., "high quality, detailed")
4. Adjust the upscale factor if needed (1-8x)
5. Set a seed for reproducible results (or use -1 for random)
6. Run the workflow

### Advanced Image Restoration

1. Add the **HYPIR Advanced Restoration** node
2. This node provides additional control over:
   - `model_t`: Model timestep (default: 200)
   - `coeff_t`: Coefficient timestep (default: 200)
   - `lora_rank`: LoRA rank (default: 256)
   - `patch_size`: Processing patch size (default: 512)

### Model Management

For better performance when processing multiple images:

1. Use **HYPIR Model Loader** to load the model once
2. Connect the loaded model to **HYPIR Restore with Model** nodes
3. This avoids reloading the model for each image

### Automatic Model Download

The plugin automatically downloads required models on first use:

1. **HYPIR Model Manager** - Check model status and manually download models
2. **HYPIR Model Path Getter** - Get model paths with optional auto-download
3. Models are stored in `ComfyUI/models/HYPIR/` for easy management

### Base Model Selection

The plugin now supports dropdown selection for base models:
- **Available Models**: Currently supports `stable-diffusion-2-1-base`
- **Smart Download**: Only downloads when not found locally
- **Local Priority**: Uses local models when available
- **Automatic Management**: Handles model paths automatically

### Batch Processing

1. Use **HYPIR Batch Restoration** to process multiple images
2. Connect a batch of images to the `images` input
3. All images will be processed with the same prompt and settings

## Node Reference

### HYPIR Image Restoration
- **Inputs**: image, prompt, upscale_factor, seed, model_name, base_model_path
- **Outputs**: restored_image, status_message
- **Description**: Basic image restoration with default parameters

### HYPIR Advanced Restoration
- **Inputs**: image, prompt, upscale_factor, seed, model_name, base_model_path, model_t, coeff_t, lora_rank, patch_size
- **Outputs**: restored_image, status_message
- **Description**: Advanced restoration with full parameter control

### HYPIR Batch Restoration
- **Inputs**: images, prompt, upscale_factor, seed, model_name, base_model_path
- **Outputs**: restored_images, status_message
- **Description**: Process multiple images at once

### HYPIR Model Loader
- **Inputs**: model_name, base_model_path, model_t, coeff_t, lora_rank
- **Outputs**: hypir_model, status_message
- **Description**: Load and cache HYPIR model for reuse

### HYPIR Restore with Model
- **Inputs**: hypir_model, image, prompt, upscale_factor, seed
- **Outputs**: restored_image, status_message
- **Description**: Use pre-loaded model for restoration

### HYPIR Model Manager
- **Inputs**: action, model_name
- **Outputs**: status_message, result_message
- **Description**: Manage HYPIR models (check status, download, list)

### HYPIR Model Path Getter
- **Inputs**: model_name, auto_download
- **Outputs**: model_path, status_message
- **Description**: Get model path with optional auto-download

## Configuration

You can modify the default settings in `hypir_config.py`:

```python
HYPIR_CONFIG = {
    "default_weight_path": "HYPIR_sd2.pth",
    "default_base_model_path": "stable-diffusion-2-1-base",
    "available_base_models": ["stable-diffusion-2-1-base"],
    "model_t": 200,
    "coeff_t": 200,
    "lora_rank": 256,
    # ... more settings
}
```

### Model Path Management

The plugin includes intelligent model path management:

- **HYPIR Model**: Automatically downloaded to `ComfyUI/models/HYPIR/HYPIR_sd2.pth`
- **Base Model**: Automatically downloaded to `ComfyUI/models/HYPIR/stable-diffusion-2-1-base/` when needed
- **Local Priority**: The plugin checks for local models first before downloading
- **Automatic Download**: Only downloads when models are not found locally

## Tips for Best Results

1. **Prompts**: Use descriptive prompts that match the image content
   - For portraits: "high quality portrait, detailed face, sharp features"
   - For landscapes: "high quality landscape, detailed scenery, sharp focus"
   - For general images: "high quality, detailed, sharp, clear"

2. **Upscaling**: 
   - Use 1x for restoration without size change
   - Use 2x-4x for moderate upscaling
   - Use 8x for maximum upscaling (may be slower)

3. **Parameters**:
   - Higher `model_t` values (200-500) for stronger restoration
   - Higher `coeff_t` values (200-500) for more aggressive enhancement
   - Higher `lora_rank` (256-512) for better quality (uses more memory)

4. **Memory Management**:
   - Use smaller `patch_size` (256-512) if you encounter memory issues
   - Process images in smaller batches
   - Use the Model Loader node to avoid repeated model loading

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure HYPIR dependencies are installed
   ```bash
   cd HYPIR
   pip install -r requirements.txt
   ```

2. **Model Not Found**: The plugin will automatically download missing models
   - Check internet connection for automatic download
   - **HYPIR Model**: Place `HYPIR_sd2.pth` in plugin directory or ComfyUI models directory
   - **Base Model**: Place `stable-diffusion-2-1-base` folder in `ComfyUI/models/HYPIR/`
   - The plugin will automatically check and download missing models

3. **CUDA Out of Memory**: 
   - Reduce `patch_size` to 256
   - Process smaller images
   - Use CPU mode if available

4. **Slow Processing**:
   - Use Model Loader to avoid repeated model loading
   - Reduce upscale factor
   - Use smaller images

### Error Messages

- **"Error importing HYPIR"**: Install HYPIR dependencies
- **"Error loading model"**: Check model file path and permissions
- **"Error during restoration"**: Check image format and size

## License

This plugin is provided under the same license as the original HYPIR project. Please refer to the [HYPIR repository](https://github.com/XPixelGroup/HYPIR) for license details.

## Acknowledgments

- Original HYPIR implementation by [XPixelGroup](https://github.com/XPixelGroup/HYPIR)
- ComfyUI community for the excellent framework

## Support

For issues related to this ComfyUI plugin, please check:
1. This README for troubleshooting
2. The original [HYPIR repository](https://github.com/XPixelGroup/HYPIR) for model-specific issues
3. ComfyUI documentation for general ComfyUI questions 