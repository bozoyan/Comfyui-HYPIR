# HYPIR ComfyUI Plugin 修改日志

## 2024年修改 - 基础模型路径优化

### 修改内容

1. **配置文件修改** (`hypir_config.py`)
   - 添加了 `available_base_models` 列表，包含可用的基础模型
   - 新增 `get_base_model_path()` 函数，实现智能基础模型路径管理
   - 只有在 `ComfyUI\models\HYPIR` 没有基础模型文件夹时才自动下载

2. **节点文件修改**
   - **`hypir_node.py`**: 将 `base_model_path` 从字符串输入改为列表选择
   - **`hypir_advanced_node.py`**: 将 `base_model_path` 从字符串输入改为列表选择
   - 所有节点现在都使用 `get_base_model_path()` 函数来处理基础模型路径

### 功能特性

- **智能路径管理**: 自动检查本地是否有基础模型，避免重复下载
- **列表选择**: 用户可以从预定义的基础模型列表中选择，而不是手动输入
- **自动下载**: 只有在需要时才自动下载基础模型
- **本地优先**: 优先使用本地已存在的基础模型

### 技术细节

- 基础模型路径检查逻辑：
  1. 检查 `ComfyUI\models\HYPIR` 目录是否存在
  2. 如果存在，检查是否有 `stable-diffusion-2-1-base` 文件夹
  3. 如果有本地模型，使用本地路径 `ComfyUI/models/HYPIR/stable-diffusion-2-1-base/`
  4. 如果没有，返回 HuggingFace 路径 `stabilityai/stable-diffusion-2-1-base` 让 diffusers 自动下载

### 兼容性

- 保持向后兼容性
- 现有工作流无需修改
- 自动处理模型路径转换

### 使用说明

用户现在可以从下拉列表中选择基础模型，系统会自动处理：
- 如果本地有模型，直接使用
- 如果本地没有模型，自动下载
- 无需手动配置模型路径

### 修正说明

- **基础模型路径**: 修正了基础模型路径配置，使用本地路径 `stable-diffusion-2-1-base` 而不是完整的 HuggingFace 路径
- **智能路径管理**: 本地模型存储在 `ComfyUI/models/HYPIR/stable-diffusion-2-1-base/` 目录下
- **自动下载**: 当本地没有模型时，自动从 HuggingFace 下载 `stabilityai/stable-diffusion-2-1-base` 