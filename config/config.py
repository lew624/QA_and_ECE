# config.py
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    """模型配置类"""
    model_type: str  # "deepseek", "qwen7b", "qwen32b"
    output_type: str  # "probability", "logits"
    use_multi_gpu: bool = False
    
    # API配置
    api_key: str = None
    api_url: str = "https://api.deepseek.com/v1/chat/completions"
    
    # 本地模型配置
    model_path: str = None
    device: str = "auto"
    quantization: bool = True
    
    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 512

@dataclass
class DataConfig:
    """数据配置类"""
    excel_path: str = "./data/question.xlsx"
    sheet_name: str = "Sheet1"

@dataclass
class OutputConfig:
    """输出配置类"""
    output_dir: str = "./output"
    save_json: bool = True
    save_excel: bool = True
    save_csv: bool = True

class ProjectConfig:
    """项目配置管理器"""
    
    def __init__(self):
        self.model = ModelConfig(
            model_type="deepseek",  # 修改这里切换模型
            output_type="logits",   # 修改这里切换输出类型
            api_key=os.getenv("DEEPSEEK_API_KEY", "your_api_key_here")
        )
        
        self.data = DataConfig()
        self.output = OutputConfig()
    
    def update_model_config(self, **kwargs):
        """更新模型配置"""
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
    
    def get_output_paths(self, model_type=None, output_type=None):
        """获取输出文件路径"""
        model_type = model_type or self.model.model_type
        output_type = output_type or self.model.output_type
        
        base_name = f"predict_{model_type}_{output_type}"
        if self.model.use_multi_gpu and "qwen32b" in model_type:
            base_name += "_multi_gpu"
        
        return {
            "json": f"{self.output.output_dir}/{base_name}.json",
            "excel": f"{self.output.output_dir}/{base_name}.xlsx", 
            "csv": f"{self.output.output_dir}/{base_name}.csv"
        }

# 全局配置实例
config = ProjectConfig()
