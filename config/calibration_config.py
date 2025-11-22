# config/calibration_config.py
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class CalibrationConfig:
    """校准分析配置类"""
    model_type: str  # "deepseek", "qwen7b", "qwen32b"
    output_type: str  # "probability", "logits"
    use_multi_gpu: bool = False
    
    # 数据配置
    prediction_file_pattern: str = "./output/predict_{model_type}_{output_type}.xlsx"
    human_data_path: str = "./data/question_human.xlsx"
    question_data_path: str = "./data/question.xlsx"
    
    # 分析配置
    n_bins: int = 10
    plot_size: tuple = (16, 6)
    dpi: int = 300
    
    # 输出配置
    output_dir: str = "./output"
    plot_name_pattern: str = "{model_type}_calibration_analysis_{output_type}.png"

class CalibrationConfigManager:
    """校准配置管理器"""
    
    def __init__(self):
        self.config = CalibrationConfig(
            model_type="deepseek",  # 修改这里切换模型
            output_type="logits"    # 修改这里切换输出类型
        )
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_prediction_file_path(self):
        """获取预测文件路径"""
        if self.config.use_multi_gpu and "qwen32b" in self.config.model_type:
            base_name = f"predict_{self.config.model_type}_{self.config.output_type}_multi_gpu"
        else:
            base_name = f"predict_{self.config.model_type}_{self.config.output_type}"
        
        return f"./output/{base_name}.xlsx"
    
    def get_plot_file_path(self):
        """获取绘图文件路径"""
        if self.config.use_multi_gpu and "qwen32b" in self.config.model_type:
            base_name = f"{self.config.model_type}_calibration_analysis_{self.config.output_type}_multi_gpu"
        else:
            base_name = f"{self.config.model_type}_calibration_analysis_{self.config.output_type}"
        
        return f"./output/{base_name}.png"
    
    def get_model_display_name(self):
        """获取模型显示名称"""
        display_names = {
            "deepseek": "DeepSeek",
            "qwen7b": "Qwen-7B",
            "qwen32b": "Qwen-32B"
        }
        base_name = display_names.get(self.config.model_type, self.config.model_type)
        
        if self.config.use_multi_gpu and "qwen32b" in self.config.model_type:
            return f"{base_name} (Multi-GPU)"
        else:
            return base_name

# 全局配置实例
config_manager = CalibrationConfigManager()
