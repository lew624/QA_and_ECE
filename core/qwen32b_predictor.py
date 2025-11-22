# qwen32b_predictor.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from accelerate import infer_auto_device_map, init_empty_weights
from typing import Dict
import os

from core.base_predictor import BasePredictor
from config.config import config

class LocalPredictor(BasePredictor):
    """本地模型预测器"""
    
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
    
    def initialize(self):
        """初始化本地模型"""
        print(f"正在加载 {config.model.model_type} 模型...")
        
        # 设置模型路径
        if "qwen32b" in config.model.model_type:
            model_path = "/root/autodl-fs/models/Qwen/Qwen2.5-32B-Instruct"
        else:  # qwen7b
            model_path = "./model/qwen/Qwen2.5-7B-Instruct"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        # 多GPU配置
        num_gpus = torch.cuda.device_count()
        
        if num_gpus > 1 and config.model.use_multi_gpu:
            print(f"使用 {num_gpus} 个GPU进行模型并行")
            device_map = "auto"
            
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            
        else:
            # 单GPU或CPU配置
            if config.model.quantization:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                
                # 生成设备映射
                config_obj = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                with init_empty_weights():
                    dummy_model = AutoModelForCausalLM.from_config(config_obj, trust_remote_code=True)
                max_memory = {0: "22GiB", "cpu": "99GiB"}
                device_map = infer_auto_device_map(
                    dummy_model,
                    max_memory=max_memory,
                    dtype=torch.int8,
                    no_split_module_classes=["Qwen2DecoderLayer"]
                )
            else:
                bnb_config = None
                device_map = "auto"
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        print(f"{config.model.model_type} 模型加载完成！")
    
    def predict_one(self, question_raw: str, mapping: Dict[str, str]) -> Dict[str, float]:
        """预测单个问题的分布"""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("模型未初始化，请先调用initialize()方法")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.build_user_prompt(question_raw, mapping)}
        ]
        
        try:
            # 应用聊天模板
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 确定输入设备
            if hasattr(self.model, 'device'):
                device = self.model.device
            else:
                device = f"cuda:0" if torch.cuda.is_available() else "cpu"
                
            inputs = self.tokenizer(text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=config.model.max_tokens,
                    do_sample=False,
                    temperature=config.model.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            reply = self.tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return self.parse_response(reply, mapping)
            
        except Exception as e:
            print(f"预测过程中出错: {e}")
            # 返回默认分布
            if config.model.output_type == "logits":
                default_dist = {k: 0.0 for k in mapping}
            else:
                default_dist = {k: 1.0 for k in mapping}
            
            return self.remove_refused_options(default_dist, mapping)
