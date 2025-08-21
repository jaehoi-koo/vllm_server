"""
Configuration settings for vLLM Server
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseSettings, Field

class VLLMServerConfig(BaseSettings):
    """Configuration for vLLM Server"""
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="VLLM_HOST")
    port: int = Field(default=8001, env="VLLM_PORT")
    workers: int = Field(default=1, env="VLLM_WORKERS")
    
    # Model settings
    model_path: Optional[str] = Field(default=None, env="VLLM_MODEL_PATH")
    tensor_parallel_size: int = Field(default=1, env="VLLM_TENSOR_PARALLEL_SIZE")
    gpu_memory_utilization: float = Field(default=0.8, env="VLLM_GPU_MEMORY_UTILIZATION")
    trust_remote_code: bool = Field(default=True, env="VLLM_TRUST_REMOTE_CODE")
    
    # LoRA settings
    enable_lora: bool = Field(default=True, env="VLLM_ENABLE_LORA")
    max_loras: int = Field(default=4, env="VLLM_MAX_LORAS")
    max_lora_rank: int = Field(default=64, env="VLLM_MAX_LORA_RANK")
    lora_cache_dir: str = Field(default="./lora_cache", env="VLLM_LORA_CACHE_DIR")
    max_cache_size_gb: float = Field(default=50.0, env="VLLM_MAX_CACHE_SIZE_GB")
    
    # Generation defaults
    default_temperature: float = Field(default=0.7, env="VLLM_DEFAULT_TEMPERATURE")
    default_max_tokens: int = Field(default=512, env="VLLM_DEFAULT_MAX_TOKENS")
    default_top_p: float = Field(default=0.9, env="VLLM_DEFAULT_TOP_P")
    default_top_k: int = Field(default=-1, env="VLLM_DEFAULT_TOP_K")
    
    # Hugging Face settings
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")
    hf_cache_dir: Optional[str] = Field(default=None, env="HF_HOME")
    
    # Logging
    log_level: str = Field(default="INFO", env="VLLM_LOG_LEVEL")
    log_dir: str = Field(default="./logs", env="VLLM_LOG_DIR")
    
    # Security
    enable_cors: bool = Field(default=True, env="VLLM_ENABLE_CORS")
    cors_origins: List[str] = Field(default=["*"], env="VLLM_CORS_ORIGINS")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="VLLM_ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="VLLM_METRICS_PORT")
    
    # Performance
    max_concurrent_requests: int = Field(default=100, env="VLLM_MAX_CONCURRENT_REQUESTS")
    request_timeout: float = Field(default=300.0, env="VLLM_REQUEST_TIMEOUT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Popular model configurations
POPULAR_MODELS = {
    "llama2-7b": {
        "repo_id": "meta-llama/Llama-2-7b-hf",
        "description": "Llama 2 7B parameter model",
        "recommended_gpu_memory": 16,
        "tensor_parallel_size": 1
    },
    "llama2-13b": {
        "repo_id": "meta-llama/Llama-2-13b-hf", 
        "description": "Llama 2 13B parameter model",
        "recommended_gpu_memory": 32,
        "tensor_parallel_size": 2
    },
    "mistral-7b": {
        "repo_id": "mistralai/Mistral-7B-v0.1",
        "description": "Mistral 7B parameter model",
        "recommended_gpu_memory": 16,
        "tensor_parallel_size": 1
    },
    "codellama-7b": {
        "repo_id": "codellama/CodeLlama-7b-hf",
        "description": "Code Llama 7B for code generation",
        "recommended_gpu_memory": 16,
        "tensor_parallel_size": 1
    },
    "yi-6b": {
        "repo_id": "01-ai/Yi-6B",
        "description": "Yi 6B parameter model",
        "recommended_gpu_memory": 12,
        "tensor_parallel_size": 1
    }
}

# Popular LoRA configurations
POPULAR_LORAS = {
    "alpaca-7b": {
        "repo_id": "tloen/alpaca-lora-7b",
        "description": "Alpaca fine-tuned LoRA for instruction following",
        "base_model": "decapoda-research/llama-7b-hf"
    },
    "vicuna-7b": {
        "repo_id": "eachadea/vicuna-7b-1.1-lora",
        "description": "Vicuna LoRA for conversation",
        "base_model": "decapoda-research/llama-7b-hf"
    },
    "wizardlm-7b": {
        "repo_id": "microsoft/WizardLM-7B-V1.0-lora",
        "description": "WizardLM LoRA for complex instructions",
        "base_model": "decapoda-research/llama-7b-hf"
    }
}

# Default sampling parameters for different use cases
SAMPLING_PRESETS = {
    "creative": {
        "temperature": 0.9,
        "top_p": 0.9,
        "top_k": 40,
        "description": "High creativity for creative writing"
    },
    "balanced": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": -1,
        "description": "Balanced creativity and coherence"
    },
    "precise": {
        "temperature": 0.3,
        "top_p": 0.85,
        "top_k": -1,
        "description": "Low temperature for factual responses"
    },
    "deterministic": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "description": "Deterministic responses"
    },
    "code": {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": -1,
        "description": "Optimized for code generation"
    }
}

def get_config() -> VLLMServerConfig:
    """Get configuration instance"""
    return VLLMServerConfig()

def get_model_info(model_name: str) -> Optional[Dict]:
    """Get information about a popular model"""
    return POPULAR_MODELS.get(model_name)

def get_lora_info(lora_name: str) -> Optional[Dict]:
    """Get information about a popular LoRA"""
    return POPULAR_LORAS.get(lora_name)

def get_sampling_preset(preset_name: str) -> Optional[Dict]:
    """Get sampling parameters for a preset"""
    return SAMPLING_PRESETS.get(preset_name)

def list_popular_models() -> List[str]:
    """List all popular model names"""
    return list(POPULAR_MODELS.keys())

def list_popular_loras() -> List[str]:
    """List all popular LoRA names"""
    return list(POPULAR_LORAS.keys())

def list_sampling_presets() -> List[str]:
    """List all sampling preset names"""
    return list(SAMPLING_PRESETS.keys())