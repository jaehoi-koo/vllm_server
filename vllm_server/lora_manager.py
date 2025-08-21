"""
LoRA Management System for vLLM Server
Handles downloading, caching, and switching between different LoRA adapters
"""

import os
import json
import asyncio
import aiofiles
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
import hashlib
import logging

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
import torch

logger = logging.getLogger(__name__)

class LoRAManager:
    def __init__(self, cache_dir: str = "./lora_cache", max_cache_size_gb: float = 50.0):
        """
        Initialize LoRA Manager
        
        Args:
            cache_dir: Directory to cache downloaded LoRAs
            max_cache_size_gb: Maximum cache size in GB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size = max_cache_size_gb * 1024**3  # Convert to bytes
        self.registry_file = self.cache_dir / "lora_registry.json"
        self.registry = self._load_registry()
        
        logger.info(f"LoRA Manager initialized with cache dir: {self.cache_dir}")
    
    def _load_registry(self) -> Dict:
        """Load LoRA registry from disk"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
        
        return {
            "loras": {},
            "download_history": [],
            "cache_info": {
                "total_size": 0,
                "last_cleanup": None
            }
        }
    
    def _save_registry(self):
        """Save LoRA registry to disk"""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _get_cache_size(self) -> int:
        """Calculate total cache size"""
        total_size = 0
        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                try:
                    total_size += os.path.getsize(os.path.join(root, file))
                except OSError:
                    continue
        return total_size
    
    def _cleanup_cache(self):
        """Clean up old cached LoRAs if cache size exceeds limit"""
        current_size = self._get_cache_size()
        
        if current_size <= self.max_cache_size:
            return
        
        logger.info(f"Cache size ({current_size / 1024**3:.2f} GB) exceeds limit, cleaning up...")
        
        # Sort LoRAs by last access time
        loras_by_access = sorted(
            self.registry["loras"].items(),
            key=lambda x: x[1].get("last_accessed", 0)
        )
        
        # Remove oldest LoRAs until under limit
        for lora_id, lora_info in loras_by_access:
            if current_size <= self.max_cache_size * 0.8:  # Leave some headroom
                break
            
            lora_path = Path(lora_info["local_path"])
            if lora_path.exists():
                try:
                    shutil.rmtree(lora_path)
                    removed_size = lora_info.get("size", 0)
                    current_size -= removed_size
                    logger.info(f"Removed cached LoRA: {lora_id}")
                except Exception as e:
                    logger.error(f"Failed to remove {lora_path}: {e}")
            
            # Remove from registry
            del self.registry["loras"][lora_id]
        
        self._save_registry()
    
    async def download_lora(
        self, 
        repo_id: str, 
        lora_name: Optional[str] = None,
        revision: str = "main",
        token: Optional[str] = None
    ) -> str:
        """
        Download LoRA from Hugging Face Hub
        
        Args:
            repo_id: Hugging Face repository ID
            lora_name: Custom name for the LoRA (defaults to repo_id)
            revision: Git revision to download
            token: Hugging Face access token
            
        Returns:
            Local path to downloaded LoRA
        """
        if lora_name is None:
            lora_name = repo_id.replace("/", "_")
        
        # Check if already cached
        lora_id = f"{repo_id}:{revision}"
        if lora_id in self.registry["loras"]:
            lora_info = self.registry["loras"][lora_id]
            local_path = Path(lora_info["local_path"])
            
            if local_path.exists():
                # Update last accessed time
                import time
                lora_info["last_accessed"] = time.time()
                self._save_registry()
                logger.info(f"Using cached LoRA: {lora_name}")
                return str(local_path)
            else:
                # Clean up invalid entry
                del self.registry["loras"][lora_id]
        
        # Download to cache
        local_dir = self.cache_dir / lora_name / revision
        local_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Downloading LoRA: {repo_id} (revision: {revision})")
            
            # Download the entire repository
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=str(local_dir),
                token=token,
                local_dir_use_symlinks=False
            )
            
            # Calculate size
            lora_size = sum(
                os.path.getsize(os.path.join(root, file))
                for root, dirs, files in os.walk(downloaded_path)
                for file in files
            )
            
            # Update registry
            import time
            self.registry["loras"][lora_id] = {
                "repo_id": repo_id,
                "lora_name": lora_name,
                "revision": revision,
                "local_path": str(local_dir),
                "size": lora_size,
                "downloaded_at": time.time(),
                "last_accessed": time.time()
            }
            
            self.registry["download_history"].append({
                "repo_id": repo_id,
                "lora_name": lora_name,
                "timestamp": time.time()
            })
            
            self._save_registry()
            
            # Clean up cache if needed
            self._cleanup_cache()
            
            logger.info(f"Successfully downloaded LoRA: {lora_name} to {local_dir}")
            return str(local_dir)
            
        except HfHubHTTPError as e:
            logger.error(f"Failed to download LoRA {repo_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading LoRA {repo_id}: {e}")
            raise
    
    def list_cached_loras(self) -> List[Dict]:
        """List all cached LoRAs"""
        cached_loras = []
        for lora_id, lora_info in self.registry["loras"].items():
            local_path = Path(lora_info["local_path"])
            if local_path.exists():
                cached_loras.append({
                    "lora_id": lora_id,
                    "lora_name": lora_info["lora_name"],
                    "repo_id": lora_info["repo_id"],
                    "revision": lora_info["revision"],
                    "local_path": str(local_path),
                    "size_mb": lora_info["size"] / 1024**2,
                    "downloaded_at": lora_info["downloaded_at"],
                    "last_accessed": lora_info["last_accessed"]
                })
        
        return sorted(cached_loras, key=lambda x: x["last_accessed"], reverse=True)
    
    def remove_lora(self, lora_id: str) -> bool:
        """Remove a cached LoRA"""
        if lora_id not in self.registry["loras"]:
            return False
        
        lora_info = self.registry["loras"][lora_id]
        local_path = Path(lora_info["local_path"])
        
        try:
            if local_path.exists():
                shutil.rmtree(local_path)
            
            del self.registry["loras"][lora_id]
            self._save_registry()
            
            logger.info(f"Removed LoRA: {lora_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove LoRA {lora_id}: {e}")
            return False
    
    def get_lora_path(self, lora_name: str) -> Optional[str]:
        """Get local path for a LoRA by name"""
        for lora_id, lora_info in self.registry["loras"].items():
            if lora_info["lora_name"] == lora_name:
                local_path = Path(lora_info["local_path"])
                if local_path.exists():
                    return str(local_path)
        return None
    
    def validate_lora(self, lora_path: str) -> bool:
        """Validate that a LoRA directory contains required files"""
        lora_path = Path(lora_path)
        
        # Check for common LoRA files
        required_files = ["adapter_config.json"]
        adapter_files = ["adapter_model.bin", "adapter_model.safetensors"]
        
        # Check required config
        if not (lora_path / "adapter_config.json").exists():
            return False
        
        # Check for at least one adapter file
        if not any((lora_path / f).exists() for f in adapter_files):
            return False
        
        return True
    
    def get_cache_info(self) -> Dict:
        """Get cache information"""
        cache_size = self._get_cache_size()
        return {
            "cache_dir": str(self.cache_dir),
            "total_size_gb": cache_size / 1024**3,
            "max_size_gb": self.max_cache_size / 1024**3,
            "usage_percent": (cache_size / self.max_cache_size) * 100,
            "total_loras": len(self.registry["loras"]),
            "registry": self.registry
        }


# Utility functions for LoRA management
async def download_popular_loras(manager: LoRAManager) -> List[str]:
    """Download a collection of popular LoRAs"""
    popular_loras = [
        ("microsoft/DialoGPT-small-lora", "dialogpt_lora"),
        ("huggingface/CodeBERTa-small-v1-lora", "codeberta_lora"),
        # Add more popular LoRAs as needed
    ]
    
    downloaded_paths = []
    for repo_id, lora_name in popular_loras:
        try:
            path = await manager.download_lora(repo_id, lora_name)
            downloaded_paths.append(path)
            logger.info(f"Downloaded popular LoRA: {lora_name}")
        except Exception as e:
            logger.warning(f"Failed to download {lora_name}: {e}")
    
    return downloaded_paths


def create_lora_config(
    base_model: str,
    lora_paths: List[str],
    lora_names: List[str]
) -> Dict:
    """Create configuration for multiple LoRAs"""
    return {
        "base_model": base_model,
        "loras": [
            {"name": name, "path": path}
            for name, path in zip(lora_names, lora_paths)
        ]
    }