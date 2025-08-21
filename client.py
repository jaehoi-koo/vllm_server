"""
vLLM Server Client
Python client for interacting with the vLLM server with LoRA support
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any, AsyncGenerator
import logging

logger = logging.getLogger(__name__)

class VLLMClient:
    """Client for vLLM Server with LoRA Management"""
    
    def __init__(self, base_url: str = "http://localhost:8001", timeout: float = 300.0):
        """
        Initialize vLLM client
        
        Args:
            base_url: Base URL of the vLLM server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        
    async def initialize_server(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_loras: int = 4
    ) -> Dict:
        """Initialize the vLLM server with a base model"""
        payload = {
            "model_path": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_loras": max_loras
        }
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(f"{self.base_url}/initialize", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Server initialization failed: {error_text}")
    
    async def load_lora(self, lora_name: str, lora_path: str) -> Dict:
        """Load a LoRA adapter"""
        payload = {
            "lora_name": lora_name,
            "lora_path": lora_path
        }
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(f"{self.base_url}/load_lora", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"LoRA loading failed: {error_text}")
    
    async def unload_lora(self, lora_name: str) -> Dict:
        """Unload a LoRA adapter"""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.delete(f"{self.base_url}/unload_lora/{lora_name}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"LoRA unloading failed: {error_text}")
    
    async def generate(
        self,
        prompt: str,
        lora_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        top_k: int = -1,
        stop: Optional[List[str]] = None
    ) -> Dict:
        """Generate text with optional LoRA"""
        payload = {
            "prompt": prompt,
            "lora_name": lora_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "stop": stop
        }
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(f"{self.base_url}/generate", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Generation failed: {error_text}")
    
    async def get_status(self) -> Dict:
        """Get server status"""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(f"{self.base_url}/status") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Status check failed: {error_text}")
    
    async def list_loras(self) -> Dict:
        """List all loaded LoRAs"""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(f"{self.base_url}/loras") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"LoRA listing failed: {error_text}")
    
    async def health_check(self) -> bool:
        """Check if server is healthy"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{self.base_url}/") as response:
                    return response.status == 200
        except:
            return False


class LoRAManagerClient:
    """Client for LoRA Manager functionality"""
    
    def __init__(self, vllm_client: VLLMClient):
        self.client = vllm_client
        
    async def download_and_load_lora(
        self,
        repo_id: str,
        lora_name: Optional[str] = None,
        revision: str = "main",
        token: Optional[str] = None
    ) -> str:
        """Download LoRA from Hugging Face and load it"""
        from .lora_manager import LoRAManager
        
        # Initialize LoRA manager
        manager = LoRAManager()
        
        # Download LoRA
        local_path = await manager.download_lora(
            repo_id=repo_id,
            lora_name=lora_name,
            revision=revision,
            token=token
        )
        
        # Load into vLLM server
        final_name = lora_name or repo_id.replace("/", "_")
        await self.client.load_lora(final_name, local_path)
        
        return local_path
    
    async def switch_lora(self, lora_name: str) -> Dict:
        """Switch to a different LoRA for generation"""
        # Just return the current status - switching happens per request
        return await self.client.get_status()
    
    async def benchmark_loras(
        self,
        prompt: str,
        lora_names: List[str],
        num_iterations: int = 5
    ) -> Dict[str, Dict]:
        """Benchmark different LoRAs with the same prompt"""
        results = {}
        
        for lora_name in lora_names:
            lora_results = []
            
            for i in range(num_iterations):
                try:
                    import time
                    start_time = time.time()
                    
                    result = await self.client.generate(
                        prompt=prompt,
                        lora_name=lora_name
                    )
                    
                    end_time = time.time()
                    
                    lora_results.append({
                        "iteration": i + 1,
                        "response_time": end_time - start_time,
                        "generated_text": result.get("generated_text", ""),
                        "success": True
                    })
                    
                except Exception as e:
                    lora_results.append({
                        "iteration": i + 1,
                        "error": str(e),
                        "success": False
                    })
            
            # Calculate statistics
            successful_runs = [r for r in lora_results if r["success"]]
            if successful_runs:
                response_times = [r["response_time"] for r in successful_runs]
                results[lora_name] = {
                    "runs": lora_results,
                    "success_rate": len(successful_runs) / len(lora_results),
                    "avg_response_time": sum(response_times) / len(response_times),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times)
                }
            else:
                results[lora_name] = {
                    "runs": lora_results,
                    "success_rate": 0,
                    "avg_response_time": None,
                    "min_response_time": None,
                    "max_response_time": None
                }
        
        return results


# Example usage and testing functions
async def example_usage():
    """Example usage of the vLLM client"""
    client = VLLMClient("http://localhost:8001")
    lora_client = LoRAManagerClient(client)
    
    try:
        # Check if server is healthy
        if not await client.health_check():
            print("Server is not running or unhealthy")
            return
        
        # Initialize server (if needed)
        try:
            status = await client.get_status()
            if status["status"] != "running":
                print("Initializing server...")
                await client.initialize_server(
                    model_path="microsoft/DialoGPT-medium",
                    max_loras=4
                )
        except:
            print("Server already initialized or initialization failed")
        
        # Download and load a LoRA
        print("Downloading and loading LoRA...")
        await lora_client.download_and_load_lora(
            repo_id="microsoft/DialoGPT-small-lora",
            lora_name="dialog_lora"
        )
        
        # List loaded LoRAs
        loras = await client.list_loras()
        print(f"Loaded LoRAs: {loras}")
        
        # Generate with base model
        print("\nGenerating with base model...")
        result = await client.generate(
            prompt="Hello, how are you?",
            max_tokens=50
        )
        print(f"Base model: {result['generated_text']}")
        
        # Generate with LoRA
        print("\nGenerating with LoRA...")
        result = await client.generate(
            prompt="Hello, how are you?",
            lora_name="dialog_lora",
            max_tokens=50
        )
        print(f"LoRA model: {result['generated_text']}")
        
        # Get final status
        status = await client.get_status()
        print(f"\nFinal status: {status}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(example_usage())