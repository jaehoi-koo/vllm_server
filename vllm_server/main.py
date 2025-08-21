"""
vLLM Server with LoRA Management
Supports dynamic LoRA loading and switching for efficient inference
"""

import asyncio
import logging
import os
import json
from typing import Dict, List, Optional, Any
from contextual import contextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="vLLM Server with LoRA Management", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VLLMServer:
    def __init__(self):
        self.engine: Optional[AsyncLLMEngine] = None
        self.loaded_loras: Dict[str, str] = {}  # lora_name -> lora_path
        self.current_lora: Optional[str] = None
        self.base_model_path: str = ""
        self.max_loras: int = 4  # Maximum number of LoRAs to keep loaded
        
    async def initialize(self, model_path: str, **kwargs):
        """Initialize the vLLM engine with base model"""
        try:
            self.base_model_path = model_path
            
            # Engine arguments
            engine_args = AsyncEngineArgs(
                model=model_path,
                tensor_parallel_size=1,
                dtype="auto",
                trust_remote_code=True,
                enable_lora=True,  # Enable LoRA support
                max_loras=self.max_loras,
                max_lora_rank=64,
                gpu_memory_utilization=0.8,
                **kwargs
            )
            
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info(f"vLLM engine initialized with model: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise
    
    async def load_lora(self, lora_name: str, lora_path: str) -> bool:
        """Load a LoRA adapter"""
        try:
            if not os.path.exists(lora_path):
                logger.error(f"LoRA path does not exist: {lora_path}")
                return False
            
            # Check if LoRA is already loaded
            if lora_name in self.loaded_loras:
                logger.info(f"LoRA {lora_name} already loaded")
                return True
            
            # Remove old LoRA if max limit reached
            if len(self.loaded_loras) >= self.max_loras:
                oldest_lora = list(self.loaded_loras.keys())[0]
                await self.unload_lora(oldest_lora)
            
            self.loaded_loras[lora_name] = lora_path
            logger.info(f"LoRA {lora_name} loaded from {lora_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LoRA {lora_name}: {e}")
            return False
    
    async def unload_lora(self, lora_name: str) -> bool:
        """Unload a LoRA adapter"""
        try:
            if lora_name in self.loaded_loras:
                del self.loaded_loras[lora_name]
                if self.current_lora == lora_name:
                    self.current_lora = None
                logger.info(f"LoRA {lora_name} unloaded")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to unload LoRA {lora_name}: {e}")
            return False
    
    async def generate(
        self, 
        prompt: str, 
        lora_name: Optional[str] = None,
        sampling_params: Optional[SamplingParams] = None
    ) -> str:
        """Generate text with optional LoRA"""
        if not self.engine:
            raise RuntimeError("Engine not initialized")
        
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=512,
                top_p=0.9
            )
        
        # Create LoRA request if specified
        lora_request = None
        if lora_name and lora_name in self.loaded_loras:
            lora_request = LoRARequest(
                lora_name=lora_name,
                lora_int_id=hash(lora_name) % 1000000,  # Generate unique ID
                lora_local_path=self.loaded_loras[lora_name]
            )
            self.current_lora = lora_name
        
        try:
            # Generate
            request_id = f"req_{len(self.loaded_loras)}_{hash(prompt) % 10000}"
            results = self.engine.generate(
                prompt,
                sampling_params,
                request_id=request_id,
                lora_request=lora_request
            )
            
            # Get the result
            async for request_output in results:
                if request_output.finished:
                    return request_output.outputs[0].text
                    
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

# Global server instance
vllm_server = VLLMServer()

# Pydantic models for API
class InitializeRequest(BaseModel):
    model_path: str = Field(..., description="Path to the base model")
    tensor_parallel_size: int = Field(1, description="Number of GPUs to use")
    gpu_memory_utilization: float = Field(0.8, description="GPU memory utilization ratio")
    max_loras: int = Field(4, description="Maximum number of LoRAs to keep loaded")

class LoadLoRARequest(BaseModel):
    lora_name: str = Field(..., description="Name identifier for the LoRA")
    lora_path: str = Field(..., description="Path to the LoRA adapter")

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    lora_name: Optional[str] = Field(None, description="LoRA to use for generation")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=1, le=4096)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(-1, ge=-1)
    stop: Optional[List[str]] = Field(None, description="Stop sequences")

class ServerStatus(BaseModel):
    status: str
    base_model: str
    loaded_loras: List[str]
    current_lora: Optional[str]
    gpu_memory_usage: Optional[float] = None

# API Endpoints
@app.get("/")
async def root():
    return {"message": "vLLM Server with LoRA Management"}

@app.post("/initialize")
async def initialize_engine(request: InitializeRequest):
    """Initialize the vLLM engine with base model"""
    try:
        await vllm_server.initialize(
            model_path=request.model_path,
            tensor_parallel_size=request.tensor_parallel_size,
            gpu_memory_utilization=request.gpu_memory_utilization
        )
        vllm_server.max_loras = request.max_loras
        return {"status": "success", "message": "Engine initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_lora")
async def load_lora_endpoint(request: LoadLoRARequest):
    """Load a LoRA adapter"""
    success = await vllm_server.load_lora(request.lora_name, request.lora_path)
    if success:
        return {"status": "success", "message": f"LoRA {request.lora_name} loaded successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to load LoRA {request.lora_name}")

@app.delete("/unload_lora/{lora_name}")
async def unload_lora_endpoint(lora_name: str):
    """Unload a LoRA adapter"""
    success = await vllm_server.unload_lora(lora_name)
    if success:
        return {"status": "success", "message": f"LoRA {lora_name} unloaded successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"LoRA {lora_name} not found")

@app.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    """Generate text with optional LoRA"""
    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k > 0 else None,
            stop=request.stop
        )
        
        result = await vllm_server.generate(
            prompt=request.prompt,
            lora_name=request.lora_name,
            sampling_params=sampling_params
        )
        
        return {
            "status": "success",
            "generated_text": result,
            "used_lora": request.lora_name,
            "prompt": request.prompt
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=ServerStatus)
async def get_status():
    """Get server status"""
    gpu_memory_usage = None
    if torch.cuda.is_available():
        gpu_memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
    
    return ServerStatus(
        status="running" if vllm_server.engine else "not_initialized",
        base_model=vllm_server.base_model_path,
        loaded_loras=list(vllm_server.loaded_loras.keys()),
        current_lora=vllm_server.current_lora,
        gpu_memory_usage=gpu_memory_usage
    )

@app.get("/loras")
async def list_loras():
    """List all loaded LoRAs"""
    return {
        "loaded_loras": vllm_server.loaded_loras,
        "current_lora": vllm_server.current_lora,
        "max_loras": vllm_server.max_loras
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Server with LoRA Management")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--model", help="Base model path (can be set via API)")
    
    args = parser.parse_args()
    
    # Initialize with model if provided
    if args.model:
        @app.on_event("startup")
        async def startup_event():
            await vllm_server.initialize(args.model)
    
    uvicorn.run(app, host=args.host, port=args.port)