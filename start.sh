#!/bin/bash

# vLLM Server Startup Script
# This script provides various options for starting the vLLM server

set -e

# Default configuration
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8001"
DEFAULT_MODEL=""
DEFAULT_LORA_DIR="./lora_cache"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE} vLLM Server with LoRA Support${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to show help
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -m, --model MODEL       Base model path or Hugging Face repo ID
    -p, --port PORT         Server port (default: 8001)
    --host HOST             Host to bind to (default: 0.0.0.0)
    --gpu-memory RATIO      GPU memory utilization (default: 0.8)
    --tensor-parallel SIZE  Number of GPUs for tensor parallelism (default: 1)
    --max-loras COUNT       Maximum number of LoRAs (default: 4)
    --lora-dir DIR          LoRA cache directory (default: ./lora_cache)
    --dev                   Start in development mode with auto-reload
    --docker                Start using Docker Compose
    --download-model MODEL  Download a popular model before starting
    --list-models           List available popular models
    --list-loras            List available popular LoRAs

Examples:
    $0 --model microsoft/DialoGPT-medium
    $0 --model llama2-7b --gpu-memory 0.9 --tensor-parallel 2
    $0 --download-model mistral-7b --model mistral-7b
    $0 --docker
    $0 --dev

Environment Variables:
    VLLM_MODEL_PATH         Base model path
    VLLM_PORT              Server port
    VLLM_HOST              Host to bind to
    HF_TOKEN               Hugging Face access token
    CUDA_VISIBLE_DEVICES   GPU devices to use

EOF
}

# Function to list popular models
list_models() {
    print_status "Popular models available:"
    python3 -c "
from config import POPULAR_MODELS
for name, info in POPULAR_MODELS.items():
    print(f'  {name}: {info[\"description\"]} (GPU: {info[\"recommended_gpu_memory\"]}GB)')
"
}

# Function to list popular LoRAs
list_loras() {
    print_status "Popular LoRAs available:"
    python3 -c "
from config import POPULAR_LORAS
for name, info in POPULAR_LORAS.items():
    print(f'  {name}: {info[\"description\"]}')
"
}

# Function to download model
download_model() {
    local model_name=$1
    print_status "Downloading model: $model_name"
    
    python3 -c "
import sys
sys.path.append('.')
from config import POPULAR_MODELS
from huggingface_hub import snapshot_download
import os

model_name = '$model_name'
if model_name in POPULAR_MODELS:
    repo_id = POPULAR_MODELS[model_name]['repo_id']
    local_dir = f'./models/{model_name}'
    os.makedirs(local_dir, exist_ok=True)
    
    print(f'Downloading {repo_id} to {local_dir}...')
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print(f'Model downloaded successfully to {local_dir}')
else:
    print(f'Model {model_name} not found in popular models list')
    sys.exit(1)
"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi not found. GPU may not be available."
    else
        print_status "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    fi
    
    # Check required Python packages
    python3 -c "
import sys
required = ['vllm', 'fastapi', 'uvicorn', 'transformers', 'torch']
missing = []
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print('Missing required packages:', ', '.join(missing))
    print('Please install with: pip install -r requirements.txt')
    sys.exit(1)
"
    
    print_status "Prerequisites check passed"
}

# Function to start server
start_server() {
    local model_path=$1
    local host=$2
    local port=$3
    local extra_args=$4
    local dev_mode=$5
    
    if [ -z "$model_path" ]; then
        print_error "Model path is required. Use --model option or set VLLM_MODEL_PATH"
        exit 1
    fi
    
    print_status "Starting vLLM server..."
    print_status "Model: $model_path"
    print_status "Host: $host"
    print_status "Port: $port"
    
    # Create necessary directories
    mkdir -p logs
    mkdir -p "$DEFAULT_LORA_DIR"
    
    if [ "$dev_mode" = "true" ]; then
        print_status "Starting in development mode with auto-reload"
        uvicorn main:app --host "$host" --port "$port" --reload $extra_args
    else
        python3 main.py --host "$host" --port "$port" --model "$model_path" $extra_args
    fi
}

# Function to start with Docker
start_docker() {
    print_status "Starting vLLM server with Docker Compose..."
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose is not installed"
        exit 1
    fi
    
    # Check if nvidia-docker is available
    if ! docker info | grep -q "nvidia"; then
        print_warning "NVIDIA Docker runtime not detected. GPU support may not work."
    fi
    
    docker-compose up --build
}

# Parse command line arguments
MODEL_PATH="${VLLM_MODEL_PATH:-$DEFAULT_MODEL}"
HOST="${VLLM_HOST:-$DEFAULT_HOST}"
PORT="${VLLM_PORT:-$DEFAULT_PORT}"
EXTRA_ARGS=""
DEV_MODE="false"
DOCKER_MODE="false"
DOWNLOAD_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --gpu-memory)
            EXTRA_ARGS="$EXTRA_ARGS --gpu-memory-utilization $2"
            shift 2
            ;;
        --tensor-parallel)
            EXTRA_ARGS="$EXTRA_ARGS --tensor-parallel-size $2"
            shift 2
            ;;
        --max-loras)
            EXTRA_ARGS="$EXTRA_ARGS --max-loras $2"
            shift 2
            ;;
        --lora-dir)
            DEFAULT_LORA_DIR="$2"
            shift 2
            ;;
        --dev)
            DEV_MODE="true"
            shift
            ;;
        --docker)
            DOCKER_MODE="true"
            shift
            ;;
        --download-model)
            DOWNLOAD_MODEL="$2"
            shift 2
            ;;
        --list-models)
            list_models
            exit 0
            ;;
        --list-loras)
            list_loras
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
print_header

# Download model if requested
if [ -n "$DOWNLOAD_MODEL" ]; then
    download_model "$DOWNLOAD_MODEL"
    if [ -z "$MODEL_PATH" ] || [ "$MODEL_PATH" = "$DEFAULT_MODEL" ]; then
        MODEL_PATH="./models/$DOWNLOAD_MODEL"
    fi
fi

if [ "$DOCKER_MODE" = "true" ]; then
    start_docker
else
    check_prerequisites
    start_server "$MODEL_PATH" "$HOST" "$PORT" "$EXTRA_ARGS" "$DEV_MODE"
fi