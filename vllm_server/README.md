# vLLM Server with LoRA 관리 시스템

동적 LoRA (Low-Rank Adaptation) 로딩 및 전환 기능을 제공하는 프로덕션 레벨의 vLLM 서버입니다. 서버 재시작 없이 여러 LoRA 어댑터를 동적으로 교체하면서 대용량 언어 모델을 효율적으로 서빙할 수 있습니다.

## 주요 기능

- **동적 LoRA 관리**: 실시간으로 LoRA 어댑터 로딩 및 언로딩
- **Hugging Face 통합**: 모델과 LoRA의 자동 다운로드 및 캐싱
- **고성능**: vLLM 기반의 빠른 추론
- **RESTful API**: 모든 작업을 위한 사용하기 쉬운 HTTP API
- **Docker 지원**: GPU 지원을 포함한 컨테이너화된 배포
- **모니터링**: 내장된 메트릭 및 헬스 체크
- **캐시 관리**: 크기 제한 및 자동 정리를 포함한 지능적 캐싱

## 빠른 시작

### Docker 사용 (권장)

1. **클론 및 설정**:
   ```bash
   cd vllm_server
   cp .env.example .env
   # .env 파일을 수정하여 설정 구성
   ```

2. **서버 시작**:
   ```bash
   docker-compose up --build
   ```

### 로컬 설치

1. **의존성 설치**:
   ```bash
   pip install -r requirements.txt
   ```

2. **서버 시작**:
   ```bash
   # 간단한 시작
   python main.py --model microsoft/DialoGPT-medium

   # 또는 시작 스크립트 사용
   chmod +x start.sh
   ./start.sh --model microsoft/DialoGPT-medium
   ```

## API 사용법

### 서버 초기화
```bash
curl -X POST "http://localhost:8001/initialize" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "microsoft/DialoGPT-medium",
    "max_loras": 4,
    "gpu_memory_utilization": 0.8
  }'
```

### LoRA 로딩
```bash
curl -X POST "http://localhost:8001/load_lora" \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "my_lora",
    "lora_path": "/path/to/lora"
  }'
```

### 텍스트 생성
```bash
# 베이스 모델로 생성
curl -X POST "http://localhost:8001/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "안녕하세요, 어떻게 지내세요?",
    "max_tokens": 50
  }'

# LoRA로 생성
curl -X POST "http://localhost:8001/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "안녕하세요, 어떻게 지내세요?",
    "lora_name": "my_lora",
    "max_tokens": 50
  }'
```

### 상태 확인
```bash
curl "http://localhost:8001/status"
```

### LoRA 목록 조회
```bash
curl "http://localhost:8001/loras"
```

## Python 클라이언트

```python
import asyncio
from client import VLLMClient, LoRAManagerClient

async def main():
    client = VLLMClient("http://localhost:8001")
    lora_client = LoRAManagerClient(client)
    
    # 서버 초기화
    await client.initialize_server("microsoft/DialoGPT-medium")
    
    # Hugging Face에서 LoRA 다운로드 및 로딩
    await lora_client.download_and_load_lora(
        repo_id="microsoft/DialoGPT-small-lora",
        lora_name="dialog_lora"
    )
    
    # LoRA로 텍스트 생성
    result = await client.generate(
        prompt="안녕하세요!",
        lora_name="dialog_lora",
        max_tokens=50
    )
    print(result['generated_text'])

asyncio.run(main())
```

## 설정

### 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|---------|
| `VLLM_HOST` | 서버 호스트 | 0.0.0.0 |
| `VLLM_PORT` | 서버 포트 | 8001 |
| `VLLM_MODEL_PATH` | 베이스 모델 경로 | None |
| `VLLM_MAX_LORAS` | 최대 LoRA 수 | 4 |
| `VLLM_GPU_MEMORY_UTILIZATION` | GPU 메모리 사용률 | 0.8 |
| `HF_TOKEN` | Hugging Face 토큰 | None |

### 지원 모델

이 서버와 잘 작동하는 인기 모델들:

- **Llama 2**: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`
- **Mistral**: `mistralai/Mistral-7B-v0.1`
- **Code Llama**: `codellama/CodeLlama-7b-hf`
- **Yi**: `01-ai/Yi-6B`
- **DialoGPT**: `microsoft/DialoGPT-medium`

### LoRA 지원

서버가 지원하는 표준 LoRA 형식:
- Alpaca LoRA
- Vicuna LoRA  
- 커스텀 파인튜닝 LoRA
- transformers/PEFT와 호환되는 모든 LoRA

## 고급 사용법

### 시작 스크립트 옵션

```bash
# 사용 가능한 모델과 LoRA 목록 확인
./start.sh --list-models
./start.sh --list-loras

# 인기 모델 다운로드 후 시작
./start.sh --download-model mistral-7b --model mistral-7b

# 멀티 GPU 설정
./start.sh --model llama2-13b --tensor-parallel 2 --gpu-memory 0.9

# 자동 리로드가 포함된 개발 모드
./start.sh --model microsoft/DialoGPT-medium --dev
```

### Docker Compose 서비스

포함된 `docker-compose.yml`이 제공하는 서비스:
- **vLLM Server**: 메인 추론 서버
- **Redis**: 캐싱 레이어 (선택사항)
- **Prometheus**: 메트릭 수집 (선택사항)

### LoRA 캐시 관리

서버가 자동으로 관리하는 LoRA 캐시:
- Hugging Face Hub에서 다운로드
- 재사용을 위한 로컬 캐싱
- 캐시 제한 초과시 자동 정리
- LoRA 형식 검증

## 모니터링

### 헬스 체크
```bash
curl "http://localhost:8001/"
```

### 메트릭 (활성화된 경우)
- 서버 상태 및 업타임
- 모델 및 LoRA 정보
- 생성 통계
- GPU 메모리 사용량

## 문제 해결

### 일반적인 문제들

1. **GPU 메모리 오류**: `gpu_memory_utilization` 감소 또는 더 작은 모델 사용
2. **LoRA 로딩 실패**: LoRA 형식 및 경로 확인
3. **모델 다운로드 실패**: Hugging Face 토큰 및 모델 접근 권한 확인
4. **CUDA 메모리 부족**: `max_loras` 또는 `tensor_parallel_size` 감소

### 디버그 모드

디버그 로깅 활성화:
```bash
export VLLM_LOG_LEVEL=DEBUG
python main.py --model your_model
```

## 성능 최적화 팁

1. **GPU 메모리**: 최적 성능을 위해 GPU 메모리의 ~20% 여유 공간 확보
2. **텐서 병렬화**: 대용량 모델(13B+)에는 멀티 GPU 사용
3. **LoRA 제한**: GPU 메모리에 따라 `max_loras`를 적절히 설정 (4-8개)
4. **캐싱**: 모델 및 LoRA 캐시용 SSD 스토리지 사용

## 기여하기

1. 저장소 포크
2. 기능 브랜치 생성
3. 변경사항 작성
4. 해당되는 경우 테스트 추가
5. 풀 리퀘스트 제출

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다 - 자세한 내용은 LICENSE 파일을 참조하세요.

## 지원

이슈 및 질문에 대해:
1. 문제 해결 섹션 확인
2. 기존 이슈 검색
3. 상세 정보와 함께 새 이슈 생성

## 감사의 말

- [vLLM](https://github.com/vllm-project/vllm) - 추론 엔진 제공
- [Hugging Face](https://huggingface.co/) - 모델 호스팅 및 transformers 라이브러리
- [FastAPI](https://fastapi.tiangolo.com/) - 웹 프레임워크 제공