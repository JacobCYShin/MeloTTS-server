# TTS Server Only

한국어 TTS (Text-to-Speech) 서버입니다. MeloTTS 기반으로 구축되었으며, 여러 음성 모델을 지원합니다.

## 🚀 주요 기능

- 한국어 텍스트를 자연스러운 음성으로 변환
- 여러 음성 모델 지원 (HTR, KRW, PDS Announcer, PDS Natural)
- VAD (Voice Activity Detection) 기반 무음 제거
- 실시간 음성 합성 API 제공
- Docker 컨테이너 지원

## 📋 요구사항

- NVIDIA GPU (CUDA 지원)
- Docker
- 최소 8GB RAM
- 최소 10GB 디스크 공간

## 🛠️ 설치 및 설정

### 1. 저장소 클론

```bash
git clone <repository-url>
cd TTS_server_only
```

### 2. 모델 파일 준비

**중요**: 다음 모델 파일들을 프로젝트 루트에 배치해야 합니다:

#### BERT 모델
```
bert-kor-base/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
└── vocab.txt
```

#### TTS 모델들
```
melotts_models/
├── htr_adapt_20250414_v2/
│   ├── config.json
│   └── G_40000.pth
├── krw_model_v1/
│   ├── config.json
│   └── G_707000.pth
├── pds_a_revised_20250818/
│   ├── config.json
│   └── G_258000.pth
└── pds_ms_revised_20250818/
    ├── config.json
    └── G_179000.pth
```

### 3. Docker 이미지 빌드

```bash
docker build -t melotts .
```

### 4. 서버 실행

```bash
docker run --rm -it --gpus all -p 7860:7860 \
  -v $(pwd):/workspace \
  -w /workspace \
  --entrypoint /bin/bash \
  melotts
```

컨테이너 내부에서:
```bash
uvicorn tts_server:app --host 0.0.0.0 --port 7860
```

## 📖 API 사용법

### TTS 요청

**Endpoint**: `POST /tts`

**Request Body**:
```json
{
    "text": "안녕하세요, TTS 서버입니다.",
    "sr": 16000,
    "model": "pds_announcer",
    "pre_post_silence_sec": 0.2,
    "intermittent_silence_sec": 0.15,
    "speed": 1.0
}
```

**Parameters**:
- `text` (string): 변환할 텍스트
- `sr` (int): 샘플링 레이트 (16000 또는 8000)
- `model` (string): 사용할 모델
  - `htr`: HTR 모델
  - `krw`: KRW 모델
  - `pds_announcer`: PDS 아나운서 모델
  - `pds_natural`: PDS 자연스러운 모델
- `pre_post_silence_sec` (float): 앞뒤 무음 시간 (초)
- `intermittent_silence_sec` (float): 중간 무음 시간 (초)
- `speed` (float): 재생 속도 (0.8 ~ 1.2)

**Response**: WAV 오디오 파일

### 사용 예시

```bash
curl -X POST "http://localhost:7860/tts" \
  -H "Content-Type: application/json" \
  -d @test2.json \
  -o output.wav
```

## 🔧 설정 파일

### test2.json 예시
```json
{
    "text": "안녕하세요, 하나증권 쑈미더리포트의 편다송송입니다.",
    "sr": 16000,
    "model": "pds_announcer",
    "pre_post_silence_sec": 0.2,
    "intermittent_silence_sec": 0.15,
    "speed": 1.0
}
```

## 📁 프로젝트 구조

```
TTS_server_only/
├── melo/                    # MeloTTS 핵심 모듈
├── newg2p/                  # 한국어 G2P (Grapheme-to-Phoneme) 모듈
├── bert-kor-base/           # BERT 한국어 모델 (별도 다운로드 필요)
├── melotts_models/          # TTS 모델들 (별도 다운로드 필요)
├── tts_server.py           # FastAPI 서버
├── requirements.txt        # Python 의존성
├── Dockerfile             # Docker 설정
├── test2.json            # 테스트 요청 예시
└── README.md             # 이 파일
```

## 🐛 문제 해결

### 일반적인 오류

1. **CUDA 오류**: GPU 드라이버와 CUDA 버전 확인
2. **모델 로딩 오류**: 모델 파일 경로 확인
3. **메모리 부족**: GPU 메모리 확인

### 로그 확인

```bash
# 서버 로그 확인
tail -f server.log
```

## 📝 라이선스

이 프로젝트는 MeloTTS를 기반으로 합니다. 라이선스 정보는 각 모델의 원본 저장소를 참조하세요.

## 🤝 기여

버그 리포트나 기능 요청은 이슈를 통해 제출해 주세요.

## 📞 지원

문제가 발생하면 로그와 함께 이슈를 생성해 주세요.
