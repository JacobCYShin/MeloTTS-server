FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# pip 최신화
RUN python3 -m pip install --upgrade pip

WORKDIR /app

# 빌드 시 필요한 파일들만 복사
COPY requirements.txt /app/
COPY melo/ /app/melo/

# 런타임에 마운트될 파일들은 복사하지 않음
# - newg2p/ (런타임에 마운트)
# - bert-kor-base/ (런타임에 마운트)
# - melotts_models/ (런타임에 마운트)
# - tts_server.py (런타임에 마운트)
# - test.py (런타임에 마운트)

RUN apt-get update && apt-get install -y \
    build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -e .
RUN python3 -m unidic download
RUN python3 melo/init_downloads.py

RUN python3 test.py
