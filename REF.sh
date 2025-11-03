docker run --rm -it --gpus all -p 7009:7009   -v $(pwd):/workspace   -w /workspace --entrypoint /bin/bash  melotts

uvicorn tts_server:app --host 0.0.0.0 --port 7009