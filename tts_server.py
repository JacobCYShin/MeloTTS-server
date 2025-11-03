# 2025.02.17 version
# written by ANDY

import os
# print('before: ', os.environ.get("CUDA_VISIBLE_DEVICES"))

import torch
print(torch.cuda.device_count())  # Should return num_gpus
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

import librosa
import numpy as np
import soundfile as sf
import io

from pathlib import Path
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from fastapi.responses import JSONResponse, StreamingResponse
from dataclasses import dataclass
from time import time
from silero_vad import load_silero_vad, get_speech_timestamps

from melo.api import TTS
from newg2p.trans import sentranslit as trans

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    sr: Literal[16000, 8000]
    model: str = Field(...)
    pre_post_silence_sec: float = Field(..., ge=0)
    intermittent_silence_sec: float = Field(..., ge=0)
    speed: float = Field(1.0, ge=0.8, le=1.2)

    # wav_path 제거

@dataclass
class TTSConfig:
    device: str = 'cuda'
    model_name: str = ''
    steps: str = ''
    speaker_id: str = ''

app = FastAPI()
models_path = './melotts_models'
orig_sr = 44100
tts_models = {}
vad_model = load_silero_vad()

def initialize_tts(config):
    model = TTS(
        language='KR',
        device=config.device,
        use_hf=False,
        config_path=f'{models_path}/{config.model_name}/config.json',
        ckpt_path=f'{models_path}/{config.model_name}/G_{config.steps}.pth'
    )
    speaker_ids = model.hps.data.spk2id

    text_line = f"{config.speaker_id} TTS 모델 테스트 warmup"
    audio = model.tts_to_file(text_line, speaker_ids[config.speaker_id], output_path=None, speed=1.0)
    duration = librosa.get_duration(y=audio, sr=orig_sr)
    resampled_audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)

    # do vad
    resampled_audio = remove_silence(resampled_audio, 16000, pre_post_silence_sec=0.2, intermittent_silence_sec=0.2)


    return model, speaker_ids

def remove_silence(input_wav, sampling_rate, pre_post_silence_sec, intermittent_silence_sec):
    """
    Replace silence with a fixed duration.
    
    Args:
        input_wav (numpy.ndarray): Input waveform (1D NumPy array).
        sampling_rate (int): Sample rate of the audio.
        pre_post_silence_sec (float): Amount of silence to add (in seconds) at the start and end.
        intermittent_silence_sec (float): Duration of intermittent silence to add (in seconds).

    Returns:
        numpy.ndarray: Processed audio without unwanted silence.
    """
    # Run VAD
    speech_timestamps = get_speech_timestamps(
        input_wav, vad_model, sampling_rate=sampling_rate, return_seconds=False
    )

    if not speech_timestamps:
        print("No speech detected.")
        return input_wav  # Return original audio if no speech is found

    processed_audio = []
    silence_samples = int(intermittent_silence_sec * sampling_rate)
    pre_post_silence_samples = int(pre_post_silence_sec * sampling_rate)

    print(speech_timestamps)
    print('silence :', silence_samples)

    # Add silence at the beginning
    processed_audio.append(np.zeros(pre_post_silence_samples, dtype=np.float32))

    for i in range(len(speech_timestamps)):

        # Add audio
        processed_audio.append(input_wav[speech_timestamps[i]['start']:speech_timestamps[i]['end']])

        # Add intermittent silence between segments
        if i != (len(speech_timestamps)-1):
            if speech_timestamps[i+1]["start"] - speech_timestamps[i]["end"] > silence_samples:
                processed_audio.append(np.zeros(silence_samples, dtype=np.float32))
            else:
                processed_audio.append(input_wav[speech_timestamps[i]["end"]:speech_timestamps[i+1]["start"]])

    # Add silence at the end
    processed_audio.append(np.zeros(pre_post_silence_samples, dtype=np.float32))

    final_audio = np.concatenate(processed_audio, axis=0)

    return final_audio

@app.on_event("startup")
async def startup_event():
    print("FastAPI server is starting...")
    tts_models['htr'] = initialize_tts(TTSConfig(model_name='htr_adapt_20250414_v2', steps='40000', speaker_id='htr'))
    tts_models['krw'] = initialize_tts(TTSConfig(model_name='krw_model_v1', steps='707000', speaker_id='krw'))
    tts_models['pds_announcer'] = initialize_tts(TTSConfig(model_name='pds_a_pron_20250917', steps='101000', speaker_id='pds_announcer'))
    tts_models['pds_natural'] = initialize_tts(TTSConfig(model_name='pds_ms_revised_20250818', steps='179000', speaker_id='pds_natural'))

@app.post("/tts")
async def tts_inference(request: TTSRequest):
    try:
        if request.model not in tts_models:
            raise HTTPException(status_code=400, detail="Invalid model")

        start_time = time()
        model, speaker_ids = tts_models[request.model]
        text_line = trans(request.text, if_else=False)
        audio = model.tts_to_file(text_line, speaker_ids[request.model], output_path=None, speed=request.speed)

        resampled_audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=request.sr)

        # VAD 제거
        print("requested sr: ", request.sr)
        resampled_audio = remove_silence(resampled_audio, request.sr, request.pre_post_silence_sec, request.intermittent_silence_sec)

        # BytesIO로 메모리에 쓰기
        buffer = io.BytesIO()
        try:
            sf.write(buffer, resampled_audio, request.sr, format='WAV')
            buffer.seek(0)
            audio_data = buffer.getvalue()
            buffer.close()
            return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")
        except Exception as write_error:
            buffer.close()
            raise HTTPException(status_code=500, detail=f"Audio writing failed: {str(write_error)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Inference Failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the TTS API. Use /tts endpoint for TTS inference."}

'''
source /database/venv/melotts/bin/activate 
(h100: source /share/venv_h100/melotts/bin/activate)

nohup uvicorn tts_server:app --host 0.0.0.0 --port 7009 > server.log 2>&1 &
curl -X POST "http://127.0.0.1:7009/tts" -H "Content-Type: application/json" -d @test2.json -o output.wav

curl -X POST "http://127.0.0.1:7009/tts" -H "Content-Type: application/json" -d @test2.json -o output.wav

curl -X POST "http://127.0.0.1:7009/tts" -H "Content-Type: application/json" -d @hxr1.json -o hxr_audio1.wav

curl -X POST "http://127.0.0.1:7009/tts" -H "Content-Type: application/json" -d @hxr2.json -o hxr_audio2_natural.wav
curl -X POST "http://127.0.0.1:7009/tts" -H "Content-Type: application/json" -d @hxr3.json -o hxr_audio3.wav
curl -X POST "http://127.0.0.1:7009/tts" -H "Content-Type: application/json" -d @hxr4.json -o hxr_audio4.wav
'''