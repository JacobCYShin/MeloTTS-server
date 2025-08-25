# -*- coding: utf-8 -*-
# Your other code starts here
import os
# ...
from melo.api import TTS

# Speed is adjustable
speed = 1.5
device = 'cuda:0' # or cuda:0

text = "테스트 문구"
model = TTS(language='KR', device=device)
speaker_ids = model.hps.data.spk2id

output_path = 'kr.wav'
model.tts_to_file(text, speaker_ids['KR'], output_path, speed=speed)
