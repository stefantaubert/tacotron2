
import os
import wave
with wave.open('/datasets/thchs_wav/wav/train/A11/A11_1.WAV', "rb") as wave_file:
  frame_rate = wave_file.getframerate()
  print(frame_rate)
        
x = '/datasets/LJSpeech-1.1-lite/wavs/LJ001-0001.wav'

with wave.open(x, "rb") as wave_file:
  frame_rate = wave_file.getframerate()
  print(frame_rate)
