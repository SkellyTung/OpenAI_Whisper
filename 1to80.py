import numpy as np
import librosa
import soundfile as sf

# 讀取單通道音頻
audio_data, sample_rate = librosa.load('audio.wav', sr=None, mono=True)

# 將單通道音頻擴展為80通道
expanded_audio = np.repeat(audio_data[:, np.newaxis], 80, axis=1)

# 寫入80通道音頻文件
sf.write('80_channel_audio.wav', expanded_audio, sample_rate)
