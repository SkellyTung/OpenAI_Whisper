import torch
import torch.onnx as onnx
import torchaudio
import whisper
import onnxruntime as ort
import torchaudio
torchaudio.set_audio_backend("soundfile")


# 載入模型
model = whisper.load_model("base")
transcription_result = model.transcribe("audio.flac")
# transcription_result = model.transcribe("audio.mp3")
# transcription_result = model.transcribe("audio.wav")

# print(transcription_result["text"])
print("識別结果： " + transcription_result["text"])


# 創建一個虛擬的輸入
num_samples = 16000  # 假設音頻文件有 16000 個樣本
input_size = (1, 1, num_samples)  # 設置輸入形狀為 1x1xnum_samples
dummy_input = torch.randn(*input_size)

# 將模型轉換為ONNX格式
onnx_filename = "model.onnx"  # 設定要保存的ONNX文件名稱
tokens = torch.tensor([0])  # 添加 'tokens' 參數
onnx.export(model, (dummy_input, tokens), onnx_filename, input_names=['input', 'tokens'], output_names=['output'])

print("模型已成功保存為ONNX文件：", onnx_filename)

# 載入ONNX模型
ort_session = ort.InferenceSession(onnx_filename)

# 準備輸入數據
audio, sample_rate = torchaudio.load("audio.flac")
audio = audio.squeeze()  # 去除多餘的維度，使其成為一維的音頻數據

# 截斷或填充音頻數據以符合模型的要求
desired_length = num_samples  # 模型要求的音頻長度
if audio.size(0) > desired_length:
    audio = audio[:desired_length]
else:
    padding = desired_length - audio.size(0)
    audio = torch.nn.functional.pad(audio, (0, padding))

input_data = audio.unsqueeze(0).unsqueeze(0).numpy()

# 執行推論
outputs = ort_session.run(None, {'input': input_data, 'tokens': [0]})

# 解析輸出結果
transcription = outputs[0][0].decode('utf-8')

print("識別結果：", transcription)
