import torch
import torch.onnx as onnx
import torchaudio
import whisper
import onnxruntime as ort

# 載入模型
model = whisper.load_model("base")

# 創建一個虛擬的輸入
num_samples = 16000  # 假設音頻文件有 16000 個樣本
input_size = (1, 1, num_samples)  # 設置輸入形狀為 1x1xnum_samples
dummy_input = torch.randn(*input_size)

# 添加模擬的 tokens 參數
tokens = torch.tensor([0])  # 模擬一個 tokens 張量
dummy_input = (dummy_input, tokens)

# 將模型轉換為ONNX格式
onnx_filename = "model.onnx"  # 設定要保存的ONNX文件名稱
onnx.export(model, dummy_input, onnx_filename)

print("模型已成功保存為ONNX文件：", onnx_filename)

# 載入ONNX模型
ort_session = ort.InferenceSession(onnx_filename)

# 準備輸入數據
audio, sample_rate = torchaudio.load("audio.wav")
input_data = audio.unsqueeze(0).unsqueeze(0).numpy()

# 執行推論
outputs = ort_session.run(None, {'input': input_data})

# 解析輸出結果
transcription = outputs[0][0].decode('utf-8')

print("識別結果：", transcription)
