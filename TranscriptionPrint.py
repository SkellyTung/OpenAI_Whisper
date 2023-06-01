import pyaudio
import whisper
# 將Python程式碼轉換為ONNX模型檔案
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import os

# 獲取當前檔案的路徑 test
# 在程式碼的開頭使用__file__變數來獲取當前檔案的路徑，然後搭配os.path模組中的函數來處理路徑，以獲取"audio.wav"和"model.onnx"的完整路徑。
current_path = os.path.dirname(os.path.abspath(__file__))

# 配置錄音參數
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = os.path.join(current_path, "audio.wav")  # 完整的音訊檔案路徑


model = whisper.load_model("base")
# result = model.transcribe("audio.flac")
# result = model.transcribe("audio.mp3")
result = model.transcribe("audio.wav")

# print(result["text"])
print("識別结果： " + result["text"])


# ...

# 將Python程式碼轉換為ONNX模型檔案

# 創建計算圖
graph = helper.make_graph(
    nodes=[],
    name="speech_recognition",
    inputs=[],
    outputs=[],
    initializer=[],
)

# 輸入
input_name = "input"
input_shape = (1, 16000)
input_type = onnx.TensorProto.FLOAT

input_tensor = helper.make_tensor_value_info(
    name=input_name,
    elem_type=input_type,
    shape=input_shape
)

graph.input.extend([input_tensor])

# 輸出
output_name = "output"
output_shape = (1,16000)
output_type = onnx.TensorProto.STRING

output_tensor = helper.make_tensor_value_info(
    name=output_name,
    elem_type=output_type,
    shape=output_shape
)

graph.output.extend([output_tensor])

# 初始化器
initializer = [
    numpy_helper.from_array(np.random.randn(3, 3, 1, 32).astype(np.float32), name='weight'),
    numpy_helper.from_array(np.zeros(32, dtype=np.float32), name='bias')
]

graph.initializer.extend(initializer)

# 創建ONNX模型
model = helper.make_model(graph, producer_name='speech_recognition')

# 將模型保存為ONNX檔案
onnx_file_path = os.path.join(current_path, "speech_recognition.onnx")
onnx.save_model(model, onnx_file_path)

print("Conversion to ONNX is completed. The ONNX model is saved at: " + onnx_file_path)
