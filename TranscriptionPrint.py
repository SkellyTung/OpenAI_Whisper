import whisper

model = whisper.load_model("base")
# result = model.transcribe("audio.flac")
# result = model.transcribe("audio.mp3")
result = model.transcribe("audio.wav")

# print(result["text"])
print("識別结果： " + result["text"])
