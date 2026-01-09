import whisper

model = whisper.load_model("base")
result = model.transcribe("input.wav")

print("Transcription:", result["text"])
