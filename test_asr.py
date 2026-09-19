from app.asr import ASREngine

audio_path = "audio/test_hindi.wav"

print("Loading Whisper model...")

asr = ASREngine(
    model_size="base",
    device="cpu",
    compute_type="int8",
)

print("Model loaded.")
print("Transcribing audio...")

result = asr.transcribe(audio_path)

print("\n========== ASR RESULT ==========")
print("Detected language:", result["language"])
print(
    "Language probability:",
    result["language_probability"]
)
print("Transcript:", result["text"])
print("================================")