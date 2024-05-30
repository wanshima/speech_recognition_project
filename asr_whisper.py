import whisper

# Load the pre-trained Whisper model
model = whisper.load_model("base")

# Path to your audio file
audio_path = "path/to/your/audio/file.wav"

# Transcribe the audio file
result = model.transcribe(audio_path)

# Print the transcribed text
print("Transcribed text:", result["text"])
