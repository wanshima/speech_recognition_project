import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper

# Function to record audio
def record_audio(filename, duration, sample_rate=44100):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  
    print("Recording finished.")
    wav.write(filename, sample_rate, audio)

# Record audio
audio_filename = "recorded_audio.wav"
record_duration = 5  
record_audio(audio_filename, record_duration)

# Load the pre-trained Whisper model
model = whisper.load_model("base")

# Transcribe the recorded audio file
result = model.transcribe(audio_filename)

# Print the transcribed text
print("Transcribed text:", result["text"])
