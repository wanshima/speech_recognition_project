import sounddevice as sd
from scipy.io.wavfile import write
import whisper

def record_audio(filename, duration, samplerate=16000, device=0):
    print("Recording...")
    # Start recording using the specified device
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16', device=device)
    sd.wait()  # Wait until the recording is finished
    write(filename, samplerate, recording)
    print(f"Recording saved as {filename}")

def transcribe_audio(filename):
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    return result["text"]

def main():
    # Record audio
    audio_filename = "recorded_audio.wav"
    record_duration = 10  # seconds
    record_audio(audio_filename, record_duration, device=0)  # Use device index 0 for MacBook Pro Microphone
    
    # Transcribe the recorded audio
    transcription = transcribe_audio(audio_filename)
    print("Transcribed text:", transcription)

if __name__ == "__main__":
    main()
