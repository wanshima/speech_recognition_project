import sounddevice as sd
from scipy.io.wavfile import write
import whisper

# Set the default sample rate and channels
sd.default.samplerate = 16000
sd.default.channels = 1

def record_audio(filename, duration):
    print("Recording...")
    try:
        frames = int(duration * sd.default.samplerate)
        recording = sd.rec(frames, dtype='int16')
        sd.wait()  # Wait until the recording is finished
        write(filename, sd.default.samplerate, recording)
        print(f"Recording saved as {filename}")
    except Exception as e:
        print(f"An error occurred while recording audio: {e}")
        print("Available devices:")
        print(sd.query_devices())
        return

def transcribe_audio(filename):
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    return result["text"]

def main():
    # List available devices
    print("Available devices:")
    print(sd.query_devices())

    # Record audio
    audio_filename = "recorded_audio.wav"
    record_duration = 10  # seconds
    record_audio(audio_filename, record_duration)  # Use the default device
    
    # Transcribe the recorded audio
    transcription = transcribe_audio(audio_filename)
    print("Transcribed text:", transcription)

if __name__ == "__main__":
    main()
