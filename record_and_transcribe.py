import sounddevice as sd
from scipy.io.wavfile import write
import whisper

def record_audio(filename, duration, samplerate=16000, device_index=None):
    print("Recording...")
    try:
        if device_index is not None:
            device_info = sd.query_devices(device_index, kind='input')
            print(f"Using device: {device_info['name']}")
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16', device=device_index)
        sd.wait()  # Wait until the recording is finished
        write(filename, samplerate, recording)
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
    record_audio(audio_filename, record_duration, device_index=0)  # Use the correct device index (e.g., 0)
    
    # Transcribe the recorded audio
    transcription = transcribe_audio(audio_filename)
    print("Transcribed text:", transcription)

if __name__ == "__main__":
    main()
