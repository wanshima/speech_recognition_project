# Flask ASR and TTS Application

## Prerequisites

- Linux
- Python 3.8
- pip 

## Installation

### Clone the Repository

```bash
git clone https://github.com/wanshima/speech_recognition_project
cd speech_recognition_project
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Check Available Audio Devices

Run the following code in a Python shell:

```python
import sounddevice as sd
print(sd.query_devices())
```

Example output:

```
> 0 MacBook Pro Microphone, Core Audio (1 in, 0 out)
< 1 MacBook Pro Speakers, Core Audio (0 in, 2 out)
  2 Microsoft Teams Audio, Core Audio (2 in, 2 out)
  3 ZoomAudioDevice, Core Audio (2 in, 2 out)
```

### Update `record_audio` Function

In the `app.py` file, update the `record_audio` function with the device index:

```python
def record_audio(filename, duration, sample_rate=44100, device=None):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16', device=device)
    sd.wait()
    print("Recording finished.")
    wav.write(filename, sample_rate, audio)
    return filename
```

## API Endpoints

### `GET /`

Displays basic instructions.

### `POST /transcribe`

Transcribes the audio file or recorded audio to text and converts Traditional Chinese to Simplified Chinese.

**Request:**

- If uploading a file:
  - Form-data: `file` (type: File)
- If recording audio:
  - JSON: `{ "duration": 5 }`

**Response:**

```json
{
  "transcribed_text": "你好"
}
```

### `POST /synthesize`

Synthesizes speech from the provided text and returns the audio file.

**Request:**

- JSON: `{ "text": "你好" }`

**Response:**

Returns a binary audio file.