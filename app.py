from flask import Flask, request, render_template, jsonify
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

app = Flask(__name__)

# Initialize Whisper model for ASR
asr_model = whisper.load_model("base")

# Initialize ModelScope pipeline for TTS
tts_model_id = 'damo/speech_sambert-hifigan_tts_zh-cn_16k'
tts_pipeline = pipeline(task=Tasks.text_to_speech, model=tts_model_id)

# Function to record audio
def record_audio(filename, duration, sample_rate=44100):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  
    print("Recording finished.")
    wav.write(filename, sample_rate, audio)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    duration = request.json.get('duration', 5)
    audio_filename = "recorded_audio.wav"
    record_audio(audio_filename, duration)
    
    # Transcribe the recorded audio file
    result = asr_model.transcribe(audio_filename)
    transcribed_text = result["text"]
    
    return jsonify({'transcribed_text': transcribed_text})

@app.route('/synthesize', methods=['POST'])
def synthesize():
    text = request.json.get('text', '')
    
    # Generate speech from the text
    output = tts_pipeline(input=text, voice='zhitian_emo')
    
    # Extract the generated wav file from the output
    wav_data = output[OutputKeys.OUTPUT_WAV]
    
    # Save the generated speech to a file
    output_filename = 'output.wav'
    with open(output_filename, 'wb') as f:
        f.write(wav_data)
    
    return jsonify({'audio_file': output_filename})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
