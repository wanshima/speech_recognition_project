from flask import Flask, request, render_template, jsonify, Response, send_file
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import io
import os
import chinese_converter

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Whisper model for ASR
asr_model = whisper.load_model("base")

# Initialize ModelScope pipeline for TTS
tts_model_id = 'damo/speech_sambert-hifigan_tts_zh-cn_16k'
tts_pipeline = pipeline(task=Tasks.text_to_speech, model=tts_model_id)

# Function to record audio using sounddevice
def record_audio(filename, duration, sample_rate=44100, device=None):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16', device=device)
    sd.wait()
    print("Recording finished.")
    wav.write(filename, sample_rate, audio)
    return filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        duration = request.json.get('duration', 5)
        audio_filename = "recorded_audio.wav"
        audio_filename = record_audio(audio_filename, duration)
    else:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            audio_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(audio_filename)
    
    # Transcribe the audio file
    result = asr_model.transcribe(audio_filename)
    transcribed_text = result["text"]
    
    # Convert Traditional Chinese to Simplified Chinese
    simplified_text = chinese_converter.to_simplified(transcribed_text)
    
    return jsonify({'transcribed_text': simplified_text})

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Generate speech from the text
    output = tts_pipeline(input=text, voice='zhitian_emo')
    
    # Extract the generated wav file from the output
    wav_data = output[OutputKeys.OUTPUT_WAV]
    
    # Save the generated speech to a file
    output_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'output.wav')
    with open(output_filename, 'wb') as f:
        f.write(wav_data)
    
    return send_file(output_filename, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

