from flask import Flask, request, render_template, jsonify, Response
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import subprocess
import os
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            audio_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(audio_filename)
    else:
        duration = request.json.get('duration', 5)
        audio_filename = "recorded_audio.wav"
        audio_filename = record_audio(audio_filename, duration)
    
    # Run the Whisper command with initial prompt for Simplified Chinese
    result = subprocess.run(
        ['whisper', '--language', 'Chinese', '--model', 'large', audio_filename, '--initial_prompt', '以下是普通话的句子。'],
        capture_output=True, text=True
    )
    transcribed_text = result.stdout.strip()
    
    return jsonify({'transcribed_text': transcribed_text})

@app.route('/synthesize', methods=['POST'])
def synthesize():
    text = request.json.get('text', '')
    
    # Generate speech from the text
    output = tts_pipeline(input=text, voice='zhitian_emo')
    
    # Extract the generated wav file from the output
    wav_data = output[OutputKeys.OUTPUT_WAV]
    
    # Play the generated speech directly
    return Response(io.BytesIO(wav_data), mimetype="audio/wav")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
