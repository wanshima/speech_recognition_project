from flask import Flask, request, jsonify, send_file
import os
from pydub import AudioSegment
import subprocess
import wenet  

app = Flask(__name__)

model = wenet.load_model('chinese')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav', 'mp3', 'ogg', 'flac', 'aac'}

def secure_filename(filename):
    return os.path.basename(filename)

def convert_to_wav(audio_path):
    """ Convert an audio file to WAV format. """
    sound = AudioSegment.from_file(audio_path)
    wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
    sound.export(wav_path, format='wav')
    return wav_path

@app.route("/")
def index():
    return app.send_static_file("index.html")


def generate_speech(text, output_file='output.wav', lang='zh'):
    command = ['espeak', '-v', lang, '-w', output_file, text]
    subprocess.run(command, check=True)

@app.route("/text_to_speech")
def tts():
    text = request.args.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    output_file_path = os.path.join('/tmp', 'output.wav')
    try:
        generate_speech(text, output_file_path)
        if os.path.exists(output_file_path):
            return send_file(output_file_path, mimetype="audio/wav")
        else:
            return jsonify({"error": "Failed to create audio file"}), 500
    except subprocess.CalledProcessError:
        return jsonify({"error": "Failed to generate speech"}), 500

@app.route("/speech_to_text", methods=["POST"])
def speech_to_text():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('/tmp', filename)
        file.save(file_path)

        if not file.filename.lower().endswith('.wav'):
            file_path = convert_to_wav(file_path)  
        
        try:
            result = model.transcribe(file_path)
            recognized_text = result['text']
            os.remove(file_path)  
            return jsonify({"recognized_text": recognized_text}), 200
        except Exception as e:
            os.remove(file_path)  
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Unsupported file type"}), 400

if __name__ == "__main__":
    app.run(debug=True)
