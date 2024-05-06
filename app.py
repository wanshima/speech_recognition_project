from flask import Flask, request, jsonify, send_file
from gtts import gTTS
import os

app = Flask(__name__)

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/text_to_speech")
def text_to_speech():
    text = request.args.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    tts = gTTS(text, lang='zh')
    audio_file_path = "temp_audio.mp3"
    tts.save(audio_file_path)

    return send_file(audio_file_path, mimetype="audio/mpeg")

@app.route("/speech_to_text", methods=["POST"])
def speech_to_text():
    # Your speech to text code here
    pass

if __name__ == "__main__":
    app.run(debug=True)
