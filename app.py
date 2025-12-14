from flask import Flask, request, render_template, send_from_directory
import os
import json

from model import run_target_speaker_extraction

# Local whisper + translator
from faster_whisper import WhisperModel
from googletrans import Translator

# Load Whisper Model
model = WhisperModel("small", device="cpu", compute_type="int8")

# Translator
translator = Translator()

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload")
def upload_page():
    return render_template("temp.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory("uploads", filename)


@app.route("/outputs/<filename>")
def output_file(filename):
    return send_from_directory("outputs", filename)


@app.route("/extract", methods=["POST"])
def extract():

    # -----------------------------------------------------
    # 🎤 CASE 1 — TRANSCRIPTION REQUEST
    # -----------------------------------------------------
    if "transcribe" in request.form:

        audio_path = request.form["enhanced_audio"]
        lang = request.form["language"]

        # Restore previously saved fields
        similarities_json = request.form.get("similarities")
        similarities = json.loads(similarities_json) if similarities_json else []

        selected_source_path = request.form.get("selected_source_path")
        mixed_audio_path = request.form.get("mixed_audio_path")

        # 🔊 Transcribe locally
        segments, info = model.transcribe(audio_path, beam_size=5)
        text = " ".join([seg.text for seg in segments])

        # 🌐 Translate
        if lang == "telugu":
            text = translator.translate(text, dest="te").text
        elif lang == "hindi":
            text = translator.translate(text, dest="hi").text
        else:
            text = translator.translate(text, dest="en").text

        return render_template(
            "last.html",
            enhanced_audio_path=audio_path,
            mixed_audio_path=mixed_audio_path,
            similarities=similarities,
            selected_source_path=selected_source_path,
            transcription=text
        )

    # -----------------------------------------------------
    # 🎛 CASE 2 — NORMAL EXTRACTION PIPELINE
    # -----------------------------------------------------
    try:
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)

        mixed_file = request.files["mixed"]
        target_file = request.files["target"]

        mixed_path = os.path.join("uploads", "mixed.wav")
        target_path = os.path.join("uploads", "target.wav")

        mixed_file.save(mixed_path)
        target_file.save(target_path)

        # Run your TSE model pipeline
        result = run_target_speaker_extraction(
            mixed_audio_path=mixed_path,
            target_audio_path=target_path,
            out_folder="outputs"
        )

        return render_template(
            "last.html",
            mixed_audio_path=mixed_path.replace("\\", "/"),
            enhanced_audio_path=result["enhanced_output_path"].replace("\\", "/"),
            selected_source_path=result["selected_source_path"],
            similarities=result["similarities"],
            transcription=None
        )

    except Exception as e:
        return render_template(
            "last.html",
            error=str(e),
            mixed_audio_path="",
            enhanced_audio_path="",
            similarities=[]
        )


if __name__ == "__main__":
    app.run(debug=True)
