from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os, uuid, time, asyncio, json
import websockets
import os, json, re
from flask import jsonify
from drumchart import DrumChartConfig, DrumChartGenerator
from pathlib import Path

import librosa
import numpy as np

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

#logic for reading JSON
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEATMAP_DIR = os.path.join(BASE_DIR, "beatmaps")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/uploads", methods=["GET"])
def list_uploads():
    files = sorted(os.listdir(UPLOAD_DIR), reverse=True)

    items = []
    for fn in files:
        # stored as "<upload_id>_<original>.mp3"
        if "_" in fn:
            upload_id, original = fn.split("_", 1)
        else:
            upload_id, original = fn, fn
        items.append({"id": upload_id, "name": original})

    return jsonify({"items": items})

@app.route("/uploads/<path:filename>", methods=["GET"])
def get_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# 🔵 CHANGE THIS TO YOUR PI IP
PI_IP = "10.48.44.30"
PI_WS_URL = f"ws://{PI_IP}:8765"

# -------------------------
# WebSocket sender to Pi
# -------------------------

async def _send_to_pi_async(payload: dict):
    async with websockets.connect(PI_WS_URL, max_size=50 * 1024 * 1024) as ws:
        await ws.send(json.dumps(payload))
        ack = await ws.recv()
        return json.loads(ack)

def send_to_pi(payload: dict) -> dict:
    return asyncio.run(_send_to_pi_async(payload))


# -------------------------
# MP3 → Beat JSON
# -------------------------

def mp3_to_drum_events_json(mp3_path: str, title: str) -> dict:
    cfg = DrumChartConfig()
    gen = DrumChartGenerator(cfg, prefer_ffmpeg=True)
    result = gen.generate(Path(mp3_path), title=title)

    return result
#loading by name
def load_beatmap(name: str) -> dict:
    # allow only safe filenames like brandy-1350, song_2, etc.
    if not re.match(r"^[a-zA-Z0-9_-]+$", name):
        raise ValueError("bad name")

    path = os.path.join(BEATMAP_DIR, f"{name}.json")
    with open(path, "r") as f:
        return json.load(f)



@app.route("/submit/<upload_id>", methods=["POST"])
def submit(upload_id):
    # find the saved mp3 by prefix
    matches = [fn for fn in os.listdir(UPLOAD_DIR) if fn.startswith(upload_id + "_")]
    if not matches:
        return jsonify({"ok": False, "error": "Upload not found"}), 404

    mp3_path = os.path.join(UPLOAD_DIR, matches[0])

    # OPTION A: send hardcoded beatmap JSON
    beatmap = load_beatmap("brandy-1350")

    # OPTION B: or generate from the MP3 you saved
    # beatmap = mp3_to_beats_json(mp3_path, title=matches[0])

    try:
        pi_ack = send_to_pi(beatmap)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": True, "pi_ack": pi_ack})
# -------------------------
# Upload Route
# -------------------------

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify(error="No file field named 'file'"), 400

    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify(error="No file selected"), 400

    filename = secure_filename(f.filename)
    if not filename.lower().endswith(".mp3"):
        return jsonify(error="Only .mp3 files allowed"), 400

    upload_id = uuid.uuid4().hex
    stored_name = f"{upload_id}_{filename}"
    mp3_path = os.path.join(UPLOAD_DIR, stored_name)
    f.save(mp3_path)

    # 1️⃣ Convert MP3 to beat JSON
    try:
        beat_json = mp3_to_drum_events_json(mp3_path, filename)
    except Exception as e:
        return jsonify(error=f"MP3 processing failed: {str(e)}"), 500

    # 2️⃣ Send JSON to Pi
    try:
        beatmap = load_beatmap("brandy-1350")
        pi_ack = send_to_pi(beat_json)
        #pi_ack = send_to_pi(beat_json)
    except Exception as e:
        pi_ack = {"ok": False, "error": str(e)}

    return jsonify(
        ok=True,
        id=upload_id,
        name=filename,
        tempo_bpm=beat_json["tempo_bpm"],
        beats_detected=len(beat_json["events"]),
        pi_ack=pi_ack
    ), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)