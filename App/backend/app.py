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
    beatmap_name = request.args.get("beatmap")  # if present => hardcoded mode

    # ---- NEW: read speed from query param (sent only on submit) ----
    notes_raw = request.args.get("notes_at_a_time", None)
    try:
        speed = float(notes_raw) if notes_raw is not None else 1.0
    except ValueError:
        return jsonify(ok=False, error="notes_at_a_time must be numeric (e.g. 0.5, 1.25, -1)"), 400

    # MODE A: hardcoded beatmap if query param present
    if beatmap_name:
        try:
            beatmap = load_beatmap(beatmap_name)
        except Exception as e:
            return jsonify(ok=False, error=f"Beatmap load failed: {str(e)}"), 400

        # ---- NEW: include speed in payload to Pi ----
        beatmap["speed"] = speed

        try:
            pi_ack = send_to_pi(beatmap)
        except Exception as e:
            return jsonify(ok=False, error=f"Pi send failed: {str(e)}"), 500

        return jsonify(ok=True, pi_ack=pi_ack, source="hardcoded", beatmap=beatmap_name, speed=speed)

    # MODE B: uploaded MP3 id (convert + send)
    matches = [fn for fn in os.listdir(UPLOAD_DIR) if fn.startswith(upload_id + "_")]
    if not matches:
        return jsonify(ok=False, error="Upload not found"), 404

    fn = matches[0]
    mp3_path = os.path.join(UPLOAD_DIR, fn)
    original_name = fn.split("_", 1)[1] if "_" in fn else fn

    try:
        beat_json = mp3_to_drum_events_json(mp3_path, original_name)
    except Exception as e:
        return jsonify(ok=False, error=f"MP3 processing failed: {str(e)}"), 500

    # ---- NEW: include speed in payload to Pi ----
    beat_json["speed"] = speed

    try:
        pi_ack = send_to_pi(beat_json)
    except Exception as e:
        return jsonify(ok=False, error=f"Pi send failed: {str(e)}"), 500

    return jsonify(
        ok=True,
        pi_ack=pi_ack,
        source="mp3",
        num_events=len(beat_json.get("events", [])),
        speed=speed,
    )
# -------------------------
# Upload Route
# -------------------------

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify(ok=False, error="No file field named 'file'"), 400

    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify(ok=False, error="No file selected"), 400

    filename = secure_filename(f.filename)
    if not filename.lower().endswith(".mp3"):
        return jsonify(ok=False, error="Only .mp3 files allowed"), 400

    upload_id = uuid.uuid4().hex
    stored_name = f"{upload_id}_{filename}"
    mp3_path = os.path.join(UPLOAD_DIR, stored_name)
    f.save(mp3_path)

    # STORE ONLY
    return jsonify(ok=True, id=upload_id, name=filename), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)