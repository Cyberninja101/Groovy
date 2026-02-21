from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os, uuid, time, asyncio, json
import websockets

import librosa
import numpy as np

app = Flask(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

def mp3_to_beats_json(mp3_path: str, title: str) -> dict:
    y, sr = librosa.load(mp3_path, sr=None, mono=True)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    events = [
        {"t_ms": int(round(t * 1000)), "drum": 1}
        for t in beat_times
    ]

    return {
        "title": title,
        "tempo_bpm": float(tempo),
        "events": events,
        "meta": {
            "sample_rate": int(sr),
            "num_beats": len(events)
        }
    }


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
        beat_json = mp3_to_beats_json(mp3_path, filename)
    except Exception as e:
        return jsonify(error=f"MP3 processing failed: {str(e)}"), 500

    # 2️⃣ Send JSON to Pi
    try:
        pi_ack = send_to_pi(beat_json)
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