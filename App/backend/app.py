from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os, uuid, time

app = Flask(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# super simple in-memory "DB" (resets when server restarts)
UPLOADS = []

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
    out_name = f"{upload_id}_{filename}"
    out_path = os.path.join(UPLOAD_DIR, out_name)
    f.save(out_path)

    UPLOADS.insert(0, {
        "id": upload_id,
        "name": filename,
        "stored": out_name,
        "ts": int(time.time())
    })

    return jsonify(ok=True, id=upload_id, name=filename), 200

@app.route("/uploads", methods=["GET"])
def uploads():
    return jsonify(items=UPLOADS), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)