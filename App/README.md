# Groovy App Stack

This folder contains the mobile frontend and Flask backend that control beatmap submission to the Pi/CV runtime.

## Structure
- `App/backend/`: Flask API + beatmap conversion + Pi WebSocket forwarder.
- `App/frontend/`: Expo React Native UI for upload/song selection and speed control.

## Backend (`App/backend`)

### What it does
- Accepts MP3 uploads.
- Lists uploaded files.
- Loads hardcoded beatmaps from `App/backend/beatmaps/`.
- Converts uploaded MP3 files to drum events via `drumchart.py`.
- Sends resulting JSON payload to the Pi bridge at `ws://<PI_IP>:8765`.

### Important config
- `PI_IP` in `App/backend/app.py`

### API endpoints
- `GET /uploads`
- `GET /uploads/<filename>`
- `POST /upload`
- `POST /submit/<upload_id>?notes_at_a_time=<speed>&beatmap=<optional>`

Notes:
- `notes_at_a_time` is parsed as speed and always overrides outgoing payload speed.
- If `beatmap` is present, backend loads `App/backend/beatmaps/<beatmap>.json`.

### Run backend

```bash
python3 -m pip install flask websockets librosa numpy
python3 App/backend/app.py
```

## Frontend (`App/frontend`)

### What it does
- Lets you select a hardcoded beatmap or upload an MP3.
- Lets you choose speed (`notes_at_a_time`), including `-1` for one-note-at-a-time flow.
- Sends submit requests to backend and displays backend/Pi ACK.

### Important config
- `BACKEND_BASE` in `App/frontend/app/(tabs)/index.tsx`

### Run frontend

```bash
cd App/frontend
npm install
npm run start
```

Use Expo tooling to open on iOS/Android/web as needed.
