# Groovy Frontend (Expo)

React Native app (Expo) for selecting songs/uploads, choosing speed, and submitting payloads to backend.

## Key Behavior
- Displays hardcoded songs and uploaded MP3 entries.
- Uploads MP3 files to backend (`/upload`).
- Submits selected source to backend (`/submit/...`).
- Always sends `notes_at_a_time` from speed picker for both hardcoded and upload flows.

## Important Config
- Backend URL is hardcoded in:
  - `App/frontend/app/(tabs)/index.tsx`
  - `const BACKEND_BASE = "http://<host>:8000"`

Update this to your backend machine IP before testing on a device.

## Run

```bash
cd App/frontend
npm install
npm run start
```

Optional shortcuts:

```bash
npm run ios
npm run android
npm run web
```

## Main UI File
- `App/frontend/app/(tabs)/index.tsx`

This screen owns:
- song/upload selection
- speed picker options
- upload + submit network calls
