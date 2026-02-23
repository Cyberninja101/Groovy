# Audio Tools

This folder stores beatmaps and helper scripts for previewing and generating drum-event JSON.

## Key Files
- `mp3_to_drums.py`: MP3 -> drum event conversion utility.
- `shift_beat.py`: timing-shift helper for beatmap events.
- `tap_test.py`: quick timing/tap utility.
- `make_preview.py`: generates `preview.html` for local timeline preview.
- `preview.html`: browser preview output.
- `latest-from-app.json`: latest payload mirrored from Pi bridge.

## Beatmap Preview Workflow

### 1) Generate/refresh preview

```bash
cd Audio
python3 make_preview.py
```

If supported by your local script args:

```bash
python3 make_preview.py path/to/your_chart.json
```

### 2) Serve over HTTP

```bash
python3 -m http.server 8000
```

### 3) Open in browser
- `http://localhost:8000/preview.html`

Important: do not open via `file://` if embedded content is used.

### 4) Iteration loop
1. Edit JSON.
2. Re-run `python3 make_preview.py`.
3. Refresh browser (`Cmd/Ctrl+R`).

## Common Issues
- Preview loads but no flashes:
  - verify JSON path in `make_preview.py`
  - check browser console for 404/errors
- Embedded video error (e.g., YouTube error 153):
  - serve with HTTP, do not use `file://`

## Optional Port

```bash
python3 -m http.server 3000
```

Then open:
- `http://localhost:3000/preview.html`
