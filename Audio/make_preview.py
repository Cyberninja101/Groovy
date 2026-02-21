#!/usr/bin/env python3
import json
from pathlib import Path

JSON_PATH = Path("brandy-1350.json")   # change if needed
OUT_HTML  = Path("preview.html")

def main():
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    events = data.get("events", [])
    title = data.get("title", "drum_preview")
    json_name = JSON_PATH.name

    if not events:
        raise ValueError("No events found in JSON (data['events'] is empty).")

    # basic stats so you can sanity-check whether you're actually loading the shifted file
    t_min = min(int(e["t_ms"]) for e in events if "t_ms" in e)
    t_max = max(int(e["t_ms"]) for e in events if "t_ms" in e)

    # Embed events directly into HTML as JS array
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title} – Drum Preview</title>
  <style>
    body {{
      font-family: -apple-system, system-ui, Segoe UI, Roboto, Arial, sans-serif;
      margin: 16px;
      background: #0b0b0f;
      color: #f2f2f6;
    }}
    .row {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }}
    .card {{
      background: #151522;
      border: 1px solid #2a2a3d;
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }}
    .controls input {{
      width: 340px;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid #2a2a3d;
      background: #0f0f18;
      color: #f2f2f6;
    }}
    button {{
      padding: 10px 14px;
      border-radius: 10px;
      border: 1px solid #2a2a3d;
      background: #1c1c2b;
      color: #f2f2f6;
      cursor: pointer;
    }}
    button:hover {{ background: #242438; }}
    button:disabled {{ opacity: 0.5; cursor: not-allowed; }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(200px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .pad {{
      height: 140px;
      border-radius: 18px;
      border: 2px solid #2a2a3d;
      background: #101021;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      overflow: hidden;
      user-select: none;
    }}
    .pad .label {{
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0.3px;
    }}
    .pad .sub {{
      position: absolute;
      bottom: 10px;
      left: 12px;
      font-size: 14px;
      opacity: 0.8;
    }}
    .lit {{
      filter: brightness(1.7);
      border-color: #ffffff;
      box-shadow: 0 0 30px rgba(255,255,255,0.25);
      background: #1d1d35;
    }}
    .meta {{
      margin-top: 10px;
      opacity: 0.85;
      font-size: 14px;
      line-height: 1.4;
    }}
    #player {{
      width: 100%;
      max-width: 720px;
      aspect-ratio: 16 / 9;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid #2a2a3d;
    }}
    .slider {{
      display: grid;
      grid-template-columns: 120px 1fr 80px;
      gap: 10px;
      align-items: center;
      margin-top: 10px;
    }}
    .slider input[type="range"] {{ width: 100%; }}
    code {{
      background: #0f0f18;
      border: 1px solid #2a2a3d;
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <h2 style="margin: 0 0 6px 0;">Drum Cue Preview – {title}</h2>
  <div class="meta" style="margin: 0 0 12px 0;">
    Source JSON: <code>{json_name}</code> • events: <code>{len(events)}</code> • first t_ms: <code>{t_min}</code> • last t_ms: <code>{t_max}</code>
  </div>

  <div class="row">
    <div class="card controls" style="flex: 1; min-width: 360px;">
      <div class="row" style="margin-bottom: 10px;">
        <input id="ytInput" placeholder="Paste YouTube URL or ID (e.g. dQw4w9WgXcQ)" />
        <button id="loadBtn">Load Video</button>
      </div>

      <div class="row" style="margin-bottom: 8px;">
        <button id="startBtn" disabled>Start</button>
        <button id="pauseBtn" disabled>Pause</button>
        <button id="resetBtn" disabled>Reset</button>
      </div>

      <div class="slider">
        <div>Offset (ms)</div>
        <input id="offset" type="range" min="-3000" max="3000" value="0" />
        <div id="offsetVal">0</div>
      </div>

      <div class="slider">
        <div>Flash (ms)</div>
        <input id="flash" type="range" min="40" max="400" value="120" />
        <div id="flashVal">120</div>
      </div>

      <div class="meta" id="status">
        Loaded <code>{json_name}</code> (first event at {t_min}ms). Load a video, then press <code>Start</code>.
      </div>
    </div>

    <div class="card" style="flex: 1; min-width: 360px;">
      <div id="player"></div>
      <div class="meta">
        Tip: If cues feel early/late, adjust <b>Offset</b> until kick+snare match what you hear.
      </div>
    </div>
  </div>

  <div class="grid">
    <div class="pad" id="pad1"><div class="label">KICK</div><div class="sub">a • id 1</div></div>
    <div class="pad" id="pad3"><div class="label">SNARE</div><div class="sub">s • id 3</div></div>
    <div class="pad" id="pad5"><div class="label">HI-HAT</div><div class="sub">k • id 5</div></div>
    <div class="pad" id="pad6"><div class="label">CRASH</div><div class="sub">l • id 6</div></div>
  </div>

  <script>
    const EVENTS = {json.dumps(events)};
    const JSON_NAME = {json.dumps(json_name)};
    const FIRST_T_MS = {t_min};

    const PAD = {{
      1: document.getElementById("pad1"),
      3: document.getElementById("pad3"),
      5: document.getElementById("pad5"),
      6: document.getElementById("pad6"),
    }};

    const statusEl = document.getElementById("status");
    const loadBtn = document.getElementById("loadBtn");
    const startBtn = document.getElementById("startBtn");
    const pauseBtn = document.getElementById("pauseBtn");
    const resetBtn = document.getElementById("resetBtn");

    const offsetSlider = document.getElementById("offset");
    const flashSlider = document.getElementById("flash");
    const offsetVal = document.getElementById("offsetVal");
    const flashVal = document.getElementById("flashVal");

    offsetVal.textContent = offsetSlider.value;
    flashVal.textContent = flashSlider.value;

    offsetSlider.addEventListener("input", () => offsetVal.textContent = offsetSlider.value);
    flashSlider.addEventListener("input", () => flashVal.textContent = flashSlider.value);

    function parseYouTubeId(input) {{
      const s = input.trim();
      if (!s) return null;
      if (/^[a-zA-Z0-9_-]{{11}}$/.test(s)) return s;

      const vMatch = s.match(/[?&]v=([^&]+)/);
      if (vMatch) return vMatch[1];

      const shortMatch = s.match(/youtu\\.be\\/([^?&]+)/);
      if (shortMatch) return shortMatch[1];

      const embedMatch = s.match(/\\/embed\\/([^?&]+)/);
      if (embedMatch) return embedMatch[1];

      return null;
    }}

    function flashPad(drumId, flashMs) {{
      const el = PAD[drumId];
      if (!el) return;
      el.classList.add("lit");
      setTimeout(() => el.classList.remove("lit"), flashMs);
    }}

    let player = null;
    let tickHandle = null;
    let eventIdx = 0;
    let playing = false;

    function resetCueState() {{
      eventIdx = 0;
    }}

    function tick() {{
      if (!player || !playing) return;

      const tSec = player.getCurrentTime ? player.getCurrentTime() : 0;
      const tMs = Math.floor(tSec * 1000);
      const offsetMs = parseInt(offsetSlider.value, 10);
      const flashMs = parseInt(flashSlider.value, 10);

      const chartMs = tMs + offsetMs;

      while (eventIdx < EVENTS.length && EVENTS[eventIdx].t_ms <= chartMs) {{
        flashPad(EVENTS[eventIdx].drum, flashMs);
        eventIdx++;
      }}

      const next = (eventIdx < EVENTS.length) ? EVENTS[eventIdx].t_ms : null;
      if (next !== null) {{
        statusEl.textContent =
          `JSON=${{JSON_NAME}} (first=${{FIRST_T_MS}}ms) | video=${{tMs}}ms, chart=${{chartMs}}ms | next in ${{Math.max(0, next - chartMs)}}ms | fired ${{eventIdx}}/${{EVENTS.length}}`;
      }} else {{
        statusEl.textContent = `Done: fired all ${{EVENTS.length}} events. Hit Reset to replay.`;
      }}
    }}

    const tag = document.createElement('script');
    tag.src = "https://www.youtube.com/iframe_api";
    document.head.appendChild(tag);

    window.onYouTubeIframeAPIReady = function() {{
      statusEl.textContent = `YouTube API ready. Loaded ${{JSON_NAME}} (first=${{FIRST_T_MS}}ms). Load a video.`;
    }}

    function createPlayer(videoId) {{
      if (player && player.destroy) player.destroy();

      player = new YT.Player('player', {{
        height: '100%',
        width: '100%',
        videoId: videoId,
        playerVars: {{ modestbranding: 1, rel: 0 }},
        events: {{
          'onReady': () => {{
            startBtn.disabled = false;
            pauseBtn.disabled = false;
            resetBtn.disabled = false;
            statusEl.textContent = `Video loaded. Press Start. (JSON first event = ${{FIRST_T_MS}}ms)`;
          }},
          'onStateChange': (e) => {{
            if (e.data === 0) playing = false;
          }}
        }}
      }});
    }}

    loadBtn.addEventListener("click", () => {{
      const input = document.getElementById("ytInput").value;
      const vid = parseYouTubeId(input);
      if (!vid) {{
        statusEl.textContent = "Couldn’t parse YouTube ID. Paste a normal YouTube URL or the 11-char ID.";
        return;
      }}
      if (typeof YT === "undefined" || !YT.Player) {{
        statusEl.textContent = "YouTube API not loaded yet. Wait 1–2 seconds and try again.";
        return;
      }}
      resetCueState();
      createPlayer(vid);
    }});

    startBtn.addEventListener("click", () => {{
      if (!player) return;
      playing = true;
      player.playVideo();
      if (!tickHandle) tickHandle = setInterval(tick, 16);
    }});

    pauseBtn.addEventListener("click", () => {{
      if (!player) return;
      playing = false;
      player.pauseVideo();
    }});

    resetBtn.addEventListener("click", () => {{
      if (!player) return;
      playing = false;
      player.pauseVideo();
      player.seekTo(0, true);
      resetCueState();
      statusEl.textContent = `Reset to 0. Press Start. (JSON first event = ${{FIRST_T_MS}}ms)`;
    }});
  </script>
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_HTML.resolve()} using {JSON_PATH.resolve()} ({len(events)} events).")
    print(f"Loaded first t_ms={t_min} ms, last t_ms={t_max} ms")
    print("Open preview.html in a browser (hard refresh if needed).")

if __name__ == "__main__":
    main()