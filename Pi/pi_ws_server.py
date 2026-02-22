import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import websockets


ROOT_DIR = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT_DIR / "Audio"
LATEST_JSON_PATH = AUDIO_DIR / "latest-from-app.json"
# CV/main.py currently defaults to this path, so keep it updated too.
DEFAULT_SONG_PATH = AUDIO_DIR / "brandy-1350.json"
CV_MAIN_PATH = ROOT_DIR / "CV" / "main.py"
DEFAULT_CV_ARGS = [
    "--auto-start-delay",
    "3",
    "--detect-scale",
    "1.0",
    "--detect-every",
    "1",
]


cv_process: Optional[subprocess.Popen] = None
process_lock = asyncio.Lock()


def extract_beatmap_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(data.get("events"), list):
        return data

    for key in ("beatmap", "song", "payload", "data"):
        candidate = data.get(key)
        if isinstance(candidate, dict) and isinstance(candidate.get("events"), list):
            return candidate

    raise ValueError("No beatmap payload found (expected an object with an 'events' list).")


def stop_cv_process() -> None:
    global cv_process
    if cv_process is None:
        return
    if cv_process.poll() is not None:
        cv_process = None
        return

    print(f"Stopping existing CV process (pid={cv_process.pid})")
    cv_process.terminate()
    try:
        cv_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        cv_process.kill()
        cv_process.wait(timeout=2)
    cv_process = None


def parse_speed(data: Dict[str, Any], beatmap: Dict[str, Any]) -> float:
    raw = data.get("speed", beatmap.get("speed", 1.0))
    try:
        return float(raw)
    except Exception:
        return 1.0


def is_learning_speed(speed: float) -> bool:
    return abs(speed + 1.0) < 1e-9


def build_speed_mode_args(speed: float) -> List[str]:
    # speed == -1 means "train mode" (one note at a time).
    if is_learning_speed(speed):
        return [
            "--mode",
            "train",
            "--stick-track",
            "--stick-debug",
            "--stick-hsv-lower",
            "35,80,80",
            "--stick-hsv-upper",
            "90,255,255",
        ]

    # Normal timeline playback, but still scored.
    safe_speed = speed if speed > 0.0 else 1.0
    return ["--mode", "score", "--stick-track", "--speed", f"{safe_speed:.3f}"]


def start_cv_process(extra_args: Optional[List[str]] = None) -> subprocess.Popen:
    args = [sys.executable, str(CV_MAIN_PATH), *DEFAULT_CV_ARGS]
    if extra_args:
        args.extend(extra_args)

    print("Starting CV process:", " ".join(args))
    proc = subprocess.Popen(args, cwd=str(ROOT_DIR))
    return proc


def save_payload(payload: Dict[str, Any]) -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    with LATEST_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Keep default path in sync so CV/main.py loads the latest app beatmap.
    with DEFAULT_SONG_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


async def handler(ws):
    print("Client connected")
    async for msg in ws:
        try:
            data = json.loads(msg)
            if not isinstance(data, dict):
                raise ValueError("Request must be a JSON object.")

            beatmap = extract_beatmap_payload(data)
            speed = parse_speed(data, beatmap)
            speed_mode_args = build_speed_mode_args(speed)
            cv_args = data.get("cv_args")
            if cv_args is not None and not (
                isinstance(cv_args, list) and all(isinstance(x, str) for x in cv_args)
            ):
                raise ValueError("'cv_args' must be a list of strings when provided.")
            effective_cv_args = list(speed_mode_args)
            if cv_args:
                # Let caller overrides win by appending them last.
                effective_cv_args.extend(cv_args)

            async with process_lock:
                save_payload(beatmap)
                stop_cv_process()
                global cv_process
                cv_process = start_cv_process(effective_cv_args)

            await ws.send(
                json.dumps(
                    {
                        "ok": True,
                        "saved": str(LATEST_JSON_PATH),
                        "song_path": str(DEFAULT_SONG_PATH),
                        "requested_speed": speed,
                        "mode": ("train" if is_learning_speed(speed) else "score"),
                        "cv_args": effective_cv_args,
                        "cv_started": True,
                        "pid": cv_process.pid if cv_process else None,
                    }
                )
            )
        except Exception as e:
            await ws.send(json.dumps({"ok": False, "error": str(e)}))


async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765, max_size=50 * 1024 * 1024):
        print("Listening on ws://0.0.0.0:8765")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
