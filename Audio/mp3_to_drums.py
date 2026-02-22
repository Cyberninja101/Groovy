#!/usr/bin/env python3
"""
Option B: MP3/WAV -> timestamped drum events (kick/snare/hat/crash)
Rule-based onset detection + band-energy + sustain heuristics.

Outputs:
{
  "title": "...",
  "events": [{"t_ms": 123, "drum": 1}, ...]
}

Drum IDs:
Kick  = 1
Snare = 3
HiHat = 5
Crash = 6
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# librosa is the main dependency for onset + spectral features
import librosa


# ----------------------------
# Config
# ----------------------------

@dataclasses.dataclass
class DrumChartConfig:
    # Audio
    sr: int = 22050          # resample target
    mono: bool = True

    # STFT / feature extraction
    n_fft: int = 2048
    hop_length: int = 256
    win_length: int = 2048

    # Onset detection (tune these first)
    onset_backtrack: bool = True
    onset_delta: float = 0.12 # 0.20    # higher => fewer onsets
    onset_wait: int = 1 #3          # in frames, prevents rapid double-triggers

    # Windowing around onset (seconds)
    # "attack" window for classification
    attack_window_s: float = 0.12 # was 0.09
    # "tail" window for crash vs hat
    tail_window_s: float = 0.180

    # Band definitions (Hz)
    kick_max_hz: float = 260.0 # was 160.0
    snare_low_hz: float = 180.0
    snare_high_hz: float = 2500.0
    high_low_hz: float = 3200.0
    high_high_hz: float = 11000.0

    # Gates / thresholds (these are your main knobs)
    rms_gate: float = 0.010 # 0.015          # ignore quiet onsets
    kick_ratio_th: float = 0.22 # was 0.28      # lowband/total
    high_ratio_th: float = 0.18     # highband/total (for cymbals)
    snare_mid_ratio_th: float = 0.30 # midband/total
    crash_tail_ratio_th: float = 1.30 # was 0.65 # tail_energy / attack_energy (highband)
    crash_high_abs_th: float = 1.5e5    # you'll tune this once

    # Debounce per class (ms)
    debounce_kick_ms: int = 70
    debounce_snare_ms: int = 25
    debounce_hihat_ms: int = 40
    debounce_crash_ms: int = 200

    # Allow multiple hits at same onset (kick+hat etc.)
    allow_multilabel: bool = True

    # Output time adjustment
    offset_ms: int = 0

    # Optional: cap hi-hat density (events/sec)
    # Set to None to disable; e.g. 12 means max 12 hat hits per second (rough)
    hihat_rate_limit_hz: Optional[float] = 12.0


# ----------------------------
# Audio loading
# ----------------------------

class AudioLoader:
    """
    Loads audio from MP3/WAV.
    If ffmpeg is available, it’s the most reliable MP3 decode path.
    """

    def __init__(self, sr: int, mono: bool = True, prefer_ffmpeg: bool = True):
        self.sr = sr
        self.mono = mono
        self.prefer_ffmpeg = prefer_ffmpeg

    def load(self, path: Path) -> Tuple[np.ndarray, int]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        if self.prefer_ffmpeg and self._has_ffmpeg() and path.suffix.lower() in {".mp3", ".m4a", ".aac"}:
            return self._load_via_ffmpeg(path)
        else:
            # librosa can decode many formats, but MP3 support depends on your environment
            y, sr = librosa.load(str(path), sr=self.sr, mono=self.mono)
            return y, sr

    @staticmethod
    def _has_ffmpeg() -> bool:
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception:
            return False

    def _load_via_ffmpeg(self, path: Path) -> Tuple[np.ndarray, int]:
        # Decode to WAV in a temp file, then load reliably.
        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "decoded.wav"
            cmd = [
                "ffmpeg", "-y",
                "-i", str(path),
                "-ac", "1" if self.mono else "2",
                "-ar", str(self.sr),
                str(wav_path),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            y, sr = librosa.load(str(wav_path), sr=self.sr, mono=self.mono)
            return y, sr


# ----------------------------
# Feature helpers
# ----------------------------

class SpectralHelper:
    def __init__(self, sr: int, n_fft: int, hop_length: int, win_length: int):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    def stft_mag(self, y: np.ndarray) -> np.ndarray:
        S = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        return np.abs(S)

    def band_energy(self, mag_spec: np.ndarray, f_lo: float, f_hi: float) -> float:
        idx = np.where((self.freqs >= f_lo) & (self.freqs < f_hi))[0]
        if idx.size == 0:
            return 0.0
        return float(np.sum(mag_spec[idx, :]))

    def band_energy_frame(self, mag_frame: np.ndarray, f_lo: float, f_hi: float) -> float:
        idx = np.where((self.freqs >= f_lo) & (self.freqs < f_hi))[0]
        if idx.size == 0:
            return 0.0
        return float(np.sum(mag_frame[idx]))

    def spectral_centroid_frame(self, mag_frame: np.ndarray) -> float:
        # centroid = sum(f * mag) / sum(mag)
        denom = float(np.sum(mag_frame)) + 1e-9
        return float(np.sum(self.freqs * mag_frame) / denom)


# ----------------------------
# Onset detection
# ----------------------------

class OnsetDetector:
    def __init__(self, cfg: DrumChartConfig):
        self.cfg = cfg

    def detect_onsets_s(self, y: np.ndarray, sr: int) -> np.ndarray:
        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            hop_length=self.cfg.hop_length,
            backtrack=self.cfg.onset_backtrack,
            delta=self.cfg.onset_delta,
            wait=self.cfg.onset_wait,
            units="frames",
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.cfg.hop_length)
        return onset_times


# ----------------------------
# Drum classification
# ----------------------------

class DrumClassifier:
    """
    Rule-based classifier for {kick, snare, hihat, crash}.
    Supports multi-label output at a single onset time.
    """

    KICK = 1
    SNARE = 3
    HIHAT = 5
    CRASH = 6

    def __init__(self, cfg: DrumChartConfig, spec: SpectralHelper):
        self.cfg = cfg
        self.spec = spec

    def classify_onset(self, y: np.ndarray, sr: int, t_s: float) -> List[int]:
        # Extract windows around onset
        attack = self._slice(y, sr, t_s, self.cfg.attack_window_s)
        tail   = self._slice(y, sr, t_s + self.cfg.attack_window_s, self.cfg.tail_window_s)

        # Basic loudness gate
        if self._rms(attack) < self.cfg.rms_gate:
            return []

        # Compute 1-frame-ish spectral from attack (use STFT on short window)
        mag_attack = self.spec.stft_mag(attack)
        mag_tail   = self.spec.stft_mag(tail) if tail.size > 0 else None

        # Sum over time inside window to get stable energy
        attack_sum = np.sum(mag_attack, axis=1)  # (freq,)
        total = float(np.sum(attack_sum)) + 1e-9

        low  = self._band_sum(attack_sum, 0.0, self.cfg.kick_max_hz)
        mid  = self._band_sum(attack_sum, self.cfg.snare_low_hz, self.cfg.snare_high_hz)
        high = self._band_sum(attack_sum, self.cfg.high_low_hz, self.cfg.high_high_hz)

        low_ratio  = low / total
        mid_ratio  = mid / total
        high_ratio = high / total

        centroid = self.spec.spectral_centroid_frame(attack_sum)

        # Crash tail heuristic: compare high-band energy in tail vs attack
        crash_tail_ratio = 0.0
        high_attack = 0.0
        if mag_tail is not None and mag_tail.size > 0:
            tail_sum = np.sum(mag_tail, axis=1)
            high_attack = self._band_sum(attack_sum, self.cfg.high_low_hz, self.cfg.high_high_hz) + 1e-9
            high_tail   = self._band_sum(tail_sum,   self.cfg.high_low_hz, self.cfg.high_high_hz)
            crash_tail_ratio = high_tail / high_attack

        # --- Decision logic ---
        hits: List[int] = []

        # Kick: strong low ratio
        if low_ratio > self.cfg.kick_ratio_th:
            hits.append(self.KICK)
            if not self.cfg.allow_multilabel:
                return hits

        # Cymbal-ish: strong high ratio
        if high_ratio > self.cfg.high_ratio_th:
            # Crash: sustained high tail OR very high centroid (splashy)
            if (crash_tail_ratio > self.cfg.crash_tail_ratio_th) and (high_attack > self.cfg.crash_high_abs_th):
                hits.append(self.CRASH)
            else:
                hits.append(self.HIHAT)

            if not self.cfg.allow_multilabel:
                return hits

        # Snare: mid-band burst + not kick + not purely high cymbal
        # If we already tagged cymbal and kick, snare can still happen in multilabel mode
        if mid_ratio > self.cfg.snare_mid_ratio_th:
            hits.append(self.SNARE)

        # If nothing matched, fall back to snare-ish if it’s loud and not kick-heavy
        if not hits and mid_ratio > 0.20 and low_ratio < 0.35:
            hits.append(self.SNARE)

        return self._dedupe_preserve_order(hits)

    def _band_sum(self, attack_sum: np.ndarray, f_lo: float, f_hi: float) -> float:
        idx = np.where((self.spec.freqs >= f_lo) & (self.spec.freqs < f_hi))[0]
        if idx.size == 0:
            return 0.0
        return float(np.sum(attack_sum[idx]))

    @staticmethod
    def _slice(y: np.ndarray, sr: int, t0: float, dur: float) -> np.ndarray:
        i0 = int(max(0, round(t0 * sr)))
        i1 = int(min(len(y), round((t0 + dur) * sr)))
        if i1 <= i0:
            return np.zeros((0,), dtype=np.float32)
        return y[i0:i1]

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    @staticmethod
    def _dedupe_preserve_order(xs: List[int]) -> List[int]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out


# ----------------------------
# Post-processing
# ----------------------------

class EventPostProcessor:
    def __init__(self, cfg: DrumChartConfig):
        self.cfg = cfg
        self.debounce_ms = {
            1: cfg.debounce_kick_ms,
            3: cfg.debounce_snare_ms,
            5: cfg.debounce_hihat_ms,
            6: cfg.debounce_crash_ms,
        }

    def process(self, events: List[Dict]) -> List[Dict]:
        events = sorted(events, key=lambda e: (e["t_ms"], e["drum"]))
        events = self._drop_negative(events)
        events = self._debounce_per_drum(events)
        events = self._rate_limit_hihat(events)
        events = sorted(events, key=lambda e: (e["t_ms"], e["drum"]))
        return events

    def _drop_negative(self, events: List[Dict]) -> List[Dict]:
        return [e for e in events if e["t_ms"] >= 0]

    def _debounce_per_drum(self, events: List[Dict]) -> List[Dict]:
        last_t: Dict[int, int] = {}
        out: List[Dict] = []
        for e in events:
            d = int(e["drum"])
            t = int(e["t_ms"])
            db = self.debounce_ms.get(d, 60)
            if d in last_t and (t - last_t[d]) < db:
                continue
            out.append(e)
            last_t[d] = t
        return out

    def _rate_limit_hihat(self, events: List[Dict]) -> List[Dict]:
        if self.cfg.hihat_rate_limit_hz is None:
            return events
        min_dt_ms = int(round(1000.0 / float(self.cfg.hihat_rate_limit_hz)))
        last_hat = None
        out = []
        for e in events:
            if e["drum"] != 5:
                out.append(e)
                continue
            if last_hat is None or (e["t_ms"] - last_hat) >= min_dt_ms:
                out.append(e)
                last_hat = e["t_ms"]
        return out


# ----------------------------
# Orchestrator
# ----------------------------

class DrumChartGenerator:
    def __init__(self, cfg: DrumChartConfig, prefer_ffmpeg: bool = True):
        self.cfg = cfg
        self.loader = AudioLoader(sr=cfg.sr, mono=cfg.mono, prefer_ffmpeg=prefer_ffmpeg)
        self.spec = SpectralHelper(sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length)
        self.onsets = OnsetDetector(cfg)
        self.classifier = DrumClassifier(cfg, self.spec)
        self.post = EventPostProcessor(cfg)

    def generate(self, audio_path: Path, title: Optional[str] = None) -> Dict:
        y, sr = self.loader.load(audio_path)

        # Lightly emphasize percussive component (cheap improvement)
        # HPSS is fast and helps reduce melodic false positives.
        # y_harm, y_perc = librosa.effects.hpss(y)
        # y_use = y_perc
        y_use = y

        onset_times = self.onsets.detect_onsets_s(y_use, sr)

        events: List[Dict] = []
        for t_s in onset_times:
            hits = self.classifier.classify_onset(y_use, sr, float(t_s))
            if not hits:
                continue
            t_ms = int(round(float(t_s) * 1000.0)) + self.cfg.offset_ms
            for d in hits:
                events.append({"t_ms": t_ms, "drum": int(d)})

        events = self.post.process(events)

        out = {
            "title": title if title is not None else audio_path.stem,
            "events": events
        }
        return out


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert MP3/WAV -> drum cue events (kick/snare/hat/crash)")
    p.add_argument("audio", type=str, help="Path to MP3/WAV")
    p.add_argument("-o", "--out", type=str, default=None, help="Output JSON path (default: <audio_stem>.json)")
    p.add_argument("--title", type=str, default=None, help="Song title for JSON")
    p.add_argument("--offset-ms", type=int, default=0, help="Shift all timestamps by this many ms")

    # Tuning knobs
    p.add_argument("--onset-delta", type=float, default=None, help="Onset delta (higher => fewer onsets)")
    p.add_argument("--rms-gate", type=float, default=None, help="Ignore quiet onsets below this RMS")
    p.add_argument("--kick-th", type=float, default=None, help="Kick lowband ratio threshold")
    p.add_argument("--high-th", type=float, default=None, help="Highband ratio threshold (cymbals)")
    p.add_argument("--snare-th", type=float, default=None, help="Midband ratio threshold (snare)")
    p.add_argument("--crash-tail-th", type=float, default=None, help="Crash tail ratio threshold")

    p.add_argument("--no-ffmpeg", action="store_true", help="Do not use ffmpeg decode even if installed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio)

    cfg = DrumChartConfig()
    cfg.offset_ms = int(args.offset_ms)

    # Apply overrides
    if args.onset_delta is not None:
        cfg.onset_delta = float(args.onset_delta)
    if args.rms_gate is not None:
        cfg.rms_gate = float(args.rms_gate)
    if args.kick_th is not None:
        cfg.kick_ratio_th = float(args.kick_th)
    if args.high_th is not None:
        cfg.high_ratio_th = float(args.high_th)
    if args.snare_th is not None:
        cfg.snare_mid_ratio_th = float(args.snare_th)
    if args.crash_tail_th is not None:
        cfg.crash_tail_ratio_th = float(args.crash_tail_th)

    gen = DrumChartGenerator(cfg, prefer_ffmpeg=(not args.no_ffmpeg))
    result = gen.generate(audio_path, title=args.title)
    out_path = Path(args.out) if args.out else (audio_path.parent / "song.json")
    # out_path = Path(args.out) if args.out else audio_path.with_suffix(".drums.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}  (events={len(result['events'])})")


if __name__ == "__main__":
    main()