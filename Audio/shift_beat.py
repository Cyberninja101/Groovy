#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def shift_events(data: dict, offset_ms: int, clamp_zero: bool) -> dict:
    out = dict(data)  # shallow copy
    events = data.get("events", [])
    new_events = []

    for ev in events:
        if "t_ms" not in ev:
            raise ValueError(f"Event missing 't_ms': {ev}")

        t = int(ev["t_ms"]) + offset_ms
        if clamp_zero and t < 0:
            t = 0

        new_ev = dict(ev)
        new_ev["t_ms"] = t
        new_events.append(new_ev)

    # Keep events sorted by time (good practice after shifting)
    new_events.sort(key=lambda e: e["t_ms"])
    out["events"] = new_events

    # Optional: record what you did for debugging/demo provenance
    out.setdefault("meta", {})
    out["meta"]["time_offset_ms"] = offset_ms
    out["meta"]["clamp_zero"] = clamp_zero

    return out

def main():
    ap = argparse.ArgumentParser(description="Shift all t_ms timestamps in a beat JSON by a constant offset.")
    ap.add_argument("infile", type=Path, help="Input JSON file")
    ap.add_argument("outfile", type=Path, help="Output JSON file")
    ap.add_argument("--offset-ms", type=int, required=True, help="Milliseconds to add (negative allowed)")
    ap.add_argument("--clamp-zero", action="store_true", help="Clamp any negative t_ms to 0")
    args = ap.parse_args()

    data = json.loads(args.infile.read_text(encoding="utf-8"))
    shifted = shift_events(data, args.offset_ms, args.clamp_zero)

    args.outfile.write_text(json.dumps(shifted, indent=2), encoding="utf-8")
    print(f"Wrote {args.outfile} with offset {args.offset_ms} ms")

if __name__ == "__main__":
    main()