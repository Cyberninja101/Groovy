#!/usr/bin/env python3
import curses
import json
import time
from pathlib import Path

# kick=a (1), snare=s (3), hihat=k (5), crash=l (6)
KEY_TO_DRUM = {
    "a": {"id": 1, "label": "kick"},
    "s": {"id": 3, "label": "snare"},
    "k": {"id": 5, "label": "hihat"},
    "l": {"id": 6, "label": "crash"},
}

def run(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    events = []
    running = False
    t0 = None
    title = "brandy_clip"
    out_path = Path(f"{title}.json")

    def now_ms():
        return int((time.monotonic() - t0) * 1000)

    def safe_addstr(y, x, s):
        """Write within screen bounds; truncate if needed."""
        h, w = stdscr.getmaxyx()
        if y < 0 or y >= h:
            return
        if x < 0 or x >= w:
            return
        # leave 1 col for safety
        max_len = max(0, w - x - 1)
        if max_len <= 0:
            return
        stdscr.addstr(y, x, s[:max_len])

    def draw(status_line, last_line=""):
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        lines = [
            "Beat charting (no installs) — controls:",
            "",
            "  space: start/stop timer",
            "  a: kick(1)   s: snare(3)   k: hihat(5)   l: crash(6)",
            "  u: undo last event",
            "  q: save + quit",
            "",
            f"  Output: {out_path}",
            "",
            status_line,
            "",
            last_line if last_line else "",
            "",
            f"Events: {len(events)}",
            "",
        ]

        # Render header lines
        y = 0
        for line in lines:
            if y >= h:
                break
            safe_addstr(y, 0, line)
            y += 1

        # Render recent events in remaining space
        remaining = h - y
        if remaining > 0:
            # Show as many as fit
            n_show = min(len(events), remaining)
            start = len(events) - n_show
            for e in events[start:]:
                safe_addstr(y, 0, f"  {e['t_ms']:>6} ms   {e['label']} ({e['drum']})")
                y += 1
                if y >= h:
                    break

        stdscr.refresh()

    last_action = ""

    while True:
        if running:
            status = f"STATUS: RUNNING  |  t = {now_ms()} ms   (space to stop)"
        else:
            status = "STATUS: STOPPED  |  (space to start)  |  q to save+quit"

        draw(status, last_action)

        ch = stdscr.getch()
        if ch == -1:
            time.sleep(0.005)
            continue

        if ch in (ord("q"), ord("Q")):
            break

        if ch == ord(" "):
            if not running:
                t0 = time.monotonic()
                running = True
                last_action = "STARTED timer."
            else:
                running = False
                last_action = "STOPPED timer. (Press space to restart from 0, or q to save+quit)"
            continue

        if ch in (ord("u"), ord("U")):
            if events:
                removed = events.pop()
                last_action = f"UNDO: removed {removed['label']} @ {removed['t_ms']} ms"
            else:
                last_action = "UNDO: nothing to remove."
            continue

        if not running or t0 is None:
            continue

        try:
            key = chr(ch).lower()
        except Exception:
            continue

        if key in KEY_TO_DRUM:
            drum = KEY_TO_DRUM[key]
            e = {"t_ms": now_ms(), "drum": drum["id"], "label": drum["label"]}
            events.append(e)
            last_action = f"HIT: {e['label']} ({e['drum']}) @ {e['t_ms']} ms"

    # Save JSON
    events_sorted = sorted(events, key=lambda x: x["t_ms"])
    data = {
        "title": title,
        "events": [{"t_ms": e["t_ms"], "drum": e["drum"]} for e in events_sorted],
    }
    out_path.write_text(json.dumps(data, indent=2))

    stdscr.erase()
    safe_addstr(0, 0, f"Saved {len(events_sorted)} events to {out_path.resolve()}")
    safe_addstr(2, 0, "Done.")
    stdscr.refresh()
    time.sleep(1.0)

def main():
    curses.wrapper(run)

if __name__ == "__main__":
    main()