"""
dashboard_push.py
==================
A drop-in helper for sending OpenCV frames to the Firewatch dashboard's
/ingest endpoint. This file is separate from your detection script — importing
it doesn't require changing anything about how your script works.

To wire your script up, add two lines: one import at the top, and one call
wherever you currently do cv2.imshow(...):

    from dashboard_push import send_frame
    ...
    send_frame("EMBER", frame)          # <-- add this
    cv2.imshow("Detection", frame)      # keep this too if you still want the local window

"EMBER" is the drone callsign — it must match one shown in the dashboard's
fleet list (see index.html), so the frame lands in the right feed panel.

Sends are fire-and-forget: if the dashboard server isn't running, send_frame()
just returns False instead of raising, so it never blocks or crashes your
detection loop. It prints a warning once, not on every dropped frame.
"""
import time

import cv2
import requests

DASHBOARD_URL = "http://127.0.0.1:5000"  # change if server.py runs on another machine
JPEG_QUALITY = 80
MIN_INTERVAL = 0.0  # seconds between sends per drone_id; 0 = send every frame

_last_sent = {}
_warned = False


def send_frame(drone_id, frame, timeout=1.0):
    """POST one BGR frame (a plain OpenCV frame, exactly what cv2.read() gives
    you) to the dashboard. Returns True if it was sent, False otherwise.
    """
    global _warned

    if MIN_INTERVAL > 0:
        now = time.time()
        if now - _last_sent.get(drone_id, 0) < MIN_INTERVAL:
            return False
        _last_sent[drone_id] = now

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return False

    try:
        requests.post(
            f"{DASHBOARD_URL}/ingest/{drone_id}",
            data=buf.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            timeout=timeout,
        )
        _warned = False
        return True
    except requests.RequestException as exc:
        if not _warned:
            print(f"[dashboard_push] couldn't reach dashboard at {DASHBOARD_URL} — {exc}")
            _warned = True
        return False
