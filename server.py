import math
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp"
)

import cv2
import numpy as np
import torch
import requests

from PIL import Image
from flask import Flask, Response, abort, jsonify, request, send_from_directory

from torchvision import transforms
from torchvision.models import efficientnet_b0

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent

HOST = "127.0.0.1"
PORT = 5000

VIDEO_PATH = "input.mp4"

SERVER_URL = "http://127.0.0.1:5000/ingest/EMBER"

CNN_EVERY = 4
CNN_THRESHOLD = 0.70

# ── drone position & GPS tracking ───────────────────
# Active drone coordinates and heading (updated by telemetry or GPS override)
drone_state = {
    "lat": 47.6862,
    "lon": -122.5321,
    "hdg": 118.0,
    "override": False,
}
drone_state_lock = threading.Lock()

fire_detections = []        # list of {lat, lon, intensity, dist_m, ts}
fire_lock = threading.Lock()
FIRE_COOLDOWN = 2.0         # seconds between logged points
_last_fire_ts = 0.0

JPEG_QUALITY = 75
MAX_WIDTH = 960


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


yolo_model = YOLO(
    "weights/bestyolo.pt"
)


checkpoint = torch.load(
    "weights/best.pt",
    map_location=DEVICE,
    weights_only=False
)


classes = checkpoint["classes"]
image_size = checkpoint["image_size"]
mean = checkpoint["mean"]
std = checkpoint["std"]


cnn_model = efficientnet_b0()

cnn_model.classifier[1] = torch.nn.Linear(
    cnn_model.classifier[1].in_features,
    len(classes)
)

cnn_model.load_state_dict(
    checkpoint["model_state"]
)

cnn_model.to(DEVICE)
cnn_model.eval()


transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=mean,
        std=std
    )
])


# ── tactical HUD styling for YOLO detections ────────
CLASS_COLORS = {
    "fire": (30, 110, 255),       # Vibrant flame orange (BGR)
    "smoke": (220, 185, 80),      # Tactical amber / gray (BGR)
    "fire_smoke": (20, 70, 255),  # Deep ember crimson (BGR)
}
DEFAULT_COLOR = (240, 170, 40)    # Tactical cyan / alert (BGR)


app = Flask(__name__)


class VideoStream:

    def __init__(self):

        self.condition = threading.Condition()

        self.frame = None
        self.frame_id = 0

        self.running = False

        self.last_cnn_class = "clear"
        self.last_cnn_confidence = 0.0


    def update(self, frame):

        with self.condition:

            self.frame = frame
            self.frame_id += 1

            self.condition.notify_all()


    def read(self, last_id):

        with self.condition:

            self.condition.wait_for(
                lambda:
                    self.frame is not None
                    and self.frame_id != last_id
            )

            return (
                self.frame_id,
                self.frame.copy()
            )


stream = VideoStream()


def draw_tactical_yolo_boxes(frame, results):
    """Renders tactical HUD-style bounding boxes with corner brackets,

    subtle semi-transparent target glow, and sleek label chips.
    """
    if not results or len(results) == 0:
        return frame

    r = results[0]
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return frame

    h_img, w_img = frame.shape[:2]
    overlay = frame.copy()
    has_alpha_tint = False

    det_list = []
    names = getattr(r, "names", {})

    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().item())
        cls_id = int(box.cls[0].cpu().item())
        cls_name = str(names.get(cls_id, cls_id)).lower()

        x1, y1, x2, y2 = xyxy
        x1 = max(0, min(w_img - 1, x1))
        y1 = max(0, min(h_img - 1, y1))
        x2 = max(0, min(w_img - 1, x2))
        y2 = max(0, min(h_img - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        det_list.append((x1, y1, x2, y2, cls_name, conf))

        # 1. Subtle semi-transparent tint over detection area
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        has_alpha_tint = True

    if has_alpha_tint:
        # 12% opacity tint: highlights fire/smoke without obscuring details
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    # 2. Draw crisp bounding frames and corner brackets
    for x1, y1, x2, y2, cls_name, conf in det_list:
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
        bw = x2 - x1
        bh = y2 - y1

        # Thin outer frame (1px, anti-aliased)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)

        # Tactical corner brackets (reticle HUD style)
        c_len = max(6, min(20, int(min(bw, bh) * 0.22)))
        c_thick = 2

        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + c_len, y1), color, c_thick, lineType=cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + c_len), color, c_thick, lineType=cv2.LINE_AA)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - c_len, y1), color, c_thick, lineType=cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + c_len), color, c_thick, lineType=cv2.LINE_AA)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + c_len, y2), color, c_thick, lineType=cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - c_len), color, c_thick, lineType=cv2.LINE_AA)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - c_len, y2), color, c_thick, lineType=cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - c_len), color, c_thick, lineType=cv2.LINE_AA)

        # Subtle center reticle crosshair
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.line(frame, (cx - 3, cy), (cx + 3, cy), color, 1, lineType=cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 3), (cx, cy + 3), color, 1, lineType=cv2.LINE_AA)

        # 3. Tactical Badge Label
        label_text = f"{cls_name.upper()} {int(conf * 100)}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, 1)

        chip_h = th + 8
        chip_w = tw + 20

        # Position chip above the box, or inside top if near frame ceiling
        if y1 - chip_h - 2 >= 0:
            chip_y1 = y1 - chip_h - 2
            chip_y2 = y1 - 2
        else:
            chip_y1 = y1 + 2
            chip_y2 = y1 + chip_h + 2

        chip_x1 = x1
        chip_x2 = min(w_img - 1, x1 + chip_w)

        # Dark glass chip background
        chip_bg = frame[chip_y1:chip_y2, chip_x1:chip_x2].copy()
        if chip_bg.size > 0:
            dark_matte = np.full_like(chip_bg, (16, 22, 19))
            cv2.addWeighted(dark_matte, 0.85, chip_bg, 0.15, 0, frame[chip_y1:chip_y2, chip_x1:chip_x2])

        # Accent border & left stripe
        cv2.rectangle(frame, (chip_x1, chip_y1), (chip_x2, chip_y2), (40, 52, 45), 1, lineType=cv2.LINE_AA)
        cv2.line(frame, (chip_x1, chip_y1), (chip_x1, chip_y2), color, 2, lineType=cv2.LINE_AA)

        # Status glowing dot
        dot_cx = chip_x1 + 8
        dot_cy = chip_y1 + (chip_h // 2)
        cv2.circle(frame, (dot_cx, dot_cy), 2, color, -1, lineType=cv2.LINE_AA)

        # Clean typography
        text_x = dot_cx + 6
        text_y = chip_y1 + th + 4
        cv2.putText(frame, label_text, (text_x, text_y), font, font_scale, (235, 245, 240), 1, lineType=cv2.LINE_AA)

    return frame


def draw_hud_status(frame, cnn_class, cnn_conf):
    """Draw a clean top-left drone HUD status chip for the CNN classifier."""
    is_alert = cnn_class in ["fire", "smoke", "fire_smoke"]
    status_color = (30, 90, 255) if is_alert else (70, 215, 150)  # BGR: Flame Orange or Signal Green

    label = f"AI: {cnn_class.upper()} {int(cnn_conf * 100)}%" if cnn_conf > 0 else "AI: SCANNING"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)

    pad_x = 8
    pad_y = 5
    box_x1 = 10
    box_y1 = 10
    box_x2 = box_x1 + tw + 22
    box_y2 = box_y1 + th + (pad_y * 2)

    # Dark glass background
    sub = frame[box_y1:box_y2, box_x1:box_x2]
    if sub.size > 0:
        dark = np.full_like(sub, (14, 18, 16))
        cv2.addWeighted(dark, 0.82, sub, 0.18, 0, frame[box_y1:box_y2, box_x1:box_x2])

    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (40, 52, 45), 1, lineType=cv2.LINE_AA)
    cv2.line(frame, (box_x1, box_y1), (box_x1, box_y2), status_color, 2, lineType=cv2.LINE_AA)

    # Status dot
    dot_x = box_x1 + 8
    dot_y = box_y1 + (box_y2 - box_y1) // 2
    cv2.circle(frame, (dot_x, dot_y), 3, status_color, -1, lineType=cv2.LINE_AA)

    # Text
    cv2.putText(
        frame,
        label,
        (dot_x + 7, box_y1 + th + pad_y),
        font,
        font_scale,
        (235, 245, 240),
        1,
        lineType=cv2.LINE_AA,
    )


def run_cnn(frame):

    image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(image)

    image = transform(
        image
    ).unsqueeze(0).to(
        DEVICE
    )

    with torch.no_grad():

        output = cnn_model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    class_id = probabilities.argmax(
        dim=1
    ).item()

    confidence = probabilities[
        0,
        class_id
    ].item()

    return (
        classes[class_id],
        confidence
    )


def run_detection():

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        print(
            f"Could not open {VIDEO_PATH}"
        )

        return


    frame_count = 0

    last_class = "clear"
    last_confidence = 0.0


    print(
        f"Detection running on {DEVICE}"
    )


    while True:

        ret, frame = cap.read()

        if not ret:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            continue


        # Preserve native aspect ratio (up to 640px width)
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))

        frame_count += 1

        results = yolo_model(
            frame,
            verbose=False
        )

        # Draw sleek tactical HUD bounding boxes (replacing chunky results[0].plot())
        processed_frame = draw_tactical_yolo_boxes(frame.copy(), results)

        if frame_count % CNN_EVERY == 0:
            last_class, last_confidence = run_cnn(
                frame
            )

        if (
            last_class in [
                "smoke",
                "fire",
                "fire_smoke"
            ]
        ):
            # Record fire GPS coordinates for the satellite heat map (no image overlay)
            _record_fire(last_class, last_confidence)

        # Draw sleek top-left tactical HUD chip (replacing raw text overlay)
        draw_hud_status(processed_frame, last_class, last_confidence)

        stream.update(
            processed_frame
        )


        # Remove the opencv display window part
        # cv2.imshow(
        #     "Detection",
        #     processed_frame
        # )


        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


    cap.release()

    cv2.destroyAllWindows()


def mjpeg():

    last_id = 0

    while True:

        last_id, frame = stream.read(
            last_id
        )


        h, w = frame.shape[:2]

        if w > MAX_WIDTH:

            frame = cv2.resize(
                frame,
                (
                    MAX_WIDTH,
                    int(
                        h * MAX_WIDTH / w
                    )
                )
            )


        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY
            ]
        )


        if not ok:

            continue


        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n\r\n"
            +
            buffer.tobytes()
            +
            b"\r\n"
        )


@app.route("/")
def index():

    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


@app.route("/video")
@app.route("/video/<drone_id>")
def video(drone_id="EMBER"):

    if drone_id and drone_id.upper() != "EMBER":

        abort(404)


    return Response(
        mjpeg(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache"
        }
    )


# ── fire heat-map helpers ───────────────────────────

def _record_fire(cls, confidence):
    """Log a fire detection placed strictly in front of the drone (max 50m)."""
    global _last_fire_ts

    now = time.time()
    if now - _last_fire_ts < FIRE_COOLDOWN:
        return
    _last_fire_ts = now

    # intensity: fire > fire_smoke > smoke
    intensity_map = {
        "fire": 1.0,
        "fire_smoke": 0.75,
        "smoke": 0.45,
    }
    intensity = intensity_map.get(cls, 0.5) * confidence

    with drone_state_lock:
        d_lat = drone_state["lat"]
        d_lon = drone_state["lon"]
        d_hdg = drone_state["hdg"]

    # Target is strictly in front of drone: max 50m (between 12m and 50m)
    dist_m = random.uniform(12.0, 50.0)
    # Forward camera view (+/- 18 degrees around drone heading)
    bearing_deg = (d_hdg + random.uniform(-18.0, 18.0)) % 360.0
    bearing_rad = math.radians(bearing_deg)

    lat_offset = (dist_m * math.cos(bearing_rad)) / 111139.0
    lon_offset = (dist_m * math.sin(bearing_rad)) / (
        111139.0 * max(0.0001, math.cos(math.radians(d_lat)))
    )

    lat = d_lat + lat_offset
    lon = d_lon + lon_offset

    point = {
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "intensity": round(intensity, 3),
        "label": cls,
        "confidence": round(confidence, 3),
        "dist_m": round(dist_m, 1),
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    with fire_lock:
        fire_detections.append(point)
        # keep last 500 points
        if len(fire_detections) > 500:
            del fire_detections[:len(fire_detections) - 500]


@app.route("/api/drone_gps", methods=["GET", "POST"])
def drone_gps_api():
    """Get or update drone GPS coordinates, heading, and override status."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        with drone_state_lock:
            if "lat" in data:
                try:
                    drone_state["lat"] = float(data["lat"])
                except (ValueError, TypeError):
                    pass
            if "lon" in data:
                try:
                    drone_state["lon"] = float(data["lon"])
                except (ValueError, TypeError):
                    pass
            if "hdg" in data:
                try:
                    drone_state["hdg"] = float(data["hdg"])
                except (ValueError, TypeError):
                    pass
            if "override" in data:
                drone_state["override"] = bool(data["override"])
            current = dict(drone_state)
        return jsonify({"status": "ok", "drone_state": current})
    else:
        with drone_state_lock:
            return jsonify(dict(drone_state))


@app.route("/api/heatmap", methods=["GET"])
def heatmap_api():
    """Return fire-detection GPS points for the heat-map layer."""
    with fire_lock:
        data = list(fire_detections)
    return jsonify(data)


@app.route("/api/heatmap/clear", methods=["POST"])
def heatmap_clear_api():
    """Clear recorded fire heat-map points."""
    with fire_lock:
        fire_detections.clear()
    return jsonify({"status": "ok", "cleared": True})


if __name__ == "__main__":

    detection_thread = threading.Thread(
        target=run_detection,
        daemon=True
    )

    detection_thread.start()


    print(
        f"Dashboard: http://{HOST}:{PORT}"
    )

    print(
        f"Device: {DEVICE}"
    )


    app.run(
        host=HOST,
        port=PORT,
        threaded=True
    )