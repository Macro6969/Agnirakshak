import os
import threading
import time
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
from flask import Flask, Response, abort, request, send_from_directory

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


overlay_img = cv2.imread(
    "overlay.png",
    cv2.IMREAD_UNCHANGED
)

if overlay_img is not None:
    overlay_img = cv2.resize(
        overlay_img,
        (80, 80)
    )


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


def add_overlay(frame):

    if overlay_img is None:
        return frame

    h, w = frame.shape[:2]

    oh, ow = overlay_img.shape[:2]

    x = w - ow - 20
    y = 20

    if x < 0 or y + oh > h:
        return frame

    if overlay_img.shape[2] == 4:

        alpha = (
            overlay_img[:, :, 3] / 255.0
        )

        for c in range(3):

            frame[
                y:y + oh,
                x:x + ow,
                c
            ] = (
                alpha * overlay_img[:, :, c]
                +
                (1 - alpha)
                * frame[
                    y:y + oh,
                    x:x + ow,
                    c
                ]
            )

    else:

        frame[
            y:y + oh,
            x:x + ow
        ] = overlay_img

    return frame


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


        frame = cv2.resize(
            frame,
            (320, 240)
        )

        frame_count += 1


        results = yolo_model(
            frame,
            verbose=False
        )

        processed_frame = results[0].plot()


        if frame_count % CNN_EVERY == 0:

            last_class, last_confidence = run_cnn(
                frame
            )


        if (
            last_class in [
                "fire",
                "fire_smoke"
            ]
            and
            last_confidence >= CNN_THRESHOLD
        ):

            processed_frame = add_overlay(
                processed_frame
            )


        # Remove the opencv display window part
        cv2.putText(
            processed_frame,
            f"{last_class} {last_confidence:.2f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


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


@app.route("/video/<drone_id>")
def video(drone_id):

    if drone_id.upper() != "EMBER":

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