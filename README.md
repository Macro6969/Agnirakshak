# EMBER — Aerial Wildfire Detection & Tactical Command System

EMBER is an AI-powered aerial wildfire detection, tracking, and response platform. It combines real-time computer vision inference (YOLOv8 + EfficientNet-B0) with a tactical command dashboard featuring live video streaming, high-resolution satellite imagery, forward geodesic fire localization, and interactive GPS control.

---

## 🌟 Key Features

- **Dual-Stage Deep Learning Pipeline**:
  - **YOLOv8** (`weights/bestyolo.pt`): Fast object detection identifying and framing bounding boxes around `fire`, `smoke`, and `fire_smoke`.
  - **EfficientNet-B0** (`weights/best.pt`): Secondary CNN classification running every few frames to cross-verify detection confidence and reduce false positives.
  - **Tactical HUD Visuals**: Clean, modern corner-bracket bounding boxes with class confidence chips and no intrusive overlays blocking the video feed.

- **Tactical Operator Dashboard** (`index.html`):
  - **Live Video Streaming**: Low-latency MJPEG streaming with White Hot Thermal and Low-Light RGB view modes.
  - **Satellite Heat Map**: Real-time fire and smoke detections plotted onto Esri World Imagery satellite map via Leaflet & Leaflet.heat.
  - **Forward Geodesic Projection ($\le 50$m)**: Accurately projects detected fires directly in front of the drone (within the camera's forward field of view, $12\text{m} \le d \le 50\text{m}$) using geodesic math rather than arbitrary coordinate jitter.
  - **Targeting Sector Visualizer**: Dynamic forward 50m FOV cone on the map reflecting the drone's active heading and detection coverage.
  - **GPS Override & Relocation**:
    - Toggleable **GPS Override** switch.
    - Click anywhere on the satellite map to instantly reposition the drone.
    - Live **Heading Slider (0°–359°)** updating map orientation, HUD compass, and forward fire projection.
    - One-click **"My GPS"** (browser geolocation) and **"Reset"** controls.
  - **Manual Flight & Response Controls**:
    - Virtual flight sticks (Yaw/Throttle & Pitch/Roll).
    - Retardant payload deployment with auto-return sequencing (`Return to Base`).
    - Live flight telemetry (Altitude, Airspeed, Battery, Coordinates).

---

## 📂 Project Structure

```text
.
├── server.py               # Integrated Flask server: video ingestion, ML inference, API endpoints
├── index.html              # Tactical operator web dashboard (Leaflet satellite map, HUD, controls)
├── test.py                 # Standalone script for testing local inference pipeline
├── dashboard_push.py       # Helper module to stream processed frames to the server
├── weights/
│   ├── bestyolo.pt         # Trained YOLOv8 model for fire/smoke detection
│   └── best.pt             # Trained EfficientNet-B0 classifier checkpoint
├── input.mp4               # Sample drone video footage for detection pipeline
├── input11.mp4             # High-resolution benchmark video footage
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **NVIDIA GPU with CUDA support** (recommended for real-time inference, falls back to CPU automatically)

### 1. Installation

Clone this repository and create a virtual environment:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd drone

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install torch torchvision ultralytics opencv-python pillow flask requests numpy
```

*(Or run `pip install -r requirements.txt` if using a frozen requirements file).*

### 2. Verify Weights & Video Files

Ensure the model weights and sample video footage are located in their expected paths:
- `weights/bestyolo.pt`
- `weights/best.pt`
- `input.mp4`

---

## 🎮 Running the Application

### Launching the Integrated Server

Start the integrated Flask server:

```bash
python3 server.py
```

This starts:
1. The background detection thread processing `input.mp4` with YOLOv8 and EfficientNet-B0.
2. The MJPEG video stream broadcaster.
3. The Flask web server on port `5000`.

### Accessing the Dashboard

Open your web browser and navigate to:

```text
http://127.0.0.1:5000
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the tactical command dashboard (`index.html`) |
| `GET` | `/video/<drone_id>` | MJPEG live video feed (supports `?mode=thermal` and `?mode=rgb`) |
| `POST` | `/ingest/<drone_id>` | Endpoint for external ingest of processed video frames |
| `GET` | `/api/drone_gps` | Returns current drone coordinates, heading, and override status |
| `POST` | `/api/drone_gps` | Updates drone coordinates, heading, and GPS override state |
| `GET` | `/api/heatmap` | Returns historical fire detection coordinates for the heat map |
| `POST` | `/api/heatmap/clear` | Clears all recorded heat map points |

---

## 🗺️ How GPS Geodesic Localization Works

When a fire or smoke hazard is identified by the model, its geographic coordinates are mathematically projected relative to the drone's position:

1. **Distance**: Constrained strictly in front of the drone ($12\text{m} \le d \le 50\text{m}$).
2. **Camera FOV Bearing**: $\theta = (\text{heading} \pm 18^\circ) \pmod{360^\circ}$.
3. **Geodesic Offsets**:
   $$\Delta\text{lat} = \frac{d \cdot \cos(\theta)}{111139.0}$$
   $$\Delta\text{lon} = \frac{d \cdot \sin(\theta)}{111139.0 \cdot \cos(\text{lat})}$$
4. **Calculated Fire Point**:
   $$\text{Fire}_{\text{lat}} = \text{Drone}_{\text{lat}} + \Delta\text{lat}, \quad \text{Fire}_{\text{lon}} = \text{Drone}_{\text{lon}} + \Delta\text{lon}$$

---

## 🛡️ License

This project is licensed under the MIT License.

