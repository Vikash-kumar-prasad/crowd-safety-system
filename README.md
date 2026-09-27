# AI-Based Smart Crowd Safety and Threat Detection System

A real-time AI-powered surveillance system for smart crowd monitoring, threat detection, and automated safety alerts. Built with YOLOv8 computer vision, optical flow analysis, and a professional command-center dashboard.

---

## Problem Statement

Crowd-related incidents — stampedes, crushes, fights, and weapon threats — continue to cause casualties at public gatherings, concerts, religious events, and transportation hubs worldwide. Traditional CCTV monitoring relies on human operators who cannot watch multiple feeds simultaneously, leading to delayed response times.

## Motivation

This system aims to **automate crowd safety monitoring** by combining multiple AI/computer vision techniques to detect dangerous situations *before* they escalate, giving security personnel actionable early warnings through a real-time dashboard.

## Objectives

1. Detect and count people in video feeds using custom-trained object detection
2. Estimate crowd density across spatial regions of each frame
3. Detect abnormal crowd behavior (stampedes, sudden dispersal, rapid convergence)
4. Identify weapons (guns/knives) in video frames with body-proximity filtering
5. Assess overall risk level through a multi-factor rule-based engine
6. Generate automated alerts with frame snapshots for evidence review
7. Present all information through a professional real-time monitoring dashboard

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VIDEO SOURCE                              │
│              (Webcam / IP Camera / Video File)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    AI DETECTION PIPELINE                          │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Head/Person   │  │ Crowd Density │  │ Behavior Anomaly       │ │
│  │ Detection     │──│ Estimation    │──│ Detection (Optical     │ │
│  │ (YOLOv8)     │  │ (Grid-based)  │  │ Flow + Density Delta)  │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Weapon        │  │ Threat        │  │ Risk Assessment        │ │
│  │ Detection     │──│ Assessment    │──│ (Rule-Based Engine)    │ │
│  │ (YOLOv8)     │  │ (Multi-factor)│  │ Low/Med/High/Critical  │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Alert Engine  │  │ MJPEG Stream  │  │ REST API     │
│ (Snapshots +  │  │ (Live Video)  │  │ (/api/status │
│  JSON Log)    │  │               │  │  /api/alerts)│
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ▼
              ┌──────────────────────┐
              │   REAL-TIME SAFETY   │
              │     DASHBOARD        │
              │  (Command Center UI) │
              └──────────────────────┘
```

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Primary language |
| **YOLOv8** (Ultralytics) | Object detection (heads + weapons) |
| **OpenCV** | Video I/O, optical flow, image processing |
| **Flask** | HTTP server, API, template rendering |
| **Flask-CORS** | Cross-origin resource sharing |
| **NumPy** | Numerical computation |
| **HTML5/CSS3/JS** | Dashboard UI (no framework dependencies) |

---

## AI/ML Components

### 1. Head/Person Detection (`best.pt`)
- **Architecture**: YOLOv8n (nano variant, 3.2M parameters)
- **Training Data**: SCUT-HEAD Dataset (Part A + Part B)
- **Classes**: 1 (`head`)
- **Performance**: Precision: **91.96%** | Recall: **87.41%** | mAP@50: **92.89%** | F1: **89.63%**
- **Confidence Threshold**: 0.30

### 2. Weapon Detection (`best_weapon_v4.pt`)
- **Architecture**: YOLOv8n
- **Training Data**: Multi-weapon dataset (handguns, shotguns, combat knives) across 4 refinement rounds
- **Classes**: 1 (`weapon`)
- **Performance**: Precision: **87.46%** | Recall: **78.59%** | mAP@50: **83.28%** | F1: **82.79%**
- **Confidence Threshold**: 0.48 (calibrated for zero video detection dropouts)
- **Video Detection Rate**: > 99% cumulative temporal recall across video streams

### 3. Behavior Anomaly Detection
- **Method**: Farneback Dense Optical Flow
- **Signals**: Motion magnitude + temporal density delta (rolling window)
- **Outputs**: Normal, Sudden Dispersal, Rapid Convergence, High Motion, Stampede/Panic

### 4. Automated Emergency Dispatch System
- **Engine**: Event-driven broadcast engine generating official APCO-standard Computer-Aided Dispatch (CAD) tickets
- **Broadcast Channels**: Municipal Police CAD, Tactical Quick Response Team (QRT) Radio, and Crowd Safety Marshals
- **Latency**: < 5 ms dispatch transmission with simulated IoT radio chirps and optional Telegram Bot push integration

### 5. Risk Assessment Engine
- **Method**: Multi-factor priority hierarchy (deterministic, explainable, auditable)
- **Levels**: Low → Medium → High → Critical
- **Factors**: Lethal weapons, altercation heuristics, spatial crowd density, sudden panic velocity surges

---

## Computer Vision Components

| Component | Technique | Purpose |
|-----------|-----------|---------|
| **Head Detection** | YOLOv8 inference | Count people, locate positions |
| **Density Heatmap** | Grid-based spatial binning + JET colormap | Visualize crowding hotspots |
| **Optical Flow** | Farneback algorithm (160px downscaled) | Detect motion patterns |
| **Weapon Detection** | YOLOv8 + body-proximity filtering | Identify guns/knives near people |
| **Altercation Heuristic** | Head proximity + motion correlation | Flag possible fights |
| **Frame Annotation** | OpenCV drawing + HUD overlay | Real-time visual feedback |

---

## Features

### Detection Features
- ✅ Real-time head/person detection and counting
- ✅ Grid-based crowd density estimation and classification
- ✅ Density heatmap visualization (blue→red colormap)
- ✅ Optical flow-based motion analysis
- ✅ Sudden dispersal detection (crowd scattering)
- ✅ Rapid convergence detection (crowd rushing in)
- ✅ High motion detection (potential panic/stampede)
- ✅ Weapon detection (guns, pistols, knives)
- ✅ Body-reach proximity filtering for weapon false positive reduction
- ✅ Altercation heuristic (proximity + motion)

### Risk & Alert Features
- ✅ 4-level risk assessment (Low, Medium, High, Critical)
- ✅ Compound reason accumulation (multiple factors shown simultaneously)
- ✅ Multi-frame confirmation before alerting (anti-flicker)
- ✅ Per-level cooldown throttling (prevent alert spam)
- ✅ Frame snapshot capture on alert triggers
- ✅ Persistent alert log (JSON with bounded growth)

### Dashboard Features
- ✅ Live annotated MJPEG video stream
- ✅ Real-time risk status banner with color coding
- ✅ People count, density level, behavior status metrics
- ✅ Weapon detection count
- ✅ Risk trend sparkline chart (last 60 readings)
- ✅ Alert timeline with clickable detail modals
- ✅ Alert snapshot viewer
- ✅ System health panel (pipeline status, camera, models, FPS, uptime)
- ✅ Connection-lost detection and reconnection
- ✅ Demo mode indicator
- ✅ Responsive layout (desktop/tablet/mobile)

### System Features
- ✅ Environment variable configuration
- ✅ Demo mode with synthetic frames (fallback when no camera)
- ✅ Video file support (loops automatically)
- ✅ Pipeline crash recovery (auto-restart on errors)
- ✅ Proper logging throughout
- ✅ Health check API endpoint
- ✅ Snapshot serving API endpoint

---

## Installation

### Prerequisites
- Python 3.10 or later
- pip (Python package manager)
- A webcam (optional — demo mode works without one)

### Setup

```bash
# 1. Clone or download the project
cd crowd-safety-project

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Configure environment variables
copy .env.example .env
# Edit .env as needed
```

### Model Files
The trained model weights (`models/best.pt` and `models/best_weapon_v4.pt`) must be present in the `models/` directory. These are included in the repository but excluded from version control due to size.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_SOURCE` | `0` | Camera index (0, 1, ...) or path to video file |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `5000` | Server port |
| `HEAD_MODEL` | `models/best.pt` | Head detection model path |
| `WEAPON_MODEL` | `models/best_weapon_v4.pt` | Weapon detection model path |
| `HEAD_CONF` | `0.3` | Head detection confidence (0.0-1.0) |
| `WEAPON_CONF` | `0.48` | Weapon detection confidence (0.0-1.0) |
| `DEMO_MODE` | `false` | Enable demo mode with fallback frames |

---

## How to Run

### Quick Start (One-Click Launch on Windows)
Simply double-click:
```bat
run_system.bat
```
This automatically activates the virtual environment, launches the backend pipeline, and opens `http://127.0.0.1:5000` in your default browser.

### Pre-Flight System Diagnostic Suite
To verify that all models, CUDA/CPU backends, and dispatch systems are 100% operational:
```bash
python scripts/verify_system.py
```

### Manual Command-Line Launch
```bash
cd crowd-safety-project
venv\Scripts\activate
python backend/app.py
```
Open `http://127.0.0.1:5000` in your browser.

### Option 2: Demo Mode (no webcam required)
```bash
# Set the environment variable before running
set DEMO_MODE=true
cd backend
python app.py
```

### Option 3: Video File Input
```bash
set VIDEO_SOURCE=path/to/your/video.mp4
cd backend
python app.py
```

### Option 4: Run from project root
```bash
python backend/app.py
```

---

## API Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard UI |
| `GET` | `/video_feed` | Live MJPEG stream |
| `GET` | `/api/status` | Current pipeline status (JSON) |
| `GET` | `/api/alerts` | Recent alerts, newest first (JSON) |
| `GET` | `/api/alerts/snapshots/<filename>` | Alert snapshot image |
| `GET` | `/api/health` | System health check (JSON) |

### Status Response Schema
```json
{
  "heads": 6,
  "density_level": "Medium",
  "density_score": 0.000156,
  "behavior_status": "Normal",
  "behavior_detail": { "motion_score": 1.2, "density_change_pct": null },
  "weapons_detected": 0,
  "risk_level": "Low",
  "risk_reasons": ["No significant risk factors detected"],
  "threat_level": "None",
  "updated_at": "2026-09-25 21:58:36",
  "pipeline_active": true,
  "camera_id": "CAM-01",
  "fps": 12.5,
  "uptime_seconds": 3600,
  "demo_mode": false
}
```

---

## Project Workflow

```
1. Video frames are captured from the configured source
2. YOLOv8 head model detects and counts people in each frame
3. Head positions are mapped to a spatial grid for density analysis
4. Optical flow between frames detects motion anomalies
5. YOLOv8 weapon model scans for guns/knives
6. Weapon detections are filtered by proximity to detected people
7. All signals feed into the risk assessment engine
8. Alerts are generated when risk persists above threshold
9. Annotated frames are streamed via MJPEG to the dashboard
10. Dashboard polls API endpoints for real-time status updates
```

---

## Limitations

1. **Head detection only**: The person detector is trained specifically on head/face views. People with backs turned or heavily occluded may be missed.
2. **Heuristic altercation detection**: The "possible altercation" flag is rule-based (proximity + motion), not a trained fight classifier.
3. **Camera-dependent thresholds**: Density and motion thresholds may need recalibration for different camera angles, distances, and resolutions.
4. **Single camera**: Currently processes one video source at a time.
5. **No persistent database**: Alerts are stored in a JSON file, not a production database.
6. **Development server**: Uses Flask's built-in server, not suitable for high-traffic production deployment.

---

## Future Improvements

- [ ] Train a dedicated fight/violence video classifier (e.g., SlowFast, 3D-CNN)
- [ ] Multi-camera support with camera management UI
- [ ] SQLite/MongoDB for persistent alert storage
- [ ] User authentication and role-based access control
- [ ] Notification system (email, SMS, push notifications)
- [ ] Historical analytics and reporting dashboard
- [ ] GPU acceleration optimization (TensorRT, ONNX)
- [ ] Docker containerization for deployment
- [ ] Integration with existing security camera systems (RTSP/ONVIF)
- [ ] Mobile-responsive progressive web app (PWA)

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/test_full_pipeline.py` | End-to-end webcam test |
| `scripts/test_video_file.py` | Process video file with full pipeline |
| `scripts/test_webcam.py` | Lightweight webcam test (head + density only) |
| `scripts/test_density.py` | Batch test density on images |
| `scripts/test_threat.py` | Batch test weapon detection |
| `scripts/test_batch_images.py` | Full pipeline on image directory |
| `scripts/test_risk.py` | Unit test for risk assessment logic |

---

## Deployment

### Current Setup
The system is designed for local deployment (single machine with webcam or video file).

### Production Considerations
For production deployment, consider:
- Use Gunicorn or uWSGI instead of Flask's dev server
- Deploy behind Nginx for SSL termination
- Use a proper database (PostgreSQL/MongoDB) for alerts
- Implement user authentication
- Use GPU-accelerated inference (CUDA)
- Containerize with Docker
