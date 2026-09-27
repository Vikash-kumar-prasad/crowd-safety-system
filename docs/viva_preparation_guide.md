# Major Project Viva & Evaluation Preparation Guide

## Project: AI-Based Smart Crowd Safety & Threat Detection System

This guide prepares you for common, difficult, and examiner-level questions during your college major-project viva and practical demonstration.

---

## 1. High-Level Summary (Elevator Pitch)

> **"Our project is an automated AI-powered video surveillance system designed to monitor crowded public spaces and detect threats early. It uses a 5-stage sequential pipeline: First, custom-trained YOLOv8 models detect heads and weapons in real-time. Second, a spatial grid normalizes crowd counts to estimate density heatmaps. Third, Farneback dense optical flow tracks motion velocity and density changes to detect stampedes or sudden dispersals. Fourth, an anatomical reach filter validates whether detected weapons are actually held by people, reducing false alarms. Finally, an explainable rule-based risk assessment engine evaluates compound threats (from Low to Critical) and triggers automated alerts with frame snapshots to a real-time command-center dashboard."**

---

## 2. Core Technical Questions & Answers

### Q1: Why did you train a head detector instead of a full-body person detector?
**Answer:**
- In dense crowds, severe occlusion occurs where bodies, legs, and torsos are completely blocked from view by other people.
- Heads almost always remain visible from elevated surveillance angles (CCTV perspective).
- SCUT-HEAD dataset annotations focus specifically on heads, allowing accurate counting even when people are tightly packed together.

### Q2: Why use detection-based density estimation rather than a regression density map (like CSRNet or MCNN)?
**Answer:**
- Regression CNNs (like CSRNet) generate continuous density maps using Gaussian kernels, but they require high computational power (GPU) and do not provide individual bounding boxes or positions.
- In our pipeline, we already run YOLOv8 to locate people. Reusing these detections across a spatial grid (4x6) gives us localized crowd metrics and heatmaps for free with zero extra neural network inference cost.
- This design choice enables real-time CPU performance (12-25 FPS).

### Q3: How do you handle false positives in weapon detection?
**Answer:**
- Raw object detection models often mistake everyday objects (wallets, smartphones, dark belts) for weapons.
- We implemented an **anatomical reach heuristic**: A weapon detection is only accepted if its spatial coordinates fall within a plausible human torso/hand zone relative to the nearest detected head (within 2.5x head width horizontally, and from 1x head height above the head down to 3.5x head height below).
- Any candidate detection appearing on the floor, ceiling, or distant background without an associated person is rejected.

### Q4: Why is the Risk Assessment engine rule-based rather than an ML classifier?
**Answer:**
- **Auditability and Explainability**: In safety-critical security operations, false alerts or missed emergencies must be explainable. An operator cannot accept a black-box neural network stating "Risk: 0.87" without knowing *why*.
- Our rule hierarchy explicitly tracks cumulative reasons (e.g. *"Possible altercation in a high-density crowd"*, *"Weapon detected"*).
- The priority tree ensures physical threats (weapons) can never be masked by a calm crowd reading.

### Q5: What is Farneback Optical Flow and why is it used?
**Answer:**
- Farneback is a two-frame dense optical flow algorithm based on polynomial expansion. It calculates motion vectors $(u, v)$ for all pixels across the frame.
- We resize frames to $160\text{px}$ width to compute optical flow in $<5\text{ms}$.
- Average vector magnitude indicates overall crowd velocity. Sudden spikes in velocity indicate panic, running, or stampedes.
- Temporal percentage changes in head count distinguish between calm entry/exit versus sudden dispersal (scattering from an incident).

### Q6: How does the system prevent alert spam during an ongoing situation?
**Answer:**
- **Confirmation streak**: The condition must persist for 3 consecutive frames before an alert is confirmed, preventing single-frame detector flicker from raising false alarms.
- **Cooldown throttling**: After logging an alert for a given risk level, identical alerts are suppressed for 5 seconds.
- **Thread-safe atomic logging**: Alerts are written with file locks to prevent concurrency race conditions between the background pipeline thread and Flask API workers.

---

## 3. System Architecture & Engineering Trade-offs

### Q7: Explain the concurrency architecture of your backend.
**Answer:**
- The Flask server runs a decoupled multi-threaded architecture:
  - **Thread 1 (Pipeline Worker - Daemon)**: Runs a continuous loop capturing video frames, executing YOLOv8 inference, optical flow, density computation, threat checks, risk assessment, and updating global memory buffers protected by `threading.Lock`.
  - **Flask HTTP Workers**: Handle asynchronous client requests (`/api/status`, `/api/alerts`, `/video_feed`).
  - This decoupling guarantees that slow client network connections or multiple dashboard tabs do not slow down or block video inference speed.

### Q8: What happens if the camera disconnects or the user has no webcam?
**Answer:**
- The system includes an automated **Demo Fallback Mode**:
- If `cv2.VideoCapture` fails to open a hardware device, the pipeline gracefully switches to generating synthetic grid monitoring frames with animated scanlines, keeping all dashboard telemetry, API endpoints, and risk modules fully active rather than crashing.
- Setting `DEMO_MODE=true` in `.env` allows full project demonstrations on any laptop.

---

## 4. Key Metrics & Benchmarks to Quote

| Parameter | Value |
|---|---|
| Head Detection Model | YOLOv8n (nano), 3.2M params, ~6 MB |
| Weapon Detection Model | YOLOv8n (v4 iterative training), ~6 MB |
| Grid Dimensions | 4 rows x 6 columns (24 spatial cells) |
| Reference Resolution | 640 x 480 px (scale-invariant density) |
| Pipeline Frame Rate | 12 - 25 FPS (CPU-dependent) |
| Optical Flow Resolution | 160 x 120 px (downscaled for speed) |
| Confirmation Window | 3 consecutive frames |
| Alert Cooldown | 5 seconds per risk tier |

---

## 5. Demonstration Checklist for Presentation Day

1. **Activate Environment**: `venv\Scripts\activate`
2. **Start Backend**: `cd backend && python app.py` (or `python backend/app.py`)
3. **Open Browser**: Navigate to `http://127.0.0.1:5000`
4. **Point Out Dashboard Features**:
   - Live stream with heatmap overlay and bounding boxes
   - Dynamic Risk Status Banner (color changes and pulses on High/Critical)
   - Real-time SVG sparkline tracking risk history over 60 seconds
   - Telemetry cards: Heads count, Density level & score, Motion score, Weapons detected, FPS
   - Scrollable Alert Feed with clickable modal showing snapshot proof
   - System Health indicators showing pipeline status, uptime, models loaded
5. **Demonstrate Threat Triggers**:
   - Hold a sample weapon image or smartphone facing camera to trigger weapon alert
   - Wave rapidly or stand close to someone to demonstrate motion/proximity reaction
