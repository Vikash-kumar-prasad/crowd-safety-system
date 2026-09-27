"""
Flask backend for the AI-Based Smart Crowd Safety & Threat Detection System.

Runs the full AI pipeline (head detection, density, behavior, weapon
detection, risk assessment, alerts) continuously on a video source in a
background thread, and exposes:

    GET  /                           -> monitoring dashboard
    GET  /video_feed                 -> live annotated MJPEG video stream
    GET  /api/status                 -> current pipeline status as JSON
    GET  /api/alerts                 -> alert history as JSON
    GET  /api/alerts/snapshots/<fn>  -> serve alert snapshot images
    GET  /api/health                 -> system health check

Usage:
    cd backend
    python app.py
    # or from project root:
    python backend/app.py

Then open http://127.0.0.1:5000 in a browser.

Environment variables (optional, loaded from .env if present):
    VIDEO_SOURCE   - camera index (0) or path to video file
    HOST           - bind address (default: 0.0.0.0)
    PORT           - port number (default: 5000)
    HEAD_MODEL     - path to head detection model (default: models/best.pt)
    WEAPON_MODEL   - path to weapon detection model (default: models/best_weapon_v4.pt)
    HEAD_CONF      - head detection confidence threshold (default: 0.3)
    WEAPON_CONF    - weapon detection confidence threshold (default: 0.65)
    DEMO_MODE      - set to "true" to enable demo mode with fallback video
"""

import os
import sys
import json
import logging
import threading
import time
import traceback

import base64
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, send_from_directory, request
from flask_cors import CORS

# --- Path setup: resolve project root regardless of CWD ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

# Add modules to path for backward compatibility with scripts
sys.path.insert(0, os.path.join(BACKEND_DIR, "app", "modules"))

from density import CrowdDensityEstimator
from behavior import BehaviorAnomalyDetector
from threat import ThreatDetector
from risk import RiskAssessor
from alert import AlertLogger
from dispatch import EmergencyDispatcher

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crowd-safety")

# --- Configuration from environment ---
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
HEAD_MODEL_PATH = os.getenv("HEAD_MODEL", os.path.join(PROJECT_ROOT, "models", "best.pt"))
WEAPON_MODEL_PATH = os.getenv("WEAPON_MODEL", os.path.join(PROJECT_ROOT, "models", "best_weapon_v4.pt"))
HEAD_CONF = float(os.getenv("HEAD_CONF", "0.3"))
WEAPON_CONF = float(os.getenv("WEAPON_CONF", "0.48"))
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
ALERTS_DIR = os.path.join(PROJECT_ROOT, "notebooks", "alerts")
ALERTS_LOG_PATH = os.path.join(ALERTS_DIR, "alerts.json")
SNAPSHOTS_DIR = os.path.join(ALERTS_DIR, "snapshots")

emergency_dispatcher = EmergencyDispatcher(log_dir=ALERTS_DIR)

# --- Flask app ---
app = Flask(__name__, template_folder="templates")
CORS(app)

# ---- shared state, updated by the background pipeline thread ----
state_lock = threading.Lock()
latest_frame = None
latest_status = {
    "heads": 0,
    "density_level": "Low",
    "density_score": 0.0,
    "behavior_status": "Normal",
    "behavior_detail": {"motion_score": 0.0, "density_change_pct": None},
    "weapons_detected": 0,
    "risk_level": "Low",
    "risk_reasons": [],
    "threat_level": "None",
    "updated_at": None,
    "pipeline_active": False,
    "camera_id": "CAM-01",
    "fps": 0.0,
    "uptime_seconds": 0,
    "demo_mode": DEMO_MODE,
}

pipeline_start_time = None
models_loaded = False
camera_connected = False
head_model = None
weapon_model = None
density_estimator = None
threat_detector = None
risk_assessor = None

RISK_COLORS = {
    "Low": (0, 200, 0),
    "Medium": (0, 200, 255),
    "High": (0, 100, 255),
    "Critical": (0, 0, 255),
}

requested_source = None
requested_camera_id = None
active_source_name = "webcam"
source_switch_lock = threading.Lock()


def _parse_video_source(source_str):
    """Parse video source: integer for camera index, string for file path or network URL."""
    try:
        return int(source_str)
    except ValueError:
        s = str(source_str)
        if s.startswith("rtsp://") or s.startswith("http://") or s.startswith("https://"):
            return s
        path = s
        if not os.path.isabs(path):
            path = os.path.join(PROJECT_ROOT, path)
        return path


def _generate_demo_frame(frame_count):
    """Generate a synthetic demo frame when no camera/video is available."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Dark gradient background
    for i in range(480):
        frame[i, :] = [int(15 + i * 0.02), int(20 + i * 0.03), int(30 + i * 0.04)]

    # Grid pattern
    for x in range(0, 640, 107):
        cv2.line(frame, (x, 0), (x, 480), (40, 50, 60), 1)
    for y in range(0, 480, 120):
        cv2.line(frame, (0, y), (640, y), (40, 50, 60), 1)

    # Title
    cv2.putText(frame, "DEMO MODE", (200, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 120, 140), 2, cv2.LINE_AA)
    cv2.putText(frame, "No camera connected - simulated feed", (120, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 100, 120), 1, cv2.LINE_AA)

    # Animated scan line
    scan_y = (frame_count * 3) % 480
    cv2.line(frame, (0, scan_y), (640, scan_y), (50, 80, 100), 1)

    # Timestamp
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, ts, (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 100, 120), 1, cv2.LINE_AA)

    return frame


def pipeline_worker(source=0):
    """Runs continuously in a background thread, processing frames and
    updating the shared state that the API routes read from."""
    global latest_frame, latest_status, pipeline_start_time, models_loaded, camera_connected, head_model, weapon_model, density_estimator, threat_detector, risk_assessor

    pipeline_start_time = time.time()

    # --- Load models ---
    try:
        from ultralytics import YOLO
        logger.info(f"Loading head model: {HEAD_MODEL_PATH}")
        head_model = YOLO(HEAD_MODEL_PATH)
        logger.info(f"Loading weapon model: {WEAPON_MODEL_PATH}")
        weapon_model = YOLO(WEAPON_MODEL_PATH)
        models_loaded = True
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        models_loaded = False
        # Even without models, keep pipeline alive in demo mode
        if not DEMO_MODE:
            with state_lock:
                latest_status["pipeline_active"] = False
            return

    density_estimator = CrowdDensityEstimator(grid_rows=4, grid_cols=6)
    behavior_detector = BehaviorAnomalyDetector()
    threat_detector = ThreatDetector(weapon_model, weapon_conf=WEAPON_CONF) if models_loaded else None
    risk_assessor = RiskAssessor()
    alert_logger = AlertLogger(
        log_path=ALERTS_LOG_PATH,
        snapshot_dir=SNAPSHOTS_DIR,
    )

    # --- Open video source ---
    parsed_source = _parse_video_source(str(source))
    cap = cv2.VideoCapture(parsed_source)
    is_file = isinstance(parsed_source, str)

    if not cap.isOpened():
        logger.warning(f"Could not open video source: {parsed_source}")
        if DEMO_MODE:
            logger.info("Running in DEMO MODE with synthetic frames.")
            camera_connected = False
        else:
            logger.error("No video source available and DEMO_MODE is off. Pipeline inactive.")
            with state_lock:
                latest_status["pipeline_active"] = False
            return
    else:
        camera_connected = True
        logger.info(f"Video source opened: {parsed_source}")

    with state_lock:
        latest_status["pipeline_active"] = True

    logger.info("Pipeline worker started.")
    frame_count = 0
    fps_timer = time.time()
    fps_frame_count = 0

    while True:
        try:
            # Check for dynamic video source switch request
            global requested_source, requested_camera_id, active_source_name
            with source_switch_lock:
                if requested_source is not None:
                    new_src = requested_source
                    new_cam_id = requested_camera_id
                    requested_source = None
                    requested_camera_id = None
                    logger.info(f"Switching video source to: {new_src}")
                    if cap is not None:
                        cap.release()
                    parsed_source = _parse_video_source(str(new_src))
                    cap = cv2.VideoCapture(parsed_source)
                    is_file = isinstance(parsed_source, str) and not (str(parsed_source).startswith("rtsp://") or str(parsed_source).startswith("http://"))
                    if cap.isOpened():
                        camera_connected = True
                        logger.info(f"Video source switched successfully: {parsed_source}")
                    else:
                        logger.warning(f"Could not open new video source: {parsed_source}")
                        camera_connected = False
                    # Reset optical flow history to avoid artificial motion spike
                    behavior_detector.prev_gray = None
                    behavior_detector.density_history.clear()
                    with state_lock:
                        if new_cam_id:
                            latest_status["camera_id"] = new_cam_id
                        latest_status["camera_connected"] = camera_connected

            # --- Read frame ---
            if camera_connected:
                ret, frame = cap.read()
                if not ret:
                    if is_file:
                        # Video file ended, loop back
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        logger.warning("Failed to read frame, retrying...")
                        time.sleep(0.5)
                        continue
                if is_file:
                    time.sleep(0.035)  # Pace playback to ~25-28 FPS natural speed
            else:
                # Demo mode: generate synthetic frame
                frame = _generate_demo_frame(frame_count)
                time.sleep(1.0 / 15)  # Simulate 15 FPS

            frame_count += 1
            h, w = frame.shape[:2]

            # --- Run AI pipeline ---
            if models_loaded:
                head_results = head_model.predict(frame, conf=HEAD_CONF, verbose=False)
                head_boxes = [tuple(b[:4]) for b in head_results[0].boxes.xyxy.cpu().numpy()]
            else:
                head_boxes = []
                head_results = None

            density_grid, normalized_grid = density_estimator.compute_density_grid(head_boxes, w, h)
            density_level, density_score = density_estimator.classify_density(density_grid)

            behavior_result = behavior_detector.update(frame, density_grid)

            if threat_detector is not None:
                threat_result = threat_detector.assess(frame, head_boxes, behavior_result["motion_score"])
            else:
                threat_result = {
                    "weapons": [], "weapon_detected": False,
                    "possible_altercation": False, "threat_level": "None",
                }

            risk_result = risk_assessor.assess(density_level, behavior_result["status"], threat_result)

            alert_rec = alert_logger.log_if_needed(
                frame, risk_result,
                extra_info={
                    "heads": len(head_boxes),
                    "density_level": density_level,
                    "behavior_status": behavior_result["status"],
                }
            )

            if alert_rec and alert_rec.get("risk_level") == "Critical":
                snap_path = os.path.join(SNAPSHOTS_DIR, alert_rec["snapshot_filename"]) if alert_rec.get("snapshot_filename") else None
                emergency_dispatcher.dispatch(alert_rec, snapshot_path=snap_path, camera_id=latest_status.get("camera_id", "CAM-01"))

            # ---- draw annotated frame ----
            if head_results is not None:
                annotated = head_results[0].plot()
            else:
                annotated = frame.copy()

            annotated = density_estimator.draw_heatmap_overlay(annotated, normalized_grid)
            annotated = density_estimator.draw_grid_lines(annotated, heads_count=len(head_boxes))

            for (x1, y1, x2, y2, conf) in threat_result["weapons"]:
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(annotated, f"WEAPON {conf:.2f}", (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            risk_color = RISK_COLORS.get(risk_result["risk_level"], (255, 255, 255))
            cv2.rectangle(annotated, (0, 0), (w, 40), risk_color, -1)
            risk_text = f"RISK: {risk_result['risk_level']} - {'; '.join(risk_result['reasons'])}"
            cv2.putText(annotated, risk_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)

            status_text = (f"Heads: {len(head_boxes)}  Density: {density_level} ({density_score:.5f})  "
                           f"Behavior: {behavior_result['status']}  Weapons: {len(threat_result['weapons'])}")
            cv2.putText(annotated, status_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # --- Calculate FPS ---
            fps_frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 2.0:
                current_fps = fps_frame_count / elapsed
                fps_timer = time.time()
                fps_frame_count = 0
            else:
                current_fps = latest_status.get("fps", 0.0)

            # --- Update shared state ---
            with state_lock:
                latest_frame = annotated
                latest_status = {
                    "heads": int(len(head_boxes)),
                    "density_level": str(density_level),
                    "density_score": float(round(float(density_score), 6)),
                    "behavior_status": str(behavior_result["status"]),
                    "behavior_detail": {
                        "motion_score": float(round(float(behavior_result["motion_score"]), 2)),
                        "density_change_pct": float(round(float(behavior_result["density_change_pct"]), 4)) if behavior_result["density_change_pct"] is not None else None,
                    },
                    "weapons_detected": int(len(threat_result["weapons"])),
                    "risk_level": str(risk_result["risk_level"]),
                    "risk_reasons": [str(r) for r in risk_result["reasons"]],
                    "threat_level": str(threat_result["threat_level"]),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "pipeline_active": True,
                    "camera_id": "CAM-01",
                    "fps": float(round(float(current_fps), 1)),
                    "uptime_seconds": int(time.time() - pipeline_start_time),
                    "demo_mode": bool(DEMO_MODE and not camera_connected),
                }

            # Small sleep to prevent 100% CPU when processing is fast
            time.sleep(0.01)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            logger.debug(traceback.format_exc())
            time.sleep(1.0)  # Back off on errors, then retry
            continue

    # Cleanup (unreachable in normal operation, but good practice)
    if cap is not None:
        cap.release()


def generate_mjpeg():
    """Yields JPEG-encoded frames for the /video_feed MJPEG stream."""
    while True:
        with state_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.1)
            continue

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

        # Throttle to ~20 FPS max for the stream to save bandwidth
        time.sleep(0.05)


# ===================== ROUTES =====================

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify(latest_status)


@app.route("/api/alerts")
def api_alerts():
    try:
        if not os.path.exists(ALERTS_LOG_PATH):
            return jsonify([])
        with open(ALERTS_LOG_PATH, "r") as f:
            alerts = json.load(f)
        # newest first, limit to last 50
        return jsonify(list(reversed(alerts))[:50])
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Error reading alerts: {e}")
        return jsonify([])


@app.route("/api/alerts/snapshots/<filename>")
def serve_snapshot(filename):
    """Serve alert snapshot images."""
    # Security: only serve .jpg files, no path traversal
    if not filename.endswith(".jpg") or "/" in filename or "\\" in filename:
        return jsonify({"error": "Invalid filename"}), 400
    return send_from_directory(SNAPSHOTS_DIR, filename)


@app.route("/api/health")
def api_health():
    """System health check endpoint."""
    uptime = int(time.time() - pipeline_start_time) if pipeline_start_time else 0
    return jsonify({
        "status": "healthy" if latest_status.get("pipeline_active") else "degraded",
        "pipeline_active": latest_status.get("pipeline_active", False),
        "uptime_seconds": uptime,
        "models_loaded": models_loaded,
        "camera_connected": camera_connected,
        "demo_mode": DEMO_MODE and not camera_connected,
    })


@app.route("/api/analyze_image", methods=["POST"])
def api_analyze_image():
    """Analyze an uploaded static image through the complete detection pipeline."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image file selected"}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    h, w = frame.shape[:2]

    # Ensure models are loaded
    global head_model, weapon_model, threat_detector, models_loaded
    if head_model is None or threat_detector is None:
        try:
            from ultralytics import YOLO
            if head_model is None and os.path.exists(HEAD_MODEL_PATH):
                head_model = YOLO(HEAD_MODEL_PATH)
            if weapon_model is None and os.path.exists(WEAPON_MODEL_PATH):
                weapon_model = YOLO(WEAPON_MODEL_PATH)
            if threat_detector is None and weapon_model is not None:
                threat_detector = ThreatDetector(weapon_model, weapon_conf=WEAPON_CONF)
            models_loaded = True
        except Exception as e:
            logger.error(f"Lazy model loading error for image analysis: {e}")

    # Run Head Detection
    if head_model is not None:
        head_results = head_model.predict(frame, conf=HEAD_CONF, verbose=False)
        head_boxes = [tuple(b[:4]) for b in head_results[0].boxes.xyxy.cpu().numpy()]
    else:
        head_results = None
        head_boxes = []

    # Density Estimation
    de = density_estimator if density_estimator is not None else CrowdDensityEstimator(grid_rows=4, grid_cols=6)
    density_grid, normalized_grid = de.compute_density_grid(head_boxes, w, h)
    density_level, density_score = de.classify_density(density_grid)

    # Threat Assessment (use 0.50 threshold for static image analysis to reliably detect test guns)
    analysis_threat_detector = ThreatDetector(weapon_model, weapon_conf=0.50) if weapon_model is not None else None
    if analysis_threat_detector is not None:
        threat_result = analysis_threat_detector.assess(frame, head_boxes, motion_score=0.0)
    elif threat_detector is not None:
        threat_result = threat_detector.assess(frame, head_boxes, motion_score=0.0)
    else:
        threat_result = {
            "weapons": [], "weapon_detected": False,
            "possible_altercation": False, "threat_level": "None",
        }

    # Risk Assessment
    ra = risk_assessor if risk_assessor is not None else RiskAssessor()
    risk_result = ra.assess(density_level, "Normal", threat_result)

    # Render annotated frame
    if head_results is not None:
        annotated = head_results[0].plot()
    else:
        annotated = frame.copy()

    annotated = de.draw_heatmap_overlay(annotated, normalized_grid)
    annotated = de.draw_grid_lines(annotated, heads_count=len(head_boxes))

    for (x1, y1, x2, y2, conf) in threat_result["weapons"]:
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(annotated, f"WEAPON {conf:.2f}", (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    risk_color = RISK_COLORS.get(risk_result["risk_level"], (255, 255, 255))
    cv2.rectangle(annotated, (0, 0), (w, 40), risk_color, -1)
    risk_text = f"RISK: {risk_result['risk_level']} - {'; '.join(risk_result['reasons'])}"
    cv2.putText(annotated, risk_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)

    status_text = (f"Heads: {len(head_boxes)}  Density: {density_level} ({density_score:.5f})  "
                   f"Weapons: {len(threat_result['weapons'])}")
    cv2.putText(annotated, status_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Encode to JPEG Base64
    ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return jsonify({"error": "Failed to encode result image"}), 500

    img_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")

    # If critical threat OR high crowd density detected, trigger emergency dispatch
    dispatch_info = None
    safe_time = time.strftime("%Y%m%d_%H%M%S")
    if risk_result["risk_level"] == "Critical":
        snap_fn = f"upload_critical_{safe_time}.jpg"
        snap_path = os.path.join(SNAPSHOTS_DIR, snap_fn)
        try:
            cv2.imwrite(snap_path, annotated)
        except Exception:
            snap_path = None
            snap_fn = None

        dispatch_info = emergency_dispatcher.dispatch(
            {
                "reasons": risk_result["reasons"],
                "heads": len(head_boxes),
                "severity": "CRITICAL EMERGENCY",
                "snapshot_filename": snap_fn
            },
            snapshot_path=snap_path,
            camera_id="MANUAL-INSPECTION (Uploaded Static Image)"
        )
    elif density_level == "High":
        snap_fn = f"upload_density_{safe_time}.jpg"
        snap_path = os.path.join(SNAPSHOTS_DIR, snap_fn)
        try:
            cv2.imwrite(snap_path, annotated)
        except Exception:
            snap_path = None
            snap_fn = None

        dispatch_info = emergency_dispatcher.dispatch(
            {
                "reasons": [f"High Crowd Density Bottleneck ({len(head_boxes)} people, Score: {density_score:.5f})"],
                "heads": len(head_boxes),
                "severity": "HIGH DENSITY ADVISORY",
                "snapshot_filename": snap_fn,
                "recipients": [
                    {"unit": "Campus Crowd Safety Marshals", "channel": "Public Address & Gate Control", "status": "ACK 200 OK"},
                    {"unit": "Central Security Operations Center (SOC)", "channel": "Density Hotspot Monitor", "status": "ACK 200 OK"},
                ]
            },
            snapshot_path=snap_path,
            camera_id="MANUAL-INSPECTION (Uploaded Static Image)"
        )

    return jsonify({
        "success": True,
        "heads": int(len(head_boxes)),
        "density_level": str(density_level),
        "density_score": float(round(float(density_score), 6)),
        "weapons_detected": int(len(threat_result["weapons"])),
        "threat_level": str(threat_result["threat_level"]),
        "risk_level": str(risk_result["risk_level"]),
        "risk_reasons": [str(r) for r in risk_result["reasons"]],
        "annotated_image": img_b64,
        "dispatch": dispatch_info,
    })


@app.route("/api/dispatches")
def get_dispatches():
    """Return list of recent emergency broadcasts."""
    return jsonify(emergency_dispatcher.get_recent_dispatches())


@app.route("/api/dispatches/test", methods=["POST"])
def trigger_test_dispatch():
    """Trigger a simulated tactical emergency broadcast drill."""
    disp = emergency_dispatcher.dispatch(
        {
            "reasons": ["Manual Emergency Response Drill (Operator Override)"],
            "heads": 2,
            "snapshot_filename": None
        },
        camera_id="CAM-01 (Main Gate - DRILL MODE)"
    )
    return jsonify({"success": True, "dispatch": disp})


@app.route("/api/sample_image/<sample_name>")
def get_sample_image(sample_name):
    """Serve a pre-packaged sample test image for one-click demonstration."""
    demo_dir = os.path.join(PROJECT_ROOT, "dataset", "demo_showcase")
    sample_paths = {
        "handgun": os.path.join(demo_dir, "handgun_threat.jpg"),
        "knife": os.path.join(demo_dir, "knife_threat.jpg"),
        "multi_weapon": os.path.join(demo_dir, "multi_weapon.jpg"),
        "dense_crowd": os.path.join(demo_dir, "dense_crowd.jpg"),
        "medium_crowd": os.path.join(demo_dir, "medium_crowd.jpg"),
        "calm_crowd": os.path.join(demo_dir, "calm_crowd.jpg"),
        # legacy aliases
        "weapon": os.path.join(demo_dir, "handgun_threat.jpg"),
        "crowd": os.path.join(demo_dir, "dense_crowd.jpg"),
        "weapons_multi": os.path.join(demo_dir, "multi_weapon.jpg"),
    }
    target_path = sample_paths.get(sample_name)
    if not target_path or not os.path.exists(target_path):
        return jsonify({"error": f"Sample '{sample_name}' not found"}), 404
    dir_name = os.path.dirname(target_path)
    file_name = os.path.basename(target_path)
    return send_from_directory(dir_name, file_name, mimetype="image/jpeg")


@app.route("/api/sources")
def get_sources():
    """Return available video sources and the currently active source."""
    return jsonify({
        "active": active_source_name,
        "camera_id": latest_status.get("camera_id", "CAM-01"),
        "sources": [
            {"id": "webcam", "name": "📷 Laptop Webcam", "desc": "Live local camera stream (Default)"},
            {"id": "crowd_walking", "name": "👥 Crowd Walking Feed", "desc": "Continuous dense crowd flow (crowd_test_video.mp4)"},
            {"id": "crowd_panic", "name": "🏃 Crowd Panic / Running", "desc": "Simulated stampede & high motion (crowd_running_test.mp4)"},
            {"id": "weapon_threat", "name": "🔫 Armed Threat Feed", "desc": "Active weapon confrontation (weapon_test_video.mp4)"},
            {"id": "custom", "name": "🌐 Network IP CCTV (RTSP)", "desc": "Live RTSP or HTTP IP camera stream URL"},
        ]
    })


@app.route("/api/set_source", methods=["POST"])
def set_video_source():
    """Dynamically switch live video source on the fly."""
    global requested_source, requested_camera_id, active_source_name
    data = request.get_json() or {}
    source_type = data.get("source_type", "webcam")
    custom_url = data.get("custom_url", "").strip()

    source_catalog = {
        "webcam": {
            "source": VIDEO_SOURCE,
            "camera_id": "CAM-01 (Laptop Webcam)",
            "label": "Live Laptop Webcam",
        },
        "crowd_walking": {
            "source": os.path.join(PROJECT_ROOT, "dataset", "crowd_test_video.mp4"),
            "camera_id": "CAM-02 (Concourse / Crowd Walk)",
            "label": "Crowd Walking Feed",
        },
        "crowd_panic": {
            "source": os.path.join(PROJECT_ROOT, "dataset", "crowd_running_test.mp4"),
            "camera_id": "CAM-03 (East Exit / Stampede Drill)",
            "label": "Crowd Panic / Stampede Feed",
        },
        "weapon_threat": {
            "source": os.path.join(PROJECT_ROOT, "dataset", "weapon_test_video.mp4"),
            "camera_id": "CAM-04 (Terminal 2 / Armed Threat)",
            "label": "Weapon Threat Feed",
        },
    }

    if source_type == "custom":
        if not custom_url:
            return jsonify({"error": "No RTSP/HTTP URL provided"}), 400
        chosen_source = custom_url
        chosen_cam_id = "CAM-EXT (Network CCTV Stream)"
        chosen_label = "Custom Network CCTV"
    elif source_type in source_catalog:
        cfg = source_catalog[source_type]
        chosen_source = cfg["source"]
        chosen_cam_id = cfg["camera_id"]
        chosen_label = cfg["label"]
    else:
        return jsonify({"error": f"Unknown source type: {source_type}"}), 400

    with source_switch_lock:
        requested_source = chosen_source
        requested_camera_id = chosen_cam_id
        active_source_name = source_type

    logger.info(f"Video source switch requested -> {source_type} ({chosen_cam_id})")
    return jsonify({
        "success": True,
        "source_type": source_type,
        "camera_id": chosen_cam_id,
        "label": chosen_label,
    })





# ===================== MAIN =====================

if __name__ == "__main__":
    source = _parse_video_source(VIDEO_SOURCE)
    logger.info(f"Starting Crowd Safety System...")
    logger.info(f"Video source: {source}")
    logger.info(f"Demo mode: {'ON' if DEMO_MODE else 'OFF'}")
    logger.info(f"Dashboard: http://127.0.0.1:{PORT}")

    worker_thread = threading.Thread(target=pipeline_worker, args=(source,), daemon=True)
    worker_thread.start()

    app.run(host=HOST, port=PORT, debug=False, threaded=True)