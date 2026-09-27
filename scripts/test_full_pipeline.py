"""
Fully integrated pipeline test: runs ALL modules together on your webcam feed.

Head detection -> Density estimation -> Behavior/anomaly detection
                -> Weapon detection -> Risk assessment

This is the real end-to-end system, not individual module tests.

Usage:
    python test_full_pipeline.py

Press 'q' to quit the window.
"""

import os
import sys
import cv2
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend", "app", "modules"))
from density import CrowdDensityEstimator
from behavior import BehaviorAnomalyDetector
from threat import ThreatDetector
from risk import RiskAssessor


# colors for risk levels, used for the on-screen banner
RISK_COLORS = {
    "Low": (0, 200, 0),        # green
    "Medium": (0, 200, 255),   # yellow/orange
    "High": (0, 100, 255),     # orange/red
    "Critical": (0, 0, 255),   # red
}


def main():
    print("Loading models...")
    head_model = YOLO("models/best.pt")
    weapon_model = YOLO("models/best_weapon_v4.pt")

    density_estimator = CrowdDensityEstimator(grid_rows=4, grid_cols=6)
    behavior_detector = BehaviorAnomalyDetector()
    threat_detector = ThreatDetector(weapon_model, weapon_conf=0.65)
    risk_assessor = RiskAssessor()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Pipeline started. Press 'q' in the video window to quit.")

    frame_count = 0
    debug_dir = "notebooks/weapon_debug_captures"
    os.makedirs(debug_dir, exist_ok=True)
    last_weapon_state = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        frame_count += 1
        h, w = frame.shape[:2]

        # 1. Head detection
        head_results = head_model.predict(frame, conf=0.3, verbose=False)
        head_boxes = [tuple(b[:4]) for b in head_results[0].boxes.xyxy.cpu().numpy()]

        # 2. Density estimation
        density_grid, normalized_grid = density_estimator.compute_density_grid(head_boxes, w, h)
        density_level, density_score = density_estimator.classify_density(density_grid)

        # 3. Behavior/anomaly detection
        behavior_result = behavior_detector.update(frame, density_grid)

        # 4. Weapon detection + altercation heuristic
        threat_result = threat_detector.assess(frame, head_boxes, behavior_result["motion_score"])

        # 5. Overall risk assessment
        risk_result = risk_assessor.assess(density_level, behavior_result["status"], threat_result)

        # ---- draw everything ----
        annotated = head_results[0].plot()  # head boxes
        annotated = density_estimator.draw_heatmap_overlay(annotated, normalized_grid)
        annotated = density_estimator.draw_grid_lines(annotated)

        # weapon boxes (red)
        for (x1, y1, x2, y2, conf) in threat_result["weapons"]:
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(annotated, f"weapon {conf:.2f}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # risk banner at the top
        risk_color = RISK_COLORS.get(risk_result["risk_level"], (255, 255, 255))
        cv2.rectangle(annotated, (0, 0), (w, 40), risk_color, -1)
        risk_text = f"RISK: {risk_result['risk_level']} - {'; '.join(risk_result['reasons'])}"
        cv2.putText(annotated, risk_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # status line below the banner
        status_text = (f"Heads: {len(head_boxes)}  Density: {density_level} ({density_score:.5f})  "
                        f"Behavior: {behavior_result['status']}  "
                        f"Weapons: {len(threat_result['weapons'])}")
        cv2.putText(annotated, status_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Crowd Safety - Full Pipeline", annotated)

        # debug: save a screenshot the moment weapon detection turns on,
        # so we can inspect exactly what triggered it
        weapon_now = threat_result["weapon_detected"]
        if weapon_now and not last_weapon_state:
            debug_path = os.path.join(debug_dir, f"weapon_trigger_frame{frame_count}.jpg")
            cv2.imwrite(debug_path, annotated)
            print(f"  [DEBUG] Weapon detection triggered -> saved {debug_path}")
        last_weapon_state = weapon_now

        if frame_count % 10 == 0:
            print(f"[frame {frame_count}] {status_text} | RISK: {risk_result['risk_level']} "
                  f"({'; '.join(risk_result['reasons'])})")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Pipeline test ended.")


if __name__ == "__main__":
    main()