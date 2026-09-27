"""
Live webcam test: runs YOLOv8 head detection + density estimation + behavior
anomaly detection together on your webcam feed, in real time.

Usage:
    python test_webcam.py

Press 'q' to quit the window.
"""

import os
import sys
import cv2
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend", "app", "modules"))
from density import CrowdDensityEstimator
from behavior import BehaviorAnomalyDetector


def main():
    model = YOLO("models/best.pt")
    density_estimator = CrowdDensityEstimator(grid_rows=4, grid_cols=6)
    behavior_detector = BehaviorAnomalyDetector()

    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        print("Could not open webcam. Check that no other app is using it.")
        return

    print("Webcam started. Press 'q' in the video window to quit.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        frame_count += 1
        h, w = frame.shape[:2]

        # run detection every frame (small model, should be fast enough on CPU
        # at webcam resolution; if it's laggy, we can skip frames later)
        results = model.predict(frame, conf=0.3, verbose=False)
        boxes = [tuple(b[:4]) for b in results[0].boxes.xyxy.cpu().numpy()]

        density_grid, normalized_grid = density_estimator.compute_density_grid(boxes, w, h)
        density_level, density_score = density_estimator.classify_density(density_grid)

        behavior_result = behavior_detector.update(frame, density_grid)

        # draw everything
        annotated = results[0].plot()
        annotated = density_estimator.draw_heatmap_overlay(annotated, normalized_grid)
        annotated = density_estimator.draw_grid_lines(annotated)

        line1 = f"Heads: {len(boxes)}  Density: {density_level} ({density_score:.5f})"
        line2 = f"Motion: {behavior_result['motion_score']:.2f}  Status: {behavior_result['status']}"

        cv2.putText(annotated, line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, line2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Crowd Safety - Live Test", annotated)

        # print to console too, every 10 frames, so you have a log even without watching the window
        if frame_count % 10 == 0:
            print(f"[frame {frame_count}] {line1} | {line2}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam test ended.")


if __name__ == "__main__":
    main()