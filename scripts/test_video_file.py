"""
Run the full integrated pipeline on a video FILE (instead of webcam), and
save an annotated output video so you can review results frame-by-frame
or use it directly in your demo/report.

Usage:
    python test_video_file.py --source path/to/video.mp4
    python test_video_file.py --source path/to/video.mp4 --skip_frames 2
"""

import argparse
import os
import sys
import time
import cv2
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend", "app", "modules"))
from density import CrowdDensityEstimator
from behavior import BehaviorAnomalyDetector
from threat import ThreatDetector
from risk import RiskAssessor
from alert import AlertLogger


RISK_COLORS = {
    "Low": (0, 200, 0),
    "Medium": (0, 200, 255),
    "High": (0, 100, 255),
    "Critical": (0, 0, 255),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="path to input video file")
    parser.add_argument("--out", type=str, default=None, help="output video path (default: auto-named)")
    parser.add_argument("--skip_frames", type=int, default=1,
                         help="process every Nth frame (1=every frame, 2=every other, etc. -- use higher for faster CPU processing)")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Source video not found: {args.source}")
        return

    out_path = args.out or os.path.join("notebooks", "video_test_output",
                                         "annotated_" + os.path.basename(args.source))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("Loading models...")
    head_model = YOLO("models/best.pt")
    weapon_model = YOLO("models/best_weapon_v4.pt")

    density_estimator = CrowdDensityEstimator(grid_rows=4, grid_cols=6)
    behavior_detector = BehaviorAnomalyDetector()
    threat_detector = ThreatDetector(weapon_model, weapon_conf=0.65)
    risk_assessor = RiskAssessor()
    alert_logger = AlertLogger()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Could not open video: {args.source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps / args.skip_frames, (frame_w, frame_h))

    print(f"Processing '{args.source}' ({total_frames} frames, {fps:.1f} fps, {frame_w}x{frame_h})")
    print(f"Output will be saved to: {out_path}")

    frame_count = 0
    processed_count = 0
    last_weapon_state = False
    debug_dir = os.path.join("notebooks", "video_test_output", "weapon_debug")
    os.makedirs(debug_dir, exist_ok=True)

    # track risk level counts for a summary at the end
    risk_summary = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    frame_processing_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % args.skip_frames != 0:
            continue

        processed_count += 1
        h, w = frame.shape[:2]
        _frame_start_time = time.time()

        head_results = head_model.predict(frame, conf=0.3, verbose=False)
        head_boxes = [tuple(b[:4]) for b in head_results[0].boxes.xyxy.cpu().numpy()]

        density_grid, normalized_grid = density_estimator.compute_density_grid(head_boxes, w, h)
        density_level, density_score = density_estimator.classify_density(density_grid)

        behavior_result = behavior_detector.update(frame, density_grid)

        threat_result = threat_detector.assess(frame, head_boxes, behavior_result["motion_score"])

        risk_result = risk_assessor.assess(density_level, behavior_result["status"], threat_result)
        risk_summary[risk_result["risk_level"]] += 1

        alert = alert_logger.log_if_needed(
            frame, risk_result,
            extra_info={"heads": len(head_boxes), "density_level": density_level,
                        "behavior_status": behavior_result["status"], "frame_number": frame_count}
        )
        if alert:
            print(f"  [ALERT] {alert['risk_level']} at frame {frame_count}: {'; '.join(alert['reasons'])}")

        annotated = head_results[0].plot()
        annotated = density_estimator.draw_heatmap_overlay(annotated, normalized_grid)
        annotated = density_estimator.draw_grid_lines(annotated)

        for (x1, y1, x2, y2, conf) in threat_result["weapons"]:
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(annotated, f"weapon {conf:.2f}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        risk_color = RISK_COLORS.get(risk_result["risk_level"], (255, 255, 255))
        cv2.rectangle(annotated, (0, 0), (w, 40), risk_color, -1)
        risk_text = f"RISK: {risk_result['risk_level']} - {'; '.join(risk_result['reasons'])}"
        cv2.putText(annotated, risk_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

        status_text = (f"Heads: {len(head_boxes)}  Density: {density_level} ({density_score:.5f})  "
                        f"Behavior: {behavior_result['status']}  "
                        f"Weapons: {len(threat_result['weapons'])}")
        cv2.putText(annotated, status_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(annotated)

        frame_processing_times.append(time.time() - _frame_start_time)

        weapon_now = threat_result["weapon_detected"]
        if weapon_now and not last_weapon_state:
            debug_path = os.path.join(debug_dir, f"weapon_trigger_frame{frame_count}.jpg")
            cv2.imwrite(debug_path, annotated)
            print(f"  [DEBUG] Weapon detection triggered at frame {frame_count} -> saved {debug_path}")
        last_weapon_state = weapon_now

        if processed_count % 20 == 0:
            pct = (frame_count / total_frames * 100) if total_frames else 0
            print(f"[{pct:5.1f}%] frame {frame_count}/{total_frames} | {status_text} | RISK: {risk_result['risk_level']}")

    cap.release()
    writer.release()

    print(f"\nDone. Processed {processed_count} frames.")
    print(f"Output saved to: {out_path}")
    print("\nRisk level summary across video:")
    total_processed = sum(risk_summary.values())
    for level, count in risk_summary.items():
        pct = (count / total_processed * 100) if total_processed else 0
        print(f"  {level}: {count} frames ({pct:.1f}%)")

    if frame_processing_times:
        avg_time = sum(frame_processing_times) / len(frame_processing_times)
        achieved_fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"\nPerformance:")
        print(f"  Average processing time per frame: {avg_time*1000:.1f} ms")
        print(f"  Achieved pipeline speed: {achieved_fps:.2f} FPS (on this hardware, full pipeline: "
              f"head detection + density + behavior + weapon detection + risk assessment)")


if __name__ == "__main__":
    main()