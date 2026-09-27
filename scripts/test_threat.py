"""
Test script: run weapon detection on sample images from the weapon test set,
to visually confirm the threat detection module works.

Usage:
    python test_threat.py --source dataset/raw_weapons/test/images --limit 8
"""

import argparse
import os
import sys
import cv2
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend", "app", "modules"))
from threat import ThreatDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weapon_model", type=str, default="models/best_weapon.pt")
    parser.add_argument("--source", type=str, default="dataset/raw_weapons/test/images")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--out", type=str, default="notebooks/threat_test_output")
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    weapon_model = YOLO(args.weapon_model)
    threat_detector = ThreatDetector(weapon_model, weapon_conf=args.conf)

    image_files = [f for f in os.listdir(args.source) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_files = sorted(image_files)[:args.limit]

    if not image_files:
        print(f"No images found in {args.source}")
        return

    for fname in image_files:
        img_path = os.path.join(args.source, fname)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Could not read {img_path}, skipping.")
            continue

        # no head detector / motion in this simple test, so we only exercise
        # the weapon detection half of ThreatDetector here
        result = threat_detector.assess(frame, head_boxes=[], motion_score=0.0)

        annotated = frame.copy()
        for (x1, y1, x2, y2, conf) in result["weapons"]:
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(annotated, f"weapon {conf:.2f}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        label = f"Threat level: {result['threat_level']}  (weapons found: {len(result['weapons'])})"
        cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

        out_path = os.path.join(args.out, f"threat_{fname}")
        cv2.imwrite(out_path, annotated)

        print(f"{fname}: {result['threat_level']}, weapons={len(result['weapons'])} -> saved {out_path}")

    print(f"\nDone. Check images in: {args.out}")


if __name__ == "__main__":
    main()