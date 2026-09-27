"""
Batch test: run head detection + weapon detection (with proper geometric
filtering) on a folder of images, saving annotated results and printing
a summary for each.

Usage:
    python test_batch_images.py --source dataset/gun_test_images
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
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--out", type=str, default="notebooks/batch_test_output")
    parser.add_argument("--weapon_conf", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading models...")
    head_model = YOLO("models/best.pt")
    weapon_model = YOLO("models/best_weapon_v4.pt")
    threat_detector = ThreatDetector(weapon_model, weapon_conf=args.weapon_conf)

    image_files = [f for f in os.listdir(args.source)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_files.sort()

    if not image_files:
        print(f"No images found in {args.source}")
        return

    print(f"\nFound {len(image_files)} images. Processing...\n")

    for fname in image_files:
        img_path = os.path.join(args.source, fname)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"{fname}: could not read image, skipping")
            continue

        head_results = head_model.predict(frame, conf=0.3, verbose=False)
        head_boxes = [tuple(b[:4]) for b in head_results[0].boxes.xyxy.cpu().numpy()]

        threat_result = threat_detector.assess(frame, head_boxes, motion_score=0.0)
        weapons = threat_result["weapons"]

        # draw everything
        annotated = head_results[0].plot()
        for (x1, y1, x2, y2, conf) in weapons:
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.putText(annotated, f"weapon {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        out_path = os.path.join(args.out, f"result_{fname}")
        cv2.imwrite(out_path, annotated)

        conf_list = ", ".join(f"{c:.2f}" for (_, _, _, _, c) in weapons)
        print(f"{fname}: heads={len(head_boxes)}, weapons detected={len(weapons)} "
              f"[{conf_list}] -> saved {out_path}")

    print(f"\nDone. All results saved in: {args.out}")


if __name__ == "__main__":
    main()