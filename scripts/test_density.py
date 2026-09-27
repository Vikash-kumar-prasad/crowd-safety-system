"""
Test script: run the trained YOLOv8 head detector + crowd density estimation
on a folder of images, and save annotated output images so you can visually
confirm the density module is working.

Usage:
    python test_density.py --source dataset/raw/PartA/test/images --limit 5
"""

import argparse
import os
import sys
import cv2
from ultralytics import YOLO

# allow importing our density module regardless of where this script is run from
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend", "app", "modules"))
from density import CrowdDensityEstimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best.pt")
    parser.add_argument("--source", type=str, default="dataset/raw/PartA/test/images")
    parser.add_argument("--limit", type=int, default=5, help="how many images to test")
    parser.add_argument("--out", type=str, default="notebooks/density_test_output")
    parser.add_argument("--conf", type=float, default=0.3, help="detection confidence threshold")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    model = YOLO(args.model)
    estimator = CrowdDensityEstimator(grid_rows=4, grid_cols=6)

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

        h, w = frame.shape[:2]

        results = model.predict(frame, conf=args.conf, verbose=False)
        boxes = []
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box[:4]
            boxes.append((x1, y1, x2, y2))

        density_grid, normalized_grid = estimator.compute_density_grid(boxes, w, h)
        level, score = estimator.classify_density(density_grid, w, h)

        # draw detections
        annotated = results[0].plot()
        # overlay density heatmap
        annotated = estimator.draw_heatmap_overlay(annotated, normalized_grid)
        annotated = estimator.draw_grid_lines(annotated)

        # add text label
        label = f"Heads: {len(boxes)}  Density: {level}  Score: {score:.5f}"
        cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        out_path = os.path.join(args.out, f"density_{fname}")
        cv2.imwrite(out_path, annotated)

        print(f"{fname}: heads={len(boxes)}, density={level} (score={score:.5f}) -> saved {out_path}")

    print(f"\nDone. Check images in: {args.out}")


if __name__ == "__main__":
    main()