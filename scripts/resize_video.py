"""
Resize a video down to a target width (keeping aspect ratio), so large
downloaded clips (e.g. 4K) process much faster through the pipeline.

Usage:
    python resize_video.py --source path/to/big_video.mp4 --out path/to/small_video.mp4 --width 960
"""

import argparse
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--width", type=int, default=960)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Could not open {args.source}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    new_w = args.width
    new_h = int(orig_h * (new_w / orig_w))
    # ensure even dimensions (some codecs require this)
    new_h -= new_h % 2

    print(f"Resizing {orig_w}x{orig_h} -> {new_w}x{new_h}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (new_w, new_h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (new_w, new_h))
        writer.write(resized)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  {frame_count} frames processed...")

    cap.release()
    writer.release()
    print(f"Done. Saved to {args.out} ({frame_count} frames)")


if __name__ == "__main__":
    main()