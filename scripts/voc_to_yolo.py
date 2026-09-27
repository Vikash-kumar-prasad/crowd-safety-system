"""
Convert a Pascal VOC-format SCUT-HEAD part (Annotations/ + JPEGImages/ +
ImageSets/Main/{train,val,test}.txt) into YOLOv8 folder format:

    output_dir/
        train/images/*.jpg   train/labels/*.txt
        valid/images/*.jpg   valid/labels/*.txt
        test/images/*.jpg    test/labels/*.txt

Usage:
    python voc_to_yolo.py <source_voc_dir> <output_dir>

Example:
    python voc_to_yolo.py "dataset/raw_voc/head_b/SCUT_HEAD_Part_B" "dataset/raw/PartB"
"""

import os
import sys
import shutil
import xml.etree.ElementTree as ET
import cv2

# Only one class in SCUT-HEAD
CLASS_NAMES = ["head"]


def convert_box(size, box):
    """Convert VOC (xmin, ymin, xmax, ymax) to YOLO normalized (x_center, y_center, w, h)."""
    img_w, img_h = size
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return x_center, y_center, w, h


def convert_annotation(xml_path, img_path, label_out_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Some PartB XML <size> tags are corrupt (width/height = 0), since PartB
    # images were crawled from the internet. Reading the real image dimensions
    # directly is more reliable than trusting the XML metadata.
    img = cv2.imread(img_path)
    if img is None:
        return False  # unreadable image, caller should skip this file
    img_h, img_w = img.shape[:2]
    if img_w == 0 or img_h == 0:
        return False

    lines = []
    for obj in root.findall("object"):
        # SCUT-HEAD is single-class: every <object> in these files is a head,
        # even though some XML annotations use <name>person</name> instead of
        # <name>head</name>. Since nc=1, we map every object to class 0.
        cls_id = 0

        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        x, y, w, h = convert_box((img_w, img_h), (xmin, ymin, xmax, ymax))
        lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    with open(label_out_path, "w") as f:
        f.write("\n".join(lines))
    return True


def process_split(source_dir, split_name, split_file, output_dir, output_split_name):
    """split_name: name used in ImageSets/Main (train/val/test)
    output_split_name: folder name to write (train/valid/test)"""
    annotations_dir = os.path.join(source_dir, "Annotations")
    images_dir = os.path.join(source_dir, "JPEGImages")

    with open(split_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]

    out_images_dir = os.path.join(output_dir, output_split_name, "images")
    out_labels_dir = os.path.join(output_dir, output_split_name, "labels")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    converted = 0
    skipped = 0
    for image_id in ids:
        xml_path = os.path.join(annotations_dir, image_id + ".xml")
        img_path = os.path.join(images_dir, image_id + ".jpg")

        if not os.path.exists(xml_path) or not os.path.exists(img_path):
            skipped += 1
            continue

        ok = convert_annotation(xml_path, img_path, os.path.join(out_labels_dir, image_id + ".txt"))
        if not ok:
            skipped += 1
            continue

        shutil.copy(img_path, os.path.join(out_images_dir, image_id + ".jpg"))
        converted += 1

    print(f"[{output_split_name}] converted: {converted}, skipped (unreadable/bad image): {skipped}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python voc_to_yolo.py <source_voc_dir> <output_dir>")
        sys.exit(1)

    source_dir = sys.argv[1]
    output_dir = sys.argv[2]

    imagesets_main = os.path.join(source_dir, "ImageSets", "Main")

    # Map VOC split file -> our output folder name
    split_map = {
        "train.txt": "train",
        "val.txt": "valid",
        "test.txt": "test",
    }

    for split_file_name, out_name in split_map.items():
        split_file_path = os.path.join(imagesets_main, split_file_name)
        if not os.path.exists(split_file_path):
            print(f"Skipping {split_file_name} (not found)")
            continue
        process_split(source_dir, split_file_name, split_file_path, output_dir, out_name)

    print("\nDone. Output written to:", output_dir)


if __name__ == "__main__":
    main()