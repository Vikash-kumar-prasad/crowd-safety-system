"""
Downsample images with a given filename prefix in a dataset's train/valid/test
folders, keeping only every Nth one. Used here to reduce the heavily
augmented knife dataset (which multiplied ~1700 source images into ~18,500
via rotation/flip/brightness augmentation) back down to a balanced size
relative to the rest of the weapon dataset.

Usage:
    python downsample_prefixed_images.py <dataset_dir> <prefix> <keep_every_nth>

Example:
    python downsample_prefixed_images.py dataset/raw_weapons knife 10
"""

import os
import sys


def downsample_split(dataset_dir, split_name, prefix, keep_every_nth):
    images_dir = os.path.join(dataset_dir, split_name, "images")
    labels_dir = os.path.join(dataset_dir, split_name, "labels")

    if not os.path.isdir(images_dir):
        return 0, 0

    matching = sorted([f for f in os.listdir(images_dir) if f.startswith(prefix + "_")])

    kept = 0
    removed = 0
    for i, fname in enumerate(matching):
        if i % keep_every_nth == 0:
            kept += 1
            continue  # keep this one

        # remove this image + its label
        base_name = os.path.splitext(fname)[0]
        img_path = os.path.join(images_dir, fname)
        label_path = os.path.join(labels_dir, base_name + ".txt")

        if os.path.exists(img_path):
            os.remove(img_path)
        if os.path.exists(label_path):
            os.remove(label_path)
        removed += 1

    return kept, removed


def main():
    if len(sys.argv) != 4:
        print("Usage: python downsample_prefixed_images.py <dataset_dir> <prefix> <keep_every_nth>")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    prefix = sys.argv[2]
    keep_every_nth = int(sys.argv[3])

    for split in ["train", "valid", "test"]:
        kept, removed = downsample_split(dataset_dir, split, prefix, keep_every_nth)
        print(f"[{split}] kept: {kept}, removed: {removed}")

    print("\nDone downsampling.")


if __name__ == "__main__":
    main()