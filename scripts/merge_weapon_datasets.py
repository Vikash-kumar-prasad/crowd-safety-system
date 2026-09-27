"""
Merge a second weapon dataset into the existing single-class weapon dataset
(class 0 = 'weapon').

Filters to keep only the specified class id, remaps it to class 0, and
copies the images + remapped labels into the existing train/valid/test
folders (with a filename prefix to avoid collisions).

Usage:
    python merge_weapon_datasets.py <new_dataset_dir> <existing_dataset_dir> <keep_class_id> <prefix>

Example (pistol dataset, Pistol is class 1):
    python merge_weapon_datasets.py dataset/raw_weapons_v2 dataset/raw_weapons 1 v2

Example (knife dataset, Knife is class 0):
    python merge_weapon_datasets.py dataset/raw_knife dataset/raw_weapons 0 knife
"""

import os
import sys
import shutil


def process_split(source_dir, split_name, dest_dir, prefix, keep_class_id):
    images_dir = os.path.join(source_dir, split_name, "images")
    labels_dir = os.path.join(source_dir, split_name, "labels")

    dest_images_dir = os.path.join(dest_dir, split_name, "images")
    dest_labels_dir = os.path.join(dest_dir, split_name, "labels")
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_labels_dir, exist_ok=True)

    if not os.path.isdir(images_dir):
        print(f"  Skipping {split_name}: no images folder found")
        return 0, 0

    copied = 0
    skipped_empty = 0

    for fname in os.listdir(images_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        base_name = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_dir, base_name + ".txt")

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        # keep only lines for the target class, remap to class 0
        kept_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if cls_id == keep_class_id:
                # remap to class 0 ("weapon")
                kept_lines.append("0 " + " ".join(parts[1:]))

        if not kept_lines:
            skipped_empty += 1
            continue  # no weapon in this image, skip it

        new_fname = f"{prefix}_{fname}"
        new_base = f"{prefix}_{base_name}"

        shutil.copy(os.path.join(images_dir, fname), os.path.join(dest_images_dir, new_fname))
        with open(os.path.join(dest_labels_dir, new_base + ".txt"), "w") as f:
            f.write("\n".join(kept_lines))

        copied += 1

    return copied, skipped_empty


def main():
    if len(sys.argv) != 5:
        print("Usage: python merge_weapon_datasets.py <new_dataset_dir> <existing_dataset_dir> <keep_class_id> <prefix>")
        sys.exit(1)

    new_dataset_dir = sys.argv[1]
    existing_dataset_dir = sys.argv[2]
    keep_class_id = int(sys.argv[3])
    prefix = sys.argv[4]

    for split in ["train", "valid", "test"]:
        copied, skipped = process_split(new_dataset_dir, split, existing_dataset_dir,
                                         prefix=prefix, keep_class_id=keep_class_id)
        print(f"[{split}] copied: {copied} (with weapon boxes), skipped (no weapon): {skipped}")

    print("\nDone merging. Existing dataset folder now contains the combined data.")


if __name__ == "__main__":
    main()