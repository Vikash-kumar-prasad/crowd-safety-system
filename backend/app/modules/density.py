"""
Crowd Density Estimation Module.

Takes head/person bounding boxes (from the trained YOLOv8 detector) for a
single frame and computes a grid-based density map + an overall density
classification (Low / Medium / High).

This is a detection-based density estimator: instead of a separate CNN that
regresses a density map directly from pixels (the heavier "true" crowd-counting
approach), we reuse the YOLO detections we already have and turn them into a
density signal. This is a legitimate, commonly used technique for
detection-based crowd analysis.
"""

import numpy as np
import cv2


class CrowdDensityEstimator:
    def __init__(self, grid_rows=4, grid_cols=6,
                 low_threshold=0.00030, high_threshold=0.00080):
        """
        grid_rows, grid_cols: how many cells to split each frame into.
        low_threshold / high_threshold: heads-per-pixel-area cutoffs (per cell)
            used to classify a cell (and the overall frame) as Low/Medium/High.
            These are starting defaults — tune them after looking at real
            output on your video/images (see notes at bottom of this file).
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def compute_density_grid(self, boxes, frame_width, frame_height, ref_width=640, ref_height=480):
        """
        boxes: list of (x1, y1, x2, y2) in pixel coordinates (YOLO detections)
        frame_width, frame_height: actual dimensions of the frame the boxes came from
        ref_width, ref_height: a fixed reference resolution used to make density
            scores comparable across frames/images of different sizes (e.g. the
            SCUT-HEAD test images vary in resolution, but a real camera feed
            won't -- this keeps the classification consistent either way).
        Returns:
            density_grid: 2D numpy array (grid_rows x grid_cols) of head counts
            normalized_grid: same shape, values normalized 0-1 relative to the
                             densest cell (useful for heatmap coloring)
        """
        density_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        # scale box coordinates into the fixed reference resolution first,
        # so a 1-head image at 200x150 doesn't score differently than the
        # same crowd density at 1920x1080.
        scale_x = ref_width / frame_width
        scale_y = ref_height / frame_height

        cell_w = ref_width / self.grid_cols
        cell_h = ref_height / self.grid_rows

        for (x1, y1, x2, y2) in boxes:
            cx = (x1 + x2) / 2.0 * scale_x
            cy = (y1 + y2) / 2.0 * scale_y

            col = int(cx // cell_w)
            row = int(cy // cell_h)

            # clamp in case a box center sits exactly on the frame edge
            col = min(max(col, 0), self.grid_cols - 1)
            row = min(max(row, 0), self.grid_rows - 1)

            density_grid[row, col] += 1

        max_count = density_grid.max()
        normalized_grid = density_grid / max_count if max_count > 0 else density_grid

        return density_grid, normalized_grid

    def classify_density(self, density_grid, ref_width=640, ref_height=480):
        """
        Returns overall classification: 'Low', 'Medium', or 'High',
        plus the raw score used to decide (heads per unit area, scaled).
        Uses the fixed reference resolution so scores stay comparable.
        """
        cell_area = (ref_width / self.grid_cols) * (ref_height / self.grid_rows)
        busiest_cell_count = density_grid.max()
        score = busiest_cell_count / cell_area

        if score >= self.high_threshold:
            level = "High"
        elif score >= self.low_threshold:
            level = "Medium"
        else:
            level = "Low"

        return level, float(score)

    def draw_heatmap_overlay(self, frame, normalized_grid, alpha=0.40):
        """
        Returns a copy of `frame` with a smooth density heatmap overlaid.
        If no heads are detected (normalized_grid max is 0), returns frame untouched.
        Areas with zero crowd density remain completely transparent (no blue wash).
        """
        if normalized_grid.max() <= 0:
            return frame.copy()

        h, w = frame.shape[:2]
        heatmap_small = (normalized_grid * 255).astype(np.uint8)
        # Bicubic resize + Gaussian blur for smooth natural density contours
        heatmap_resized = cv2.resize(heatmap_small, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_blurred = cv2.GaussianBlur(heatmap_resized, (51, 51), 0)
        heatmap_color = cv2.applyColorMap(heatmap_blurred, cv2.COLORMAP_JET)

        # Dynamic opacity: zero density = 100% transparent, high density = alpha blend
        heat_weight = (heatmap_blurred.astype(np.float32) / 255.0 * alpha)[:, :, np.newaxis]

        overlay = (frame.astype(np.float32) * (1.0 - heat_weight) +
                   heatmap_color.astype(np.float32) * heat_weight).astype(np.uint8)
        return overlay

    def draw_grid_lines(self, frame, heads_count=0):
        """Draw subtle grid boundaries only when crowds are actively detected."""
        if heads_count <= 0:
            return frame.copy()

        h, w = frame.shape[:2]
        overlay = frame.copy()
        line_color = (60, 60, 60)  # Subtle dark line
        for c in range(1, self.grid_cols):
            x = int(c * w / self.grid_cols)
            cv2.line(overlay, (x, 0), (x, h), line_color, 1)
        for r in range(1, self.grid_rows):
            y = int(r * h / self.grid_rows)
            cv2.line(overlay, (0, y), (w, y), line_color, 1)
        # Blend grid lines with low opacity (30%)
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)


# --------------------------------------------------------------------------
# Tuning notes:
# The low_threshold/high_threshold defaults are a starting point, NOT
# calibrated to your actual dataset yet. After running run_density_test.py
# on a few real images, look at the printed density scores across several
# frames (empty room vs packed classroom vs street crowd) and adjust the
# two threshold values above so Low/Medium/High actually line up with what
# the image visually looks like. Mention in your report that thresholds were
# empirically tuned on the SCUT-HEAD validation images.
# --------------------------------------------------------------------------