"""
Behavior / Anomaly Detection Module.

Compares consecutive video frames to detect unusual crowd behavior:
    - Sudden Dispersal   : density drops sharply (people scattering)
    - Rapid Convergence  : density rises sharply (people rushing in)
    - High Motion        : overall movement magnitude is unusually high (possible panic)
    - Normal             : nothing unusual

Two signals are combined:
    1. Optical flow motion magnitude (how much movement is happening overall)
    2. Frame-to-frame density grid change (from density.py's output)

Improvements over initial implementation:
    - Rolling window for density change (reduces single-frame noise)
    - Aspect-ratio-preserving resize for optical flow
    - FPS-independent motion scoring is documented as a future improvement
"""

import cv2
import numpy as np
from collections import deque


class BehaviorAnomalyDetector:
    def __init__(self, motion_threshold=2.5, density_change_threshold=0.35,
                 min_count_for_density_check=5, history_window=5):
        """
        motion_threshold: average optical flow magnitude above which we flag
            "High Motion". Tune after watching real output.
        density_change_threshold: fractional change in total head count
            above which we flag dispersal/convergence.
        min_count_for_density_check: dispersal/convergence is only evaluated
            when the average count over the history window was at least this
            many heads. Prevents noise from flickering detections.
        history_window: number of recent head counts to average over for
            density change calculation (reduces single-frame jitter).
        """
        self.motion_threshold = motion_threshold
        self.density_change_threshold = density_change_threshold
        self.min_count_for_density_check = min_count_for_density_check

        self.prev_gray = None
        # Rolling window of recent head counts for temporal smoothing
        self._count_history = deque(maxlen=max(history_window, 2))

    def update(self, frame, density_grid):
        """
        Call this once per frame.
        frame: the current BGR frame (numpy array)
        density_grid: this frame's density grid from CrowdDensityEstimator

        Returns a dict:
            {
                "motion_score": float,
                "density_change_pct": float or None,
                "status": "Normal" | "Sudden Dispersal" | "Rapid Convergence" | "High Motion"
            }
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize preserving aspect ratio (target ~160px wide)
        h, w = gray.shape[:2]
        target_w = 160
        scale = target_w / w
        target_h = max(int(h * scale), 1)
        gray_small = cv2.resize(gray, (target_w, target_h))

        motion_score = 0.0
        if self.prev_gray is not None:
            # Ensure previous and current frames have compatible shapes
            if self.prev_gray.shape == gray_small.shape:
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_gray, gray_small, None,
                    pyr_scale=0.5, levels=2, winsize=15,
                    iterations=2, poly_n=5, poly_sigma=1.2, flags=0
                )
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_score = float(np.mean(magnitude))
            else:
                # Resolution changed mid-stream; skip this frame's flow
                pass

        total_count = float(density_grid.sum())
        self._count_history.append(total_count)

        # Compute density change against the rolling average (not single previous frame)
        density_change_pct = None
        if len(self._count_history) >= 2:
            # Compare current count against average of older readings
            older_avg = np.mean(list(self._count_history)[:-1])
            if older_avg >= self.min_count_for_density_check:
                density_change_pct = (total_count - older_avg) / older_avg

        # Decide status
        status = "Normal"
        if density_change_pct is not None:
            if density_change_pct <= -self.density_change_threshold:
                status = "Sudden Dispersal"
            elif density_change_pct >= self.density_change_threshold:
                status = "Rapid Convergence"

        if motion_score >= self.motion_threshold:
            status = "High Motion"  # motion takes priority as more urgent

        # Update state for next call
        self.prev_gray = gray_small

        return {
            "motion_score": round(motion_score, 3),
            "density_change_pct": round(density_change_pct, 4) if density_change_pct is not None else None,
            "status": status,
        }


# --------------------------------------------------------------------------
# Tuning notes:
# - motion_threshold: watch the printed motion_score during normal vs
#   abnormal movement and adjust accordingly.
# - density_change_threshold: test with groups of people entering/leaving.
# - history_window: larger = more stable but slower to react (5 is a good
#   balance for 15-30 FPS video).
# - For FPS-independent motion scoring, normalize motion_score by the
#   time delta between frames. This is noted as future work.
# --------------------------------------------------------------------------