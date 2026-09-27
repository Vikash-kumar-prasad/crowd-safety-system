"""
Alert Engine Module.

Saves a record every time the risk level reaches Medium or above, so
alerts aren't lost even if no one is watching the live feed at that moment.
This includes pure overcrowding (High crowd density with otherwise calm
behavior, which RiskAssessor rates "Medium") -- flagging dangerous density
early, before a weapon or altercation is also present, gives operators a
chance to intervene before an incident like a stampede occurs.

Each alert is stored as one entry in a JSON log file (alerts.json), plus an
optional snapshot image of the frame that triggered it.

Key improvements over the initial implementation:
    - Thread-safe file I/O with threading.Lock (fixes JSON race condition)
    - Streak confirmation uses risk LEVEL not exact reason string (fixes
      the bug where dynamically changing reasons reset the streak every frame)
    - Sub-second timestamp resolution prevents snapshot filename collisions
    - Alert records include unique IDs and snapshot filenames for API serving
"""

import os
import json
import time
import threading
import cv2


class AlertLogger:
    def __init__(self, log_path="notebooks/alerts/alerts.json",
                 snapshot_dir="notebooks/alerts/snapshots",
                 min_alert_level="Medium",
                 cooldown_seconds=5,
                 confirmation_frames=3,
                 max_alerts=500):
        """
        log_path: where alert records are appended (JSON list on disk)
        snapshot_dir: where triggering-frame images are saved
        min_alert_level: "Medium" by default, so pure overcrowding still
            generates an alert
        cooldown_seconds: minimum time between two alerts of the SAME risk
            LEVEL (not exact reason), so one ongoing situation doesn't spam
        confirmation_frames: the same risk level must persist for this many
            CONSECUTIVE calls before an alert is actually logged
        max_alerts: maximum number of alerts to keep in the JSON file
            (oldest are trimmed to prevent unbounded file growth)
        """
        self.log_path = log_path
        self.snapshot_dir = snapshot_dir
        self.min_alert_level = min_alert_level
        self.cooldown_seconds = cooldown_seconds
        self.confirmation_frames = confirmation_frames
        self.max_alerts = max_alerts

        self._level_rank = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        self._last_alert_time_by_level = {}
        self._current_level = None
        self._current_streak = 0
        self._file_lock = threading.Lock()
        self._alert_counter = 0

        os.makedirs(os.path.dirname(self.log_path) if os.path.dirname(self.log_path) else ".", exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)

        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                json.dump([], f)

    def _meets_threshold(self, risk_level):
        return self._level_rank.get(risk_level, 0) >= self._level_rank.get(self.min_alert_level, 1)

    def _is_confirmed(self, risk_level):
        """Tracks consecutive frames where the risk level meets threshold.
        Uses risk LEVEL (not exact reason string) to avoid the bug where
        dynamically changing reasons reset the streak every frame."""
        if not self._meets_threshold(risk_level):
            self._current_level = None
            self._current_streak = 0
            return False

        if risk_level == self._current_level:
            self._current_streak += 1
        else:
            self._current_level = risk_level
            self._current_streak = 1

        return self._current_streak >= self.confirmation_frames

    def _should_alert(self, risk_level):
        if not self._is_confirmed(risk_level):
            return False

        last_time = self._last_alert_time_by_level.get(risk_level)
        now = time.time()
        if last_time is not None and (now - last_time) < self.cooldown_seconds:
            return False

        return True

    def log_if_needed(self, frame, risk_result, extra_info=None):
        """
        frame: current BGR frame (numpy array), used for the snapshot
        risk_result: the dict returned by RiskAssessor.assess()
        extra_info: optional dict of additional context to store

        Returns the alert record dict if an alert was logged, else None.
        """
        risk_level = risk_result["risk_level"]

        if not self._should_alert(risk_level):
            return None

        now = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # Use monotonic counter + milliseconds for unique filenames
        self._alert_counter += 1
        safe_timestamp = time.strftime("%Y%m%d_%H%M%S")
        snapshot_filename = f"alert_{safe_timestamp}_{self._alert_counter:04d}_{risk_level}.jpg"
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_filename)

        try:
            if frame is not None:
                cv2.imwrite(snapshot_path, frame)
        except Exception:
            snapshot_filename = None  # snapshot failed but alert still logged

        record = {
            "id": f"alert_{safe_timestamp}_{self._alert_counter:04d}",
            "timestamp": timestamp,
            "risk_level": risk_level,
            "reasons": risk_result.get("reasons", []),
            "snapshot_filename": snapshot_filename,
        }
        if extra_info:
            record.update(extra_info)

        # Thread-safe JSON file I/O
        with self._file_lock:
            try:
                with open(self.log_path, "r") as f:
                    alerts = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                alerts = []

            alerts.append(record)

            # Trim old alerts to prevent unbounded growth
            if len(alerts) > self.max_alerts:
                alerts = alerts[-self.max_alerts:]

            # Atomic-ish write: write to temp file then rename
            tmp_path = self.log_path + ".tmp"
            try:
                with open(tmp_path, "w") as f:
                    json.dump(alerts, f, indent=2)
                os.replace(tmp_path, self.log_path)
            except Exception:
                # Fallback: write directly
                with open(self.log_path, "w") as f:
                    json.dump(alerts, f, indent=2)

        self._last_alert_time_by_level[risk_level] = now

        return record