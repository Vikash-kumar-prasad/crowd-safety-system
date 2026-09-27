"""
Threat Detection Module.

Combines two signals into a single "threat" assessment for a frame:
    1. Weapon detection -- a second YOLOv8 model (best_weapon.pt) trained
       separately from the head detector, flags guns/knives with bounding boxes.
    2. Possible Altercation heuristic -- reuses signals we already computed
       elsewhere (motion score from behavior.py, head positions from the
       head detector) to flag when people are close together AND moving a
       lot at the same time, which can indicate a physical altercation.
       This is a heuristic, not a trained fight classifier -- documented as
       such, with a real trained classifier noted as future work.
"""

import numpy as np


class ThreatDetector:
    def __init__(self, weapon_model, weapon_conf=0.65,
                 altercation_motion_threshold=3.0,
                 altercation_proximity_px=80,
                 weapon_vertical_reach=3.5,
                 weapon_horizontal_reach=2.5):
        """
        weapon_model: a loaded ultralytics YOLO model (the weapon detector,
            e.g. YOLO('models/best_weapon.pt'))
        weapon_conf: confidence threshold for weapon detections
        altercation_motion_threshold: motion_score (from BehaviorAnomalyDetector)
            above which we consider "a lot of movement" for the altercation check
        altercation_proximity_px: if two head boxes' centers are within this
            many pixels of each other, they're considered "close together"
        weapon_vertical_reach / weapon_horizontal_reach: a weapon detection
            is only kept if it falls within a "torso/hand zone" of its
            NEAREST detected head -- expressed as a multiple of that head's
            own height/width. Real held weapons are near a person's
            hands/torso, not at their feet, so this uses head size as a
            natural body-scale reference. Using the nearest head only (not
            "any" head) avoids false matches in dense crowds where many
            overlapping heads' zones could otherwise cover the same spot.
        """
        self.weapon_model = weapon_model
        self.weapon_conf = weapon_conf
        self.altercation_motion_threshold = altercation_motion_threshold
        self.altercation_proximity_px = altercation_proximity_px
        self.weapon_vertical_reach = weapon_vertical_reach
        self.weapon_horizontal_reach = weapon_horizontal_reach

    def detect_weapons(self, frame, head_boxes=None):
        """
        Returns a list of (x1, y1, x2, y2, confidence) for each weapon detected.
        If head_boxes is provided (non-empty), weapon detections outside the
        torso/hand zone of their nearest head are filtered out as likely
        false positives.
        If no heads are detected, weapons are still returned (a weapon could
        be visible without a person's head being in frame), but logged as
        unverified.
        """
        results = self.weapon_model.predict(frame, conf=self.weapon_conf, iou=0.3, verbose=False)
        weapons = []
        for box, conf in zip(results[0].boxes.xyxy.cpu().numpy(),
                              results[0].boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box[:4]
            weapons.append((x1, y1, x2, y2, float(conf)))

        if head_boxes:
            filtered = []
            for (x1, y1, x2, y2, conf) in weapons:
                wx, wy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                # find the single nearest head to this weapon detection
                nearest_head = min(
                    head_boxes,
                    key=lambda hb: np.hypot(wx - (hb[0] + hb[2]) / 2.0, wy - (hb[1] + hb[3]) / 2.0)
                )
                hx1, hy1, hx2, hy2 = nearest_head
                head_w = hx2 - hx1
                head_h = hy2 - hy1
                head_cx = (hx1 + hx2) / 2.0

                # allowed zone: horizontally centered on that nearest head,
                # extending from ABOVE the head (allowing raised weapons)
                # down through torso/hand height.
                x_ok = abs(wx - head_cx) <= self.weapon_horizontal_reach * head_w
                # FIX: Allow weapons above the head (wy < hy1) by extending
                # the zone upward by 1x head_h above the head top
                y_upper = hy1 - head_h  # one head-height above the top
                y_lower = hy1 + self.weapon_vertical_reach * head_h
                y_ok = y_upper <= wy <= y_lower

                if x_ok and y_ok:
                    filtered.append((x1, y1, x2, y2, conf))
            weapons = filtered

        return weapons

    def check_altercation(self, head_boxes, motion_score, frame_width=640):
        """
        head_boxes: list of (x1, y1, x2, y2) from the head detector for this frame
        motion_score: from BehaviorAnomalyDetector.update()['motion_score']
        frame_width: width of the current frame, used to scale proximity threshold

        Returns True if a possible altercation is flagged (high motion +
        two or more heads close together), else False.
        """
        if motion_score < self.altercation_motion_threshold:
            return False
        if len(head_boxes) < 2:
            return False  # need at least 2 people close together for "altercation"

        # Scale proximity threshold relative to frame width
        # 80px was calibrated for 640px width
        scaled_proximity = self.altercation_proximity_px * (frame_width / 640.0)

        centers = [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for (x1, y1, x2, y2) in head_boxes]

        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
                if dist <= scaled_proximity:
                    return True
        return False

    def assess(self, frame, head_boxes, motion_score):
        """
        Combines weapon detection + altercation check into one result dict.

        Returns:
            {
                "weapons": [(x1,y1,x2,y2,conf), ...],
                "weapon_detected": bool,
                "possible_altercation": bool,
                "threat_level": "None" | "Altercation" | "Weapon" | "Weapon + Altercation"
            }
        """
        weapons = self.detect_weapons(frame, head_boxes)
        weapon_detected = len(weapons) > 0
        frame_width = frame.shape[1] if frame is not None else 640
        altercation = self.check_altercation(head_boxes, motion_score, frame_width)

        if weapon_detected and altercation:
            threat_level = "Weapon + Altercation"
        elif weapon_detected:
            threat_level = "Weapon"
        elif altercation:
            threat_level = "Altercation"
        else:
            threat_level = "None"

        return {
            "weapons": weapons,
            "weapon_detected": weapon_detected,
            "possible_altercation": altercation,
            "threat_level": threat_level,
        }


# --------------------------------------------------------------------------
# Tuning notes:
# altercation_proximity_px depends heavily on your camera's resolution and
# distance from the scene -- 80px is a rough starting point at 640px frame
# width. Recalibrate after testing on real footage, same as density.py and
# behavior.py's thresholds were tuned earlier.
#
# Methodology note for your report: "Possible Altercation" is a rule-based
# heuristic (motion + proximity), not a trained violence/fight classifier.
# A dedicated video-classification model (e.g. trained on a fight-detection
# dataset) is identified as future work beyond Phase-I scope.
# --------------------------------------------------------------------------