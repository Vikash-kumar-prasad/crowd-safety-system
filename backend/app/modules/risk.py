"""
Risk Assessment Module.

Combines the outputs of density.py, behavior.py, and threat.py into one
overall risk level: Low, Medium, High, or Critical.

Logic (in priority order -- most severe check wins):
    1. Any weapon detected                        -> Critical
    2. Possible altercation + High crowd density   -> Critical
    3. Possible altercation (alone)                -> High
    4. High crowd density + (Sudden Dispersal or
       Rapid Convergence or High Motion)           -> High
    5. High crowd density (alone)                  -> Medium
    6. Medium crowd density + any abnormal behavior -> Medium
    7. Everything else (Low density, Normal behavior, no threats) -> Low

This is pure rule-based logic (no ML) -- deliberately simple and explainable,
since a crowd-safety alert system needs to be auditable: an operator or
your report should be able to see exactly why a given risk level was raised.
"""


class RiskAssessor:
    """Combines density, behavior, and threat signals into an overall risk level.

    The assessment accumulates ALL applicable reasons (instead of returning
    on the first match) so operators see the full picture, while still using
    a strict priority hierarchy for the final risk level.

    Priority order (most severe wins):
        1. Weapon detected                                -> Critical
        2. Altercation + High density                     -> Critical
        3. Altercation alone                              -> High
        4. High density + abnormal behavior               -> High
        5. Any abnormal behavior (even Low density)       -> Medium  [NEW]
        6. High density alone                             -> Medium
        7. Medium density + abnormal behavior             -> Medium
        8. Everything else                                -> Low
    """

    _LEVEL_RANK = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

    def assess(self, density_level, behavior_status, threat_result):
        """
        Returns:
            {
                "risk_level": "Low" | "Medium" | "High" | "Critical",
                "reasons": [list of strings explaining why]
            }
        """
        weapon_detected = threat_result.get("weapon_detected", False)
        altercation = threat_result.get("possible_altercation", False)
        abnormal_behavior = (behavior_status or "Normal") != "Normal"

        reasons = []
        risk_level = "Low"

        def _escalate(new_level, reason):
            nonlocal risk_level
            reasons.append(reason)
            if self._LEVEL_RANK.get(new_level, 0) > self._LEVEL_RANK.get(risk_level, 0):
                risk_level = new_level

        # --- evaluate every applicable rule (accumulate reasons) ---

        # 1. Weapon -> Critical
        if weapon_detected:
            _escalate("Critical", "Weapon detected")

        # 2. Altercation + High density -> Critical
        if altercation and density_level == "High":
            _escalate("Critical", "Possible altercation in a high-density crowd")
        elif altercation:
            # 3. Altercation alone -> High
            _escalate("High", "Possible altercation detected")

        # 4. High density + abnormal behavior -> High
        if density_level == "High" and abnormal_behavior:
            _escalate("High", f"High crowd density with abnormal behavior ({behavior_status})")
        elif density_level == "High":
            # 5. High density alone -> Medium
            _escalate("Medium", "High crowd density")

        # 6. Medium density + abnormal behavior -> Medium
        if density_level == "Medium" and abnormal_behavior:
            _escalate("Medium", f"Medium crowd density with abnormal behavior ({behavior_status})")

        # 7. [NEW] Abnormal behavior at any density -> at least Medium
        #    Previously, Sudden Dispersal at Low density was dropped to Low!
        if abnormal_behavior and density_level == "Low":
            _escalate("Medium", f"Abnormal crowd behavior detected ({behavior_status})")

        # 8. Default
        if not reasons:
            reasons.append("No significant risk factors detected")

        return {"risk_level": risk_level, "reasons": reasons}


# --------------------------------------------------------------------------
# Design notes for your report:
# - This is intentionally rule-based rather than a trained classifier, so
#   every risk decision is traceable to a specific input signal ("reasons"
#   list) -- important for a safety system where false alerts/misses need
#   to be debuggable, not a black box.
# - The priority order (weapon > altercation > density+behavior combos >
#   density alone) reflects that direct threats to physical safety should
#   never be masked by a lower-priority crowd density reading.
# --------------------------------------------------------------------------