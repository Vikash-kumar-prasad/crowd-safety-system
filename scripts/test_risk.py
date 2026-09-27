"""
Test script: run RiskAssessor against a set of made-up scenarios to sanity
check the logic produces sensible risk levels, without needing a live camera.

Usage:
    python test_risk.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend", "app", "modules"))
from risk import RiskAssessor


def main():
    assessor = RiskAssessor()

    scenarios = [
        {
            "label": "Empty room, nothing happening",
            "density_level": "Low",
            "behavior_status": "Normal",
            "threat_result": {"weapon_detected": False, "possible_altercation": False},
        },
        {
            "label": "Busy but calm crowd",
            "density_level": "Medium",
            "behavior_status": "Normal",
            "threat_result": {"weapon_detected": False, "possible_altercation": False},
        },
        {
            "label": "Medium crowd + people scattering",
            "density_level": "Medium",
            "behavior_status": "Sudden Dispersal",
            "threat_result": {"weapon_detected": False, "possible_altercation": False},
        },
        {
            "label": "Packed crowd, calm",
            "density_level": "High",
            "behavior_status": "Normal",
            "threat_result": {"weapon_detected": False, "possible_altercation": False},
        },
        {
            "label": "Packed crowd + high motion (possible panic)",
            "density_level": "High",
            "behavior_status": "High Motion",
            "threat_result": {"weapon_detected": False, "possible_altercation": False},
        },
        {
            "label": "Two people fighting, low density area",
            "density_level": "Low",
            "behavior_status": "High Motion",
            "threat_result": {"weapon_detected": False, "possible_altercation": True},
        },
        {
            "label": "Fight breaking out in a packed crowd",
            "density_level": "High",
            "behavior_status": "High Motion",
            "threat_result": {"weapon_detected": False, "possible_altercation": True},
        },
        {
            "label": "Weapon spotted, otherwise calm",
            "density_level": "Low",
            "behavior_status": "Normal",
            "threat_result": {"weapon_detected": True, "possible_altercation": False},
        },
    ]

    for s in scenarios:
        result = assessor.assess(s["density_level"], s["behavior_status"], s["threat_result"])
        print(f"{s['label']}")
        print(f"  Inputs: density={s['density_level']}, behavior={s['behavior_status']}, "
              f"weapon={s['threat_result']['weapon_detected']}, altercation={s['threat_result']['possible_altercation']}")
        print(f"  -> Risk: {result['risk_level']}  (reason: {'; '.join(result['reasons'])})")
        print()


if __name__ == "__main__":
    main()