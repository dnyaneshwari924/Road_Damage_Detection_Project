"""
utils/safety_recommendations.py
-------------------------------
Rule-based Safety Recommendations Module for AI Road Damage Detection System.

Generates explicit, rule-based safety recommendations directly from:
    Damage Type + Severity Level

No random or generated recommendations are used.
"""

from typing import Dict, Any


# Exact rule-based recommendations as specified in project requirements
RECOMMENDATIONS_REGISTRY: Dict[str, Dict[str, str]] = {
    "pothole": {
        "High": (
            "Immediate inspection and repair is recommended. "
            "Appropriate warning measures should be considered until repair."
        ),
        "Medium": "Schedule repair and continue monitoring the damaged road area.",
        "Low": "Monitor the pothole during routine road maintenance."
    },
    "crack": {
        "High": "Urgent inspection is recommended to determine the required maintenance.",
        "Medium": "Schedule inspection and maintenance of the affected road section.",
        "Low": "Monitor the crack during routine road inspection."
    },
    "manhole": {
        "High": (
            "Immediate inspection and securing of the area is recommended. "
            "Appropriate warning measures should be considered until the issue is resolved."
        ),
        "Medium": "Schedule inspection and maintenance of the manhole area.",
        "Low": "Monitor the manhole condition during routine road inspection."
    }
}


def get_safety_recommendation(damage_type: str, severity: str) -> Dict[str, Any]:
    """
    Retrieves the rule-based safety recommendation based on Damage Type and Severity.

    Parameters:
        damage_type (str): 'Pothole', 'Crack', or 'Manhole'
        severity (str): 'High', 'Medium', or 'Low'

    Returns:
        dict containing:
            - damage_type: normalized damage class name
            - severity: normalized severity level
            - recommendation: explicit safety recommendation text
            - priority: 'High', 'Medium', or 'Low'
            - urgency: description of required response time
    """
    dtype = str(damage_type).strip().lower()
    sev = str(severity).strip().capitalize()

    # Fallback to Medium if unknown severity
    if sev not in ["High", "Medium", "Low"]:
        sev = "Medium"

    # Match class in registry
    class_rules = RECOMMENDATIONS_REGISTRY.get(dtype)
    if class_rules and sev in class_rules:
        rec_text = class_rules[sev]
    else:
        # Fallback default rule
        if sev == "High":
            rec_text = "Immediate road inspection and maintenance is recommended."
        elif sev == "Medium":
            rec_text = "Schedule standard maintenance and monitor site condition."
        else:
            rec_text = "Monitor site condition during routine maintenance cycles."

    urgency_map = {
        "High": "Critical Action: Immediate (Within 24-48 Hours)",
        "Medium": "Planned Action: Scheduled (Within 1-2 Weeks)",
        "Low": "Routine Observation: Standard Maintenance Cycle"
    }

    return {
        "damage_type": damage_type.strip().capitalize(),
        "severity": sev,
        "recommendation": rec_text,
        "priority": sev,  # Priority matches Severity: High / Medium / Low
        "urgency": urgency_map.get(sev, "Standard Maintenance")
    }


def render_recommendation_card_html(
    damage_type: str,
    severity: str,
    confidence: float = None
) -> str:
    """
    Renders an attractive, professional HTML card for the safety recommendation.
    """
    info = get_safety_recommendation(damage_type, severity)
    sev = info["severity"]

    # Color definitions
    colors = {
        "High": {"border": "#EF4444", "bg": "rgba(239, 68, 68, 0.08)", "badge": "#EF4444", "icon": "🚨"},
        "Medium": {"border": "#F59E0B", "bg": "rgba(245, 158, 11, 0.08)", "badge": "#F59E0B", "icon": "⚠️"},
        "Low": {"border": "#10B981", "bg": "rgba(16, 185, 129, 0.08)", "badge": "#10B981", "icon": "ℹ️"}
    }
    theme = colors.get(sev, colors["Medium"])

    conf_display = f" • Confidence: <b>{confidence:.1f}%</b>" if confidence is not None else ""

    return f"""
    <div style="
        border-left: 5px solid {theme['border']};
        background: {theme['bg']};
        padding: 1.1rem 1.3rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
            <div style="font-weight: 700; font-size: 1.05rem; display: flex; align-items: center; gap: 8px;">
                <span>🛡️ Safety Recommendation</span>
            </div>
            <span style="
                background: {theme['badge']};
                color: #FFFFFF;
                padding: 3px 12px;
                border-radius: 20px;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.5px;
            ">
                PRIORITY: {info['priority'].upper()}
            </span>
        </div>
        <div style="font-size: 0.92rem; color: #4B5563; margin-bottom: 0.5rem;">
            Damage: <strong style="color: #111827;">{info['damage_type']}</strong>
            &nbsp;|&nbsp;
            Severity: <strong style="color: {theme['border']};">{info['severity']}</strong>
            {conf_display}
        </div>
        <div style="
            background: rgba(255, 255, 255, 0.85);
            padding: 0.8rem 1rem;
            border-radius: 6px;
            border: 1px solid rgba(0,0,0,0.06);
            color: #1F2937;
            font-size: 0.95rem;
            line-height: 1.5;
        ">
            <b>Recommendation:</b> {info['recommendation']}
        </div>
    </div>
    """
