"""
utils/severity.py
----------------
Application-based Severity Classification Module for AI Road Damage Detection.

IMPORTANT ARCHITECTURAL NOTE:
YOLO11s is an object detection model that identifies damage locations, classes
(Pothole, Crack, Manhole), and detection confidence scores. It does NOT directly
predict severity levels.

This module provides the application-level logic to derive severity classifications:
Workflow:
    YOLO11s Detection -> Damage Type + Confidence Score -> Application-Based Severity
"""

from typing import Optional


def calculate_severity(
    confidence: float,
    damage_type: str,
    box_area_ratio: Optional[float] = None
) -> str:
    """
    Calculates application-based severity level ('Low', 'Medium', 'High')
    using YOLO detection details.

    Parameters:
        confidence (float): Detection confidence score between 0.0 and 1.0.
        damage_type (str): Detected damage class ('Pothole', 'Crack', 'Manhole').
        box_area_ratio (float, optional): Ratio of bounding box area to total image area.

    Returns:
        str: Severity level ('Low', 'Medium', 'High').
    """
    dtype = str(damage_type).strip().lower()

    # Pothole Severity Rules:
    # Potholes represent hollow road cavitations. High confidence or large footprint
    # indicates dangerous structural compromise.
    if dtype == "pothole":
        if box_area_ratio is not None and box_area_ratio >= 0.06:
            return "High"
        if confidence >= 0.70:
            return "High"
        elif confidence >= 0.45:
            return "Medium"
        else:
            return "Low"

    # Crack Severity Rules:
    # Cracks indicate asphalt fatigue or reflective cracking.
    elif dtype == "crack":
        if box_area_ratio is not None and box_area_ratio >= 0.08:
            return "High"
        if confidence >= 0.75:
            return "High"
        elif confidence >= 0.50:
            return "Medium"
        else:
            return "Low"

    # Manhole Severity Rules:
    # Manholes represent surface grade irregularities or cover displacements.
    elif dtype == "manhole":
        if box_area_ratio is not None and box_area_ratio >= 0.07:
            return "High"
        if confidence >= 0.75:
            return "High"
        elif confidence >= 0.50:
            return "Medium"
        else:
            return "Low"

    # Generic fallback based strictly on confidence
    else:
        if confidence >= 0.75:
            return "High"
        elif confidence >= 0.50:
            return "Medium"
        else:
            return "Low"


def get_severity_color(severity: str) -> str:
    """
    Returns hex color code corresponding to the severity level for UI rendering.
    """
    sev = str(severity).strip().lower()
    if sev == "high":
        return "#EF4444"  # Red / Crimson
    elif sev == "medium":
        return "#F59E0B"  # Amber / Orange
    elif sev == "low":
        return "#10B981"  # Emerald / Green
    return "#6B7280"      # Muted Gray fallback


def get_severity_badge_html(severity: str) -> str:
    """
    Returns an HTML badge component styled according to severity level.
    """
    color = get_severity_color(severity)
    sev_upper = severity.upper()
    return (
        f'<span style="background-color: {color}22; color: {color}; '
        f'padding: 3px 10px; border-radius: 9999px; font-weight: 600; '
        f'font-size: 0.82rem; border: 1px solid {color}55;">'
        f'{sev_upper}</span>'
    )
