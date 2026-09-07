"""
utils/image_detection.py
------------------------
Image Detection Pipeline using trained YOLO11s model for AI Road Damage Detection.

Executes actual YOLO inference, extracts bounding boxes, confidence scores,
calculates application-based severity, and annotates images with high-contrast
visual indicators.
"""

import os
from typing import Tuple, List, Dict, Any
import numpy as np
import cv2
from PIL import Image

from utils.severity import calculate_severity
from utils.safety_recommendations import get_safety_recommendation

# Class color palette in BGR format for OpenCV
CLASS_COLORS_BGR = {
    "pothole": (40, 40, 235),    # Crimson Red
    "crack": (20, 160, 245),      # Amber / Warm Orange
    "manhole": (230, 145, 20),    # Cyan / Sky Blue
}

DEFAULT_COLOR_BGR = (180, 180, 40)


def draw_detection_box(
    img_bgr: np.ndarray,
    box: List[float],
    label: str,
    color_bgr: Tuple[int, int, int],
    severity: str
) -> None:
    """
    Draws a modern, professional styled bounding box with header badge.
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = img_bgr.shape[:2]

    # Constrain within image boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    # Box outline thickness scaled to image dimensions
    thickness = max(2, int(min(h, w) / 350))
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, thickness, cv2.LINE_AA)

    # Corner highlight accents for modern HUD feel
    corner_len = max(10, int(min(x2 - x1, y2 - y1) * 0.15))
    corner_thick = thickness + 2
    # Top-Left
    cv2.line(img_bgr, (x1, y1), (x1 + corner_len, y1), color_bgr, corner_thick)
    cv2.line(img_bgr, (x1, y1), (x1, y1 + corner_len), color_bgr, corner_thick)
    # Top-Right
    cv2.line(img_bgr, (x2, y1), (x2 - corner_len, y1), color_bgr, corner_thick)
    cv2.line(img_bgr, (x2, y1), (x2, y1 + corner_len), color_bgr, corner_thick)
    # Bottom-Left
    cv2.line(img_bgr, (x1, y2), (x1 + corner_len, y2), color_bgr, corner_thick)
    cv2.line(img_bgr, (x1, y2), (x1, y2 - corner_len), color_bgr, corner_thick)
    # Bottom-Right
    cv2.line(img_bgr, (x2, y2), (x2 - corner_len, y2), color_bgr, corner_thick)
    cv2.line(img_bgr, (x2, y2), (x2, y2 - corner_len), color_bgr, corner_thick)

    # Label text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(h, w) / 1000.0)
    font_thick = max(1, int(font_scale * 2))

    badge_text = f"{label} [{severity.upper()}]"
    (text_w, text_h), baseline = cv2.getTextSize(badge_text, font, font_scale, font_thick)

    # Badge background
    badge_y1 = max(0, y1 - text_h - 10)
    badge_y2 = y1 if y1 - text_h - 10 >= 0 else y1 + text_h + 10
    badge_x2 = min(w, x1 + text_w + 12)

    # Filled background rectangle for label
    cv2.rectangle(
        img_bgr,
        (x1, badge_y1),
        (badge_x2, badge_y2),
        color_bgr,
        -1
    )

    # Label text inside badge
    text_y = badge_y2 - 5 if badge_y1 == y1 - text_h - 10 else badge_y2 - baseline - 2
    cv2.putText(
        img_bgr,
        badge_text,
        (x1 + 6, text_y),
        font,
        font_scale,
        (255, 255, 255),
        font_thick,
        cv2.LINE_AA
    )


def detect_road_damage_image(
    model: Any,
    image_input: Any,
    conf_threshold: float = 0.25
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, int]]:
    """
    Executes road damage detection on an image using the trained YOLO11s model.

    Parameters:
        model: Loaded Ultralytics YOLO model.
        image_input: PIL Image or NumPy array (RGB or BGR).
        conf_threshold: Minimum confidence threshold (default: 0.25).

    Returns:
        annotated_image_rgb: RGB NumPy array of the annotated image.
        detections: List of dictionaries with details for each detected damage.
        summary_counts: Dictionary of aggregated damage and severity counts.
    """
    # Convert input to OpenCV BGR NumPy array for correct YOLO inference & drawing
    if isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.shape[2] == 4:
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
        else:
            # Assume BGR if coming from OpenCV, or RGB if coming from PIL/Matplotlib
            img_bgr = image_input.copy()
    else:
        raise ValueError("Unsupported image input format. Expected PIL Image or NumPy array.")

    img_h, img_w = img_bgr.shape[:2]
    total_img_area = float(img_h * img_w)

    # Run YOLO11s inference with BGR array (Ultralytics expects OpenCV BGR order for NumPy arrays)
    results = model.predict(source=img_bgr, conf=conf_threshold, verbose=False)

    detections: List[Dict[str, Any]] = []
    # Make a copy of BGR image for drawing annotations
    annotated_bgr = img_bgr.copy()

    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            for idx, box in enumerate(boxes):
                # 1. Damage Type (Class)
                cls_id = int(box.cls[0].item())
                raw_name = model.names.get(cls_id, f"Class_{cls_id}")
                damage_type = raw_name.strip().capitalize()

                # 2. Actual YOLO Confidence
                confidence = float(box.conf[0].item())
                conf_pct = round(confidence * 100, 1)

                # 3. Box coordinates & Area
                xyxy = [float(coord) for coord in box.xyxy[0].tolist()]
                box_w = max(0.0, xyxy[2] - xyxy[0])
                box_h = max(0.0, xyxy[3] - xyxy[1])
                box_area = box_w * box_h
                box_area_ratio = box_area / total_img_area if total_img_area > 0 else None

                # 4. Application-Based Severity Calculation
                severity = calculate_severity(
                    confidence=confidence,
                    damage_type=damage_type,
                    box_area_ratio=box_area_ratio
                )

                # 5. Rule-Based Safety Recommendation
                rec_info = get_safety_recommendation(
                    damage_type=damage_type,
                    severity=severity
                )

                detection_item = {
                    "id": idx + 1,
                    "damage_type": damage_type,
                    "confidence": confidence,
                    "confidence_pct": conf_pct,
                    "severity": severity,
                    "priority": rec_info["priority"],
                    "recommendation": rec_info["recommendation"],
                    "box": xyxy,
                    "box_area_ratio": box_area_ratio
                }
                detections.append(detection_item)

                # 6. Draw Bounding Box & Badge
                color_key = damage_type.lower()
                box_color = CLASS_COLORS_BGR.get(color_key, DEFAULT_COLOR_BGR)
                label_text = f"{damage_type} {conf_pct}%"

                draw_detection_box(
                    img_bgr=annotated_bgr,
                    box=xyxy,
                    label=label_text,
                    color_bgr=box_color,
                    severity=severity
                )

    # Convert back to RGB for Streamlit display
    annotated_image_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Calculate summary metrics directly from actual detections
    summary_counts = {
        "total": len(detections),
        "potholes": sum(1 for d in detections if d["damage_type"].lower() == "pothole"),
        "cracks": sum(1 for d in detections if d["damage_type"].lower() == "crack"),
        "manholes": sum(1 for d in detections if d["damage_type"].lower() == "manhole"),
        "high_severity": sum(1 for d in detections if d["severity"] == "High"),
        "medium_severity": sum(1 for d in detections if d["severity"] == "Medium"),
        "low_severity": sum(1 for d in detections if d["severity"] == "Low"),
    }

    return annotated_image_rgb, detections, summary_counts
