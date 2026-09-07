"""
utils/video_detection.py
------------------------
Video Detection Pipeline using trained YOLO11s model for AI Road Damage Detection.
"""

import os
import subprocess
from typing import Dict, Any, Callable, Optional, List

import cv2
import numpy as np

from utils.severity import calculate_severity
from utils.safety_recommendations import get_safety_recommendation
from utils.image_detection import (
    draw_detection_box,
    CLASS_COLORS_BGR,
    DEFAULT_COLOR_BGR
)


# =========================================================
# CONVERT OUTPUT VIDEO TO BROWSER-FRIENDLY H.264 MP4
# =========================================================

def _convert_to_browser_mp4(
    input_path: str,
    output_path: str
) -> str:

    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg_exe,
            "-y",
            "-i",
            input_path,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            output_path,
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if (
            result.returncode == 0
            and os.path.exists(output_path)
        ):

            try:
                os.remove(input_path)
            except OSError:
                pass

            return output_path

    except Exception:
        pass

    return input_path


# =========================================================
# VIDEO PROCESSING
# =========================================================

def process_road_damage_video(
    model: Any,
    input_video_path: str,
    output_video_path: str,
    conf_threshold: float = 0.25,
    progress_callback: Optional[
        Callable[[int, int, Dict[str, int]], None]
    ] = None,
    max_frames: Optional[int] = None
) -> Dict[str, Any]:

    """
    Processes video frame-by-frame with YOLO11s detection.
    """

    # -----------------------------------------------------
    # CHECK INPUT
    # -----------------------------------------------------

    if not os.path.exists(input_video_path):

        raise FileNotFoundError(
            f"Input video not found: {input_video_path}"
        )

    output_dir = os.path.dirname(output_video_path)

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # -----------------------------------------------------
    # OPEN VIDEO
    # -----------------------------------------------------

    cap = cv2.VideoCapture(
        input_video_path
    )

    if not cap.isOpened():

        raise ValueError(
            f"Unable to open video file: "
            f"{input_video_path}"
        )

    # -----------------------------------------------------
    # VIDEO PROPERTIES
    # -----------------------------------------------------

    raw_total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if (
        fps <= 0
        or np.isnan(fps)
    ):

        fps = 25.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_area = float(
        width * height
    )

    if width <= 0 or height <= 0:

        cap.release()

        raise ValueError(
            "Invalid video dimensions."
        )

    # -----------------------------------------------------
    # TOTAL FRAMES
    # -----------------------------------------------------

    target_total = (
        raw_total_frames
        if raw_total_frames > 0
        else 100
    )

    if (
        max_frames
        and max_frames > 0
    ):

        target_total = min(
            target_total,
            max_frames
        )

    # =====================================================
    # OUTPUT VIDEO
    # =====================================================

    final_output_path = output_video_path

    intermediate_output_path = (
        os.path.splitext(
            output_video_path
        )[0]
        + "_opencv.mp4"
    )

    # -----------------------------------------------------
    # MP4V WRITER
    # -----------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        intermediate_output_path,
        fourcc,
        fps,
        (width, height)
    )

    # -----------------------------------------------------
    # FALLBACK AVI
    # -----------------------------------------------------

    if not writer.isOpened():

        intermediate_output_path = (
            os.path.splitext(
                output_video_path
            )[0]
            + "_opencv.avi"
        )

        fourcc_alt = (
            cv2.VideoWriter_fourcc(
                *"XVID"
            )
        )

        writer = cv2.VideoWriter(
            intermediate_output_path,
            fourcc_alt,
            fps,
            (width, height)
        )

    if not writer.isOpened():

        cap.release()

        raise ValueError(
            "Unable to create output video writer."
        )

    # =====================================================
    # STATISTICS
    # =====================================================

    stats = {

        "total_frames": target_total,

        "processed_frames": 0,

        "total_detections": 0,

        "potholes": 0,

        "cracks": 0,

        "manholes": 0,

        "high_severity": 0,

        "medium_severity": 0,

        "low_severity": 0
    }

    # =====================================================
    # SEVERITY TRACKING
    # =====================================================

    class_max_severity: Dict[
        str,
        str
    ] = {}

    severity_order = {

        "High": 3,

        "Medium": 2,

        "Low": 1
    }

    # =====================================================
    # SAMPLE FRAMES
    # =====================================================

    sample_frames: List[
        np.ndarray
    ] = []

    processed_count = 0

    # =====================================================
    # FRAME PROCESSING
    # =====================================================

    try:

        while cap.isOpened():

            ret, frame_bgr = cap.read()

            if (
                not ret
                or frame_bgr is None
            ):

                break

            processed_count += 1

            if (
                max_frames
                and processed_count > max_frames
            ):

                break

            # -------------------------------------------------
            # YOLO DETECTION
            # -------------------------------------------------

            results = model.predict(

                source=frame_bgr,

                conf=conf_threshold,

                verbose=False
            )

            frame_has_detections = False

            # -------------------------------------------------
            # RESULTS
            # -------------------------------------------------

            if (
                results
                and len(results) > 0
            ):

                boxes = results[0].boxes

                if (
                    boxes is not None
                    and len(boxes) > 0
                ):

                    frame_has_detections = True

                    # -----------------------------------------
                    # EACH DETECTION
                    # -----------------------------------------

                    for box in boxes:

                        # -------------------------------------
                        # CLASS
                        # -------------------------------------

                        cls_id = int(
                            box.cls[0].item()
                        )

                        damage_type = (
                            model.names.get(
                                cls_id,
                                f"Class_{cls_id}"
                            )
                            .strip()
                            .capitalize()
                        )

                        # -------------------------------------
                        # CONFIDENCE
                        # -------------------------------------

                        confidence = float(
                            box.conf[0].item()
                        )

                        conf_pct = round(
                            confidence * 100,
                            1
                        )

                        # -------------------------------------
                        # BOUNDING BOX
                        # -------------------------------------

                        xyxy = [
                            float(coord)
                            for coord
                            in box.xyxy[0].tolist()
                        ]

                        box_w = max(
                            0.0,
                            xyxy[2]
                            - xyxy[0]
                        )

                        box_h = max(
                            0.0,
                            xyxy[3]
                            - xyxy[1]
                        )

                        box_area_ratio = (

                            (
                                box_w
                                * box_h
                            )
                            / total_area

                            if total_area > 0

                            else None
                        )

                        # -------------------------------------
                        # SEVERITY
                        # -------------------------------------

                        severity = (
                            calculate_severity(

                                confidence=confidence,

                                damage_type=damage_type,

                                box_area_ratio=box_area_ratio
                            )
                        )

                        # -------------------------------------
                        # STATISTICS
                        # -------------------------------------

                        stats[
                            "total_detections"
                        ] += 1

                        dtype_lower = (
                            damage_type.lower()
                        )

                        if (
                            dtype_lower
                            == "pothole"
                        ):

                            stats[
                                "potholes"
                            ] += 1

                        elif (
                            dtype_lower
                            == "crack"
                        ):

                            stats[
                                "cracks"
                            ] += 1

                        elif (
                            dtype_lower
                            == "manhole"
                        ):

                            stats[
                                "manholes"
                            ] += 1

                        # -------------------------------------
                        # SEVERITY COUNTS
                        # -------------------------------------

                        if severity == "High":

                            stats[
                                "high_severity"
                            ] += 1

                        elif severity == "Medium":

                            stats[
                                "medium_severity"
                            ] += 1

                        elif severity == "Low":

                            stats[
                                "low_severity"
                            ] += 1

                        # -------------------------------------
                        # MAX SEVERITY
                        # -------------------------------------

                        current_severity_value = (
                            severity_order.get(

                                class_max_severity.get(
                                    dtype_lower,
                                    "Low"
                                ),

                                1
                            )
                        )

                        new_severity_value = (
                            severity_order.get(
                                severity,
                                1
                            )
                        )

                        if (

                            dtype_lower
                            not in class_max_severity

                            or

                            new_severity_value
                            > current_severity_value

                        ):

                            class_max_severity[
                                dtype_lower
                            ] = severity

                        # -------------------------------------
                        # DRAW DETECTION BOX
                        # -------------------------------------

                        color_key = (
                            damage_type.lower()
                        )

                        box_color = (
                            CLASS_COLORS_BGR.get(

                                color_key,

                                DEFAULT_COLOR_BGR
                            )
                        )

                        label_text = (
                            f"{damage_type} "
                            f"{conf_pct}%"
                        )

                        draw_detection_box(

                            img_bgr=frame_bgr,

                            box=xyxy,

                            label=label_text,

                            color_bgr=box_color,

                            severity=severity
                        )

            # -------------------------------------------------
            # WRITE FRAME
            # -------------------------------------------------

            writer.write(
                frame_bgr
            )

            # -------------------------------------------------
            # SAMPLE FRAMES
            # -------------------------------------------------

            if (

                frame_has_detections

                and

                len(sample_frames) < 3

            ):

                frame_rgb = cv2.cvtColor(

                    frame_bgr,

                    cv2.COLOR_BGR2RGB
                )

                sample_frames.append(
                    frame_rgb
                )

            # -------------------------------------------------
            # UPDATE PROGRESS
            # -------------------------------------------------

            stats[
                "processed_frames"
            ] = processed_count

            if progress_callback:

                progress_callback(

                    processed_count,

                    target_total,

                    stats
                )

    finally:

        cap.release()

        writer.release()

    # =====================================================
    # CONVERT TO H264 MP4
    # =====================================================

    converted_path = (
        _convert_to_browser_mp4(

            intermediate_output_path,

            final_output_path
        )
    )

    final_output_path = (
        converted_path
    )

    # =====================================================
    # SAFETY RECOMMENDATIONS
    # =====================================================

    recommendations_list: List[
        Dict[str, Any]
    ] = []

    for (
        cls_name,
        severity
    ) in class_max_severity.items():

        rec_info = (
            get_safety_recommendation(

                damage_type=cls_name,

                severity=severity
            )
        )

        recommendations_list.append(
            rec_info
        )

    # -----------------------------------------------------
    # SORT HIGH PRIORITY FIRST
    # -----------------------------------------------------

    recommendations_list.sort(

        key=lambda r:
        severity_order.get(

            r.get(
                "priority",
                "Low"
            ),

            0
        ),

        reverse=True
    )

    # =====================================================
    # FINAL STATS
    # =====================================================

    stats[
        "recommendations"
    ] = recommendations_list

    stats[
        "output_video_path"
    ] = final_output_path

    stats[
        "sample_frames"
    ] = sample_frames

    return stats