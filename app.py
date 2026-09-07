import os
import io
import tempfile

import av
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from utils.image_detection import detect_road_damage_image
from utils.video_detection import process_road_damage_video


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DARK BLUE THEME
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(37, 99, 235, 0.18),
                transparent 35%
            ),
            radial-gradient(
                circle at bottom left,
                rgba(14, 165, 233, 0.10),
                transparent 30%
            ),
            #07111F;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #071A33 0%,
                #0A2342 50%,
                #071525 100%
            );
        border-right: 1px solid rgba(59,130,246,0.25);
    }

    section[data-testid="stSidebar"] * {
        color: #EAF2FF !important;
    }

    /* Main text */
    .stApp,
    .stApp p,
    .stApp label {
        color: #E8F1FF;
    }

    /* Headings */
    h1 {
        color: #60A5FA !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #3B82F6 !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #93C5FD !important;
        font-weight: 700 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(
            135deg,
            #2563EB,
            #0284C7
        );
        color: white !important;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.65rem 1rem;
        transition: 0.2s;
        box-shadow: 0 5px 20px rgba(37,99,235,0.25);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(37,99,235,0.40);
    }

    /* Download button */
    .stDownloadButton > button {
        background: #0F2D52;
        color: #BFDBFE !important;
        border: 1px solid #2563EB;
        border-radius: 10px;
        font-weight: 700;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(13, 32, 55, 0.75);
        border: 1px solid rgba(59,130,246,0.35);
        border-radius: 14px;
        padding: 8px;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: rgba(13, 32, 55, 0.80);
        border: 1px solid rgba(59,130,246,0.22);
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #93C5FD !important;
    }

    [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
    }

    /* Containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(13, 32, 55, 0.55);
        border: 1px solid rgba(59,130,246,0.22);
        border-radius: 14px;
    }

    /* Info */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Images */
    [data-testid="stImage"] img {
        border-radius: 12px;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Divider */
    hr {
        border-color: rgba(59,130,246,0.20) !important;
    }

    /* Camera */
    [data-testid="stVideo"] {
        border-radius: 14px;
        overflow: hidden;
    }

    /* Mobile */
    @media (max-width: 768px) {

        h1 {
            font-size: 2rem !important;
        }

        h2 {
            font-size: 1.5rem !important;
        }

        [data-testid="stMetric"] {
            margin-bottom: 8px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "detections" not in st.session_state:
    st.session_state.detections = []

if "image_result" not in st.session_state:
    st.session_state.image_result = None

if "image_counts" not in st.session_state:
    st.session_state.image_counts = None

if "video_stats" not in st.session_state:
    st.session_state.video_stats = None


# =========================================================
# MODEL
# =========================================================

MODEL_PATH = os.path.join(
    "models",
    "best.pt"
)


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return YOLO(MODEL_PATH)


model = load_model()


# =========================================================
# SAFETY RECOMMENDATIONS
# =========================================================

def get_recommendation(damage_type, severity):

    damage = str(damage_type).lower()
    sev = str(severity).lower()

    if "pothole" in damage:

        if sev == "high":
            return (
                "Immediately barricade or clearly mark the damaged "
                "area and notify the responsible road authority. "
                "Road users should reduce speed and maintain a safe "
                "distance while approaching the affected section."
            )

        elif sev == "medium":
            return (
                "The pothole should be inspected and repaired at the "
                "earliest opportunity. Drivers and riders should slow "
                "down and avoid sudden steering near the damaged area."
            )

        else:
            return (
                "Monitor the pothole condition and schedule preventive "
                "maintenance. Road users should remain alert and reduce "
                "speed when passing through the affected section."
            )

    if "crack" in damage:

        if sev == "high":
            return (
                "Arrange an urgent inspection of the cracked road "
                "surface. Restrict or carefully manage traffic if the "
                "crack appears to compromise road safety or structural "
                "condition."
            )

        elif sev == "medium":
            return (
                "Schedule road-surface inspection and maintenance. "
                "Monitor the crack for further expansion and advise "
                "road users to travel cautiously."
            )

        else:
            return (
                "Record and monitor the crack during routine road "
                "inspections. Preventive maintenance can help stop "
                "minor cracking from becoming more severe."
            )

    if "manhole" in damage:

        if sev == "high":
            return (
                "Immediately inspect and secure the manhole area. "
                "Ensure the cover is properly positioned and the "
                "surrounding road surface is safe for traffic."
            )

        elif sev == "medium":
            return (
                "Inspect the manhole cover and surrounding pavement "
                "and schedule corrective maintenance. Road users should "
                "approach the area carefully."
            )

        else:
            return (
                "Include the manhole in routine road-safety inspection "
                "and maintenance. Ensure the cover remains stable and "
                "properly aligned with the road surface."
            )

    return (
        "Inspect the detected road condition and schedule appropriate "
        "maintenance. Road users should proceed carefully in the "
        "affected area."
    )


# =========================================================
# LIVE CAMERA PROCESSOR
# =========================================================

class RoadDamageVideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.model = model
        self.confidence = confidence

        self.total_detections = 0
        self.potholes = 0
        self.cracks = 0
        self.manholes = 0

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        try:

            results = self.model.predict(
                source=img,
                conf=self.confidence,
                verbose=False
            )

            if results and len(results) > 0:

                result = results[0]

                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:

                    frame_potholes = 0
                    frame_cracks = 0
                    frame_manholes = 0

                    for box in boxes:

                        cls_id = int(
                            box.cls[0].item()
                        )

                        damage_type = str(
                            self.model.names.get(
                                cls_id,
                                "Unknown"
                            )
                        ).lower()

                        if damage_type == "pothole":
                            frame_potholes += 1

                        elif damage_type == "crack":
                            frame_cracks += 1

                        elif damage_type == "manhole":
                            frame_manholes += 1

                    self.total_detections += (
                        frame_potholes
                        + frame_cracks
                        + frame_manholes
                    )

                    self.potholes += frame_potholes
                    self.cracks += frame_cracks
                    self.manholes += frame_manholes

                annotated = result.plot()

            else:

                annotated = img

        except Exception:

            annotated = img

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🛣️ RoadGuard AI")
    st.caption("Road Damage & Safety Recommendation")

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Select Page",
        [
            "Home",
            "Image Detection",
            "Live Camera",
            "Video Detection",
            "Dashboard",
            "Safety Recommendations",
            "About",
        ],
        index=[
            "Home",
            "Image Detection",
            "Live Camera",
            "Video Detection",
            "Dashboard",
            "Safety Recommendations",
            "About",
        ].index(st.session_state.page),
    )

    st.session_state.page = page

    st.divider()

    st.markdown("### Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
    )

    st.divider()

    if model is not None:
        st.success("🟢 YOLO11s Model Ready")
    else:
        st.error("🔴 Model Not Found")

    st.caption("Model: YOLO11s")
    st.caption("Classes: Pothole • Crack • Manhole")


# =========================================================
# MODEL ERROR
# =========================================================

if model is None:

    st.error(
        "Model file not found. Please make sure your trained model exists at:"
    )

    st.code(
        "models/best.pt"
    )

    st.stop()


# =========================================================
# HOME
# =========================================================

if page == "Home":

    st.title(
        "🛣️ AI Road Damage Detection & Safety Intelligence"
    )

    st.subheader(
        "Intelligent Road Inspection using YOLO11s"
    )

    st.write(
        "Detect potholes, cracks and manholes from road images "
        "and videos, analyse their severity and generate practical "
        "road-safety recommendations."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Detection Classes",
            "3"
        )

    with col2:
        st.metric(
            "AI Model",
            "YOLO11s"
        )

    with col3:
        st.metric(
            "System",
            "AI Powered"
        )

    st.divider()

    st.subheader("🚀 System Capabilities")

    c1, c2 = st.columns(2)

    with c1:

        with st.container(border=True):

            st.markdown("### 📷 Image Detection")

            st.write(
                "Upload a road image and detect potholes, cracks "
                "and manholes with confidence scores."
            )

    with c2:

        with st.container(border=True):

            st.markdown("### 📹 Live Camera")

            st.write(
                "Use your webcam for real-time road damage "
                "detection using YOLO11s."
            )

    c3, c4 = st.columns(2)

    with c3:

        with st.container(border=True):

            st.markdown("### 🎥 Video Detection")

            st.write(
                "Process road inspection videos and analyse detected "
                "road-damage conditions."
            )

    with c4:

        with st.container(border=True):

            st.markdown("### 🛡️ Safety Intelligence")

            st.write(
                "Generate practical recommendations based on damage "
                "type and severity."
            )


# =========================================================
# IMAGE DETECTION
# =========================================================

elif page == "Image Detection":

    st.title("📷 Road Damage  Detection  And Safety Recommendation")

    st.write(
        "Upload a road image to analyse road damage."
    )

    uploaded_image = st.file_uploader(
        "Upload Road Image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    if uploaded_image is not None:

        image = Image.open(
            uploaded_image
        ).convert("RGB")

        st.divider()

        if st.button(
            "🔍 DETECT ROAD DAMAGE",
            use_container_width=True,
        ):

            with st.spinner(
                "YOLO11s is analysing the image..."
            ):

                try:

                    annotated_img, detections, counts = (
                        detect_road_damage_image(
                            model=model,
                            image_input=image,
                            conf_threshold=confidence,
                        )
                    )

                    if isinstance(
                        annotated_img,
                        np.ndarray,
                    ):

                        if len(
                            annotated_img.shape
                        ) == 3:

                            annotated_pil = Image.fromarray(
                                annotated_img
                            )

                        else:

                            annotated_pil = Image.fromarray(
                                annotated_img
                            )

                    else:

                        annotated_pil = annotated_img

                    st.session_state.image_result = annotated_pil

                    st.session_state.detections = (
                        detections
                        if detections
                        else []
                    )

                    st.session_state.image_counts = (
                        counts
                    )

                    st.success(
                        "✅ Road damage detection completed."
                    )

                except Exception as e:

                    st.error(
                        f"Image processing error: {e}"
                    )

        if st.session_state.image_result is not None:

            st.divider()

            st.subheader(
                "📊 Detection Result"
            )

            input_col, result_col = st.columns(
                2,
                gap="large",
            )

            with input_col:

                st.markdown(
                    "### 📥 Input Image"
                )

                st.image(
                    image,
                    width=520,
                )

            with result_col:

                st.markdown(
                    "### 🎯 Detected Result"
                )

                st.image(
                    st.session_state.image_result,
                    width=520,
                )

            result_buffer = io.BytesIO()

            st.session_state.image_result.save(
                result_buffer,
                format="PNG",
            )

            result_buffer.seek(0)

            st.download_button(
                "⬇️ Download Detection Result",
                data=result_buffer.getvalue(),
                file_name="road_damage_detection.png",
                mime="image/png",
                use_container_width=True,
            )

            st.divider()

            st.subheader(
                "📌 Detection Summary"
            )

            counts = (
                st.session_state.image_counts
                or {}
            )

            total = int(
                counts.get(
                    "total",
                    len(
                        st.session_state.detections
                    ),
                )
            )

            potholes = int(
                counts.get(
                    "potholes",
                    0,
                )
            )

            cracks = int(
                counts.get(
                    "cracks",
                    0,
                )
            )

            manholes = int(
                counts.get(
                    "manholes",
                    0,
                )
            )

            high = int(
                counts.get(
                    "high_severity",
                    0,
                )
            )

            medium = int(
                counts.get(
                    "medium_severity",
                    0,
                )
            )

            low = int(
                counts.get(
                    "low_severity",
                    0,
                )
            )

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.metric(
                    "Total",
                    total,
                )

            with m2:
                st.metric(
                    "🕳️ Potholes",
                    potholes,
                )

            with m3:
                st.metric(
                    "〰️ Cracks",
                    cracks,
                )

            with m4:
                st.metric(
                    "⭕ Manholes",
                    manholes,
                )

            s1, s2, s3 = st.columns(3)

            with s1:
                st.metric(
                    "🔴 High",
                    high,
                )

            with s2:
                st.metric(
                    "🟠 Medium",
                    medium,
                )

            with s3:
                st.metric(
                    "🟢 Low",
                    low,
                )

            st.divider()

            st.subheader(
                "🛡️ Safety Recommendations"
            )

            detections = (
                st.session_state.detections
            )

            if detections:

                best_detections = {}

                for detection in detections:

                    damage_type = str(
                        detection.get(
                            "damage_type",
                            "Unknown",
                        )
                    )

                    confidence_pct = float(
                        detection.get(
                            "confidence_pct",
                            0,
                        )
                    )

                    if (
                        damage_type
                        not in best_detections
                    ):

                        best_detections[
                            damage_type
                        ] = detection

                    else:

                        old_confidence = float(
                            best_detections[
                                damage_type
                            ].get(
                                "confidence_pct",
                                0,
                            )
                        )

                        if (
                            confidence_pct
                            >
                            old_confidence
                        ):

                            best_detections[
                                damage_type
                            ] = detection

                for (
                    damage_type,
                    detection,
                ) in best_detections.items():

                    confidence_pct = float(
                        detection.get(
                            "confidence_pct",
                            0,
                        )
                    )

                    severity = str(
                        detection.get(
                            "severity",
                            "Low",
                        )
                    )

                    priority = str(
                        detection.get(
                            "priority",
                            "Low",
                        )
                    )

                    recommendation = (
                        get_recommendation(
                            damage_type,
                            severity,
                        )
                    )

                    if (
                        severity.lower()
                        == "high"
                    ):

                        severity_icon = "🔴"

                    elif (
                        severity.lower()
                        == "medium"
                    ):

                        severity_icon = "🟠"

                    else:

                        severity_icon = "🟢"

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"### 🛣️ {damage_type}"
                        )

                        c1, c2, c3 = st.columns(3)

                        with c1:

                            st.metric(
                                "🎯 Confidence",
                                f"{confidence_pct:.2f}%",
                            )

                        with c2:

                            st.metric(
                                "Severity",
                                f"{severity_icon} {severity}",
                            )

                        with c3:

                            st.metric(
                                "🚨 Priority",
                                priority,
                            )

                        st.markdown(
                            "#### 🛡️ Recommended Action"
                        )

                        st.info(
                            recommendation
                        )

            else:

                st.success(
                    "✅ No road damage detected."
                )


# =========================================================
# LIVE CAMERA
# =========================================================

elif page == "Live Camera":

    st.title(
        "📹 Live Road Damage Detection"
    )

    st.write(
        "Use your webcam for real-time road damage detection "
        "using the trained YOLO11s model."
    )

    st.info(
        "📌 Click START and allow camera permission in your browser."
    )

    st.divider()

    ctx = webrtc_streamer(
        key="roadguard-live-camera",
        video_processor_factory=RoadDamageVideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    st.divider()

    st.subheader(
        "🎯 Live Detection"
    )

    st.write(
        f"Current Confidence Threshold: "
        f"**{confidence:.2f}**"
    )

    if ctx.video_processor:

        processor = ctx.video_processor

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Total",
                processor.total_detections,
            )

        with c2:
            st.metric(
                "🕳️ Potholes",
                processor.potholes,
            )

        with c3:
            st.metric(
                "〰️ Cracks",
                processor.cracks,
            )

        with c4:
            st.metric(
                "⭕ Manholes",
                processor.manholes,
            )

        st.divider()

        st.subheader(
            "🛡️ Safety Guidance"
        )

        st.info(
            "🕳️ **Pothole:** Slow down and maintain a safe distance "
            "from the affected road section."
        )

        st.warning(
            "〰️ **Crack:** The damaged section should be inspected "
            "and monitored for further expansion."
        )

        st.info(
            "⭕ **Manhole:** Inspect the cover and surrounding road "
            "surface to ensure safe passage."
        )

    else:

        st.info(
            "📹 Click START above to activate your webcam."
        )


# =========================================================
# VIDEO DETECTION
# =========================================================

elif page == "Video Detection":

    st.title(
        "🎥 Road Damage Video Detection"
    )

    st.write(
        "Upload a road inspection video for frame-by-frame "
        "YOLO11s detection."
    )

    uploaded_video = st.file_uploader(
        "Upload Road Video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv",
            "webm",
        ],
    )

    max_frames = st.number_input(
        "Maximum Frames (0 = Full Video)",
        min_value=0,
        value=0,
        step=30,
    )

    if uploaded_video is not None:

        if st.button(
            "🎥 PROCESS VIDEO",
            use_container_width=True,
        ):

            input_suffix = os.path.splitext(
                uploaded_video.name
            )[1]

            if not input_suffix:
                input_suffix = ".mp4"

            temp_input = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=input_suffix,
            )

            temp_output = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4",
            )

            try:

                temp_input.write(
                    uploaded_video.getbuffer()
                )

                temp_input.close()

                temp_output.close()

                progress_bar = st.progress(
                    0
                )

                status_text = st.empty()

                def update_progress(
                    current,
                    total,
                    stats=None,
                ):

                    if total and total > 0:

                        progress = min(
                            current / total,
                            1.0,
                        )

                        progress_bar.progress(
                            progress
                        )

                        status_text.write(
                            f"Processing frames: "
                            f"{current}/{total}"
                        )

                with st.spinner(
                    "Processing video..."
                ):

                    stats = (
                        process_road_damage_video(
                            model=model,
                            input_video_path=temp_input.name,
                            output_video_path=temp_output.name,
                            conf_threshold=confidence,
                            progress_callback=update_progress,
                            max_frames=(
                                int(max_frames)
                                if max_frames > 0
                                else None
                            ),
                        )
                    )

                progress_bar.progress(
                    1.0
                )

                status_text.success(
                    "✅ Video processing completed."
                )

                st.session_state.video_stats = (
                    stats
                )

                output_path = stats.get(
                    "output_video_path",
                    temp_output.name,
                )

                if os.path.exists(
                    output_path
                ):

                    st.divider()

                    st.subheader(
                        "🎬 Processed Video"
                    )

                    with open(
                        output_path,
                        "rb",
                    ) as video_file:

                        video_bytes = (
                            video_file.read()
                        )

                    st.video(
                        video_bytes
                    )

                    st.download_button(
                        "⬇️ Download Processed Video",
                        data=video_bytes,
                        file_name="road_damage_detected.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )

                st.divider()

                st.subheader(
                    "📊 Video Detection Statistics"
                )

                stats = (
                    st.session_state.video_stats
                    or {}
                )

                v1, v2, v3, v4 = st.columns(4)

                with v1:

                    st.metric(
                        "Total Detections",
                        int(
                            stats.get(
                                "total_detections",
                                0,
                            )
                        ),
                    )

                with v2:

                    st.metric(
                        "🕳️ Potholes",
                        int(
                            stats.get(
                                "potholes",
                                0,
                            )
                        ),
                    )

                with v3:

                    st.metric(
                        "〰️ Cracks",
                        int(
                            stats.get(
                                "cracks",
                                0,
                            )
                        ),
                    )

                with v4:

                    st.metric(
                        "⭕ Manholes",
                        int(
                            stats.get(
                                "manholes",
                                0,
                            )
                        ),
                    )

                a1, a2, a3 = st.columns(3)

                with a1:

                    st.metric(
                        "🔴 High Severity",
                        int(
                            stats.get(
                                "high_severity",
                                0,
                            )
                        ),
                    )

                with a2:

                    st.metric(
                        "🟠 Medium Severity",
                        int(
                            stats.get(
                                "medium_severity",
                                0,
                            )
                        ),
                    )

                with a3:

                    st.metric(
                        "🟢 Low Severity",
                        int(
                            stats.get(
                                "low_severity",
                                0,
                            )
                        ),
                    )

            except Exception as e:

                st.error(
                    f"Video processing error: {e}"
                )

            finally:

                try:

                    if os.path.exists(
                        temp_input.name
                    ):

                        os.remove(
                            temp_input.name
                        )

                except Exception:
                    pass


# =========================================================
# DASHBOARD
# =========================================================

elif page == "Dashboard":

    st.title(
        "📊 Road Damage Analytics Dashboard"
    )

    detections = (
        st.session_state.detections
    )

    if not detections:

        st.info(
            "ℹ️ Run an image detection first "
            "to populate the dashboard."
        )

    else:

        total = len(
            detections
        )

        potholes = sum(
            1
            for d in detections
            if "pothole"
            in str(
                d.get(
                    "damage_type",
                    ""
                )
            ).lower()
        )

        cracks = sum(
            1
            for d in detections
            if "crack"
            in str(
                d.get(
                    "damage_type",
                    ""
                )
            ).lower()
        )

        manholes = sum(
            1
            for d in detections
            if "manhole"
            in str(
                d.get(
                    "damage_type",
                    ""
                )
            ).lower()
        )

        high = sum(
            1
            for d in detections
            if str(
                d.get(
                    "severity",
                    ""
                )
            ).lower()
            == "high"
        )

        medium = sum(
            1
            for d in detections
            if str(
                d.get(
                    "severity",
                    ""
                )
            ).lower()
            == "medium"
        )

        low = sum(
            1
            for d in detections
            if str(
                d.get(
                    "severity",
                    ""
                )
            ).lower()
            == "low"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Total Detections",
                total,
            )

        with c2:
            st.metric(
                "🕳️ Potholes",
                potholes,
            )

        with c3:
            st.metric(
                "〰️ Cracks",
                cracks,
            )

        with c4:
            st.metric(
                "⭕ Manholes",
                manholes,
            )

        st.divider()

        st.subheader(
            "📈 Damage Distribution"
        )

        damage_df = pd.DataFrame(
            {
                "Damage Type": [
                    "Pothole",
                    "Crack",
                    "Manhole",
                ],
                "Detections": [
                    potholes,
                    cracks,
                    manholes,
                ],
            }
        )

        st.bar_chart(
            damage_df.set_index(
                "Damage Type"
            )
        )

        st.subheader(
            "🚨 Severity Distribution"
        )

        severity_df = pd.DataFrame(
            {
                "Severity": [
                    "High",
                    "Medium",
                    "Low",
                ],
                "Detections": [
                    high,
                    medium,
                    low,
                ],
            }
        )

        st.bar_chart(
            severity_df.set_index(
                "Severity"
            )
        )

        st.subheader(
            "📋 Detection Details"
        )

        table_data = []

        for d in detections:

            table_data.append(
                {
                    "ID": d.get(
                        "id",
                        "-",
                    ),
                    "Damage Type": d.get(
                        "damage_type",
                        "Unknown",
                    ),
                    "Confidence": (
                        f"{float(d.get('confidence_pct', 0)):.2f}%"
                    ),
                    "Severity": d.get(
                        "severity",
                        "Low",
                    ),
                    "Priority": d.get(
                        "priority",
                        "Low",
                    ),
                }
            )

        df = pd.DataFrame(
            table_data
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# SAFETY RECOMMENDATIONS PAGE
# =========================================================

elif page == "Safety Recommendations":

    st.title(
        "🛡️ Road Safety Recommendations"
    )

    st.write(
        "Recommendations are generated according to detected "
        "road-damage type and severity."
    )

    detections = (
        st.session_state.detections
    )

    if not detections:

        st.info(
            "ℹ️ Run an image detection first "
            "to generate recommendations."
        )

    else:

        best_detections = {}

        for detection in detections:

            damage_type = str(
                detection.get(
                    "damage_type",
                    "Unknown",
                )
            )

            confidence_pct = float(
                detection.get(
                    "confidence_pct",
                    0,
                )
            )

            if (
                damage_type
                not in best_detections
            ):

                best_detections[
                    damage_type
                ] = detection

            elif (
                confidence_pct
                >
                float(
                    best_detections[
                        damage_type
                    ].get(
                        "confidence_pct",
                        0,
                    )
                )
            ):

                best_detections[
                    damage_type
                ] = detection

        for (
            damage_type,
            detection,
        ) in best_detections.items():

            confidence_pct = float(
                detection.get(
                    "confidence_pct",
                    0,
                )
            )

            severity = str(
                detection.get(
                    "severity",
                    "Low",
                )
            )

            priority = str(
                detection.get(
                    "priority",
                    "Low",
                )
            )

            recommendation = (
                get_recommendation(
                    damage_type,
                    severity,
                )
            )

            if (
                severity.lower()
                == "high"
            ):

                icon = "🔴"

            elif (
                severity.lower()
                == "medium"
            ):

                icon = "🟠"

            else:

                icon = "🟢"

            with st.container(
                border=True
            ):

                st.markdown(
                    f"### 🛣️ {damage_type}"
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "🎯 Confidence",
                        f"{confidence_pct:.2f}%",
                    )

                with c2:

                    st.metric(
                        "Severity",
                        f"{icon} {severity}",
                    )

                with c3:

                    st.metric(
                        "🚨 Priority",
                        priority,
                    )

                st.markdown(
                    "#### 🛡️ Safety Recommendation"
                )

                st.info(
                    recommendation
                )


# =========================================================
# ABOUT
# =========================================================

elif page == "About":

    st.title(
        "ℹ️ About RoadGuard AI"
    )

    st.write(
        "RoadGuard AI is an AI-based road inspection and safety "
        "intelligence system designed to identify common road "
        "damage conditions from images and videos."
    )

    st.divider()

    st.subheader(
        "🤖 Artificial Intelligence"
    )

    st.write(
        "The system uses a trained Ultralytics YOLO11s object "
        "detection model for road-damage identification."
    )

    st.subheader(
        "🔎 Detection Classes"
    )

    classes = pd.DataFrame(
        {
            "Class": [
                "Pothole",
                "Crack",
                "Manhole",
            ],
            "Purpose": [
                "Detect road surface potholes",
                "Detect visible road cracks",
                "Detect manhole-related road conditions",
            ],
        }
    )

    st.dataframe(
        classes,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "⚙️ Technology Stack"
    )

    tech = [
        "Python",
        "Streamlit",
        "Ultralytics YOLO11s",
        "OpenCV",
        "Pillow",
        "NumPy",
        "Pandas",
        "Streamlit-WebRTC",
    ]

    for item in tech:

        st.write(
            f"• {item}"
        )

    st.subheader(
        "🔄 System Workflow"
    )

    workflow = [
        "1. Upload road image or video",
        "2. YOLO11s analyses the input",
        "3. Road damage objects are detected",
        "4. Confidence scores are calculated",
        "5. Severity and priority are determined",
        "6. Safety recommendations are generated",
        "7. Results are presented through the dashboard",
        "8. Webcam can be used for real-time detection",
    ]

    for step in workflow:

        st.write(
            step
        )

    st.divider()

    st.success(
        "🛣️ RoadGuard AI — AI-powered road inspection "
        "and safety intelligence."
    )
