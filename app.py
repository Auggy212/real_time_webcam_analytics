import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import mediapipe as mp
import pandas as pd
from collections import Counter

st.set_page_config(page_title="Webcam Posture & Gaze Tracker", layout="centered")

st.title("📸 Real-Time Posture & Gaze Tracker")
st.markdown("Tracks face visibility, gaze direction, posture centering, and logs session stats.")

# Mediapipe setup
mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Tracking stats
gaze_log = []
posture_log = []

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.pose = mp_pose.Pose()
        self.total_frames = 0
        self.centered_frames = 0

    def recv(self, frame):
        global gaze_log, posture_log

        image = frame.to_ndarray(format="bgr24")
        annotated_image = image.copy()
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.total_frames += 1

        # Face Detection
        face_results = self.face_detection.process(img_rgb)
        face_visible = False
        gaze = "Unknown"

        if face_results.detections:
            face_visible = True
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Estimate gaze direction (rough based on bounding box center)
                cx = x + w // 2
                if cx < iw * 0.4:
                    gaze = "Right"
                elif cx > iw * 0.6:
                    gaze = "Left"
                else:
                    gaze = "Center"

        else:
            cv2.putText(annotated_image, "Face Not Visible", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        gaze_log.append(gaze)

        # Posture Check
        posture_results = self.pose.process(img_rgb)
        posture_centered = False

        if posture_results.pose_landmarks:
            nose_x = posture_results.pose_landmarks.landmark[0].x
            if 0.4 < nose_x < 0.6:
                posture_centered = True
                self.centered_frames += 1
            mp_drawing.draw_landmarks(annotated_image, posture_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        posture_log.append("Centered" if posture_centered else "Not Centered")

        # Overlays
        cv2.putText(annotated_image, f"Gaze: {gaze}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(annotated_image, f"Posture: {'Centered' if posture_centered else 'Off-Center'}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return annotated_image

# Run video streamer
ctx = webrtc_streamer(
    key="tracker",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# Show stats
if ctx.video_processor:
    vp = ctx.video_processor
    if vp.total_frames > 0:
        st.metric("✅ Centered Posture %", f"{(vp.centered_frames / vp.total_frames) * 100:.2f}%")

# Gaze chart
if gaze_log:
    st.subheader("🔍 Gaze Direction Distribution")
    counts = Counter(gaze_log)
    df = pd.DataFrame.from_dict(counts, orient='index', columns=['Count'])
    st.bar_chart(df)

# Download Report
if st.button("📥 Download Session Report"):
    df_report = pd.DataFrame({
        "Gaze Direction": gaze_log,
        "Posture": posture_log
    })
    csv = df_report.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "session_report.csv", "text/csv", key='download-csv')
