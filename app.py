import streamlit as st
import av
import numpy as np
import cv2
import mediapipe as mp

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Report generation
def generate_posture_pdf(posture_log):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica", 16)
    c.drawString(30, height - 50, "📝 Posture Monitoring Report")
    c.setFont("Helvetica", 12)
    c.drawString(30, height - 80, f"Total Observations: {len(posture_log)}")

    y = height - 120
    for i, entry in enumerate(posture_log[-30:]):
        text = f"{i+1}. Face: {'Yes' if entry['face'] else 'No'}, " \
               f"Centered: {'Yes' if entry['centered'] else 'No'}, " \
               f"Posture: {entry['posture_score']}%, Gaze: {entry['gaze']}"
        c.drawString(30, y, text)
        y -= 18
        if y < 40:
            c.showPage()
            y = height - 50

    c.save()
    buffer.seek(0)
    return buffer

# Video analysis logic
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose()
        self.face = mp_face.FaceDetection(min_detection_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
        self.gaze_direction = "Unknown"
        self.posture_log = []

    def estimate_gaze(self, landmarks, image_w):
        LEFT_EYE = [33, 133]
        RIGHT_EYE = [362, 263]
        pupil_x = landmarks[468].x * image_w

        left_corner = landmarks[LEFT_EYE[0]].x * image_w
        right_corner = landmarks[RIGHT_EYE[1]].x * image_w
        mid = (left_corner + right_corner) / 2

        if pupil_x < mid - 15:
            return "Looking Left"
        elif pupil_x > mid + 15:
            return "Looking Right"
        else:
            return "Looking Center"

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        height, width, _ = image.shape

        face_visible = False
        centered = False
        posture_score = 0

        results_pose = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]

            if width * 0.4 < (nose.x * width) < width * 0.6:
                centered = True

            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            posture_score = max(0, 100 - int(shoulder_diff * 100))

            mp.solutions.drawing_utils.draw_landmarks(
                image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        results_face = self.face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results_face.detections:
            face_visible = True
            for detection in results_face.detections:
                bbox = detection.location_data.relative_bounding_box
                x, y, w, h = int(bbox.xmin * width), int(bbox.ymin * height), \
                             int(bbox.width * width), int(bbox.height * height)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_mesh_results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if face_mesh_results.multi_face_landmarks:
            landmarks = face_mesh_results.multi_face_landmarks[0].landmark
            self.gaze_direction = self.estimate_gaze(landmarks, width)

        # Emoji overlay
        if posture_score > 80:
            emoji = "😊"
        elif posture_score < 50:
            emoji = "🙁"
        else:
            emoji = "😐"
        cv2.putText(image, emoji, (width - 60, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 255), 3, cv2.LINE_AA)

        # Draw text overlays
        cv2.putText(image, f"Face Visible: {'Yes' if face_visible else 'No'}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if face_visible else (0,0,255), 2)
        cv2.putText(image, f"Centered: {'Yes' if centered else 'No'}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if centered else (0,0,255), 2)
        cv2.putText(image, f"Posture Score: {posture_score}%", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.putText(image, f"Gaze: {self.gaze_direction}", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,100,255), 2)

        self.posture_log.append({
            "face": face_visible,
            "centered": centered,
            "posture_score": posture_score,
            "gaze": self.gaze_direction
        })

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Streamlit UI
st.set_page_config(page_title="Real-time Posture & Gaze Tracker", layout="wide")
st.title("📸 Real-time Posture and Face Gaze Dashboard")

ctx = webrtc_streamer(
    key="realtime-posture",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# Alerts and PDF download
if ctx.video_processor:
    logs = ctx.video_processor.posture_log
    if logs:
        last = logs[-1]
        alerts = []
        if not last["face"]:
            alerts.append("⚠️ No face detected!")
        if last["posture_score"] < 50:
            alerts.append("⚠️ Poor posture detected!")

        for alert in alerts:
            st.warning(alert)
            st.components.v1.html(f"""
            <script>
                var audio = new Audio('https://actions.google.com/sounds/v1/alarms/beep_short.ogg');
                audio.play();
            </script>
            """, height=0)

        st.markdown("---")
        st.subheader("📥 Download Posture Session Report")
        pdf_file = generate_posture_pdf(logs)
        st.download_button("Download PDF", data=pdf_file, file_name="posture_report.pdf", mime="application/pdf")
