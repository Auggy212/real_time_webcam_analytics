from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import json
import os

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose

metrics_path = os.path.join("shared", "metrics.json")

def analyze_frame(image_np):
    h, w, _ = image_np.shape
    results = {}

    # Face detection
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        face = detector.process(image_np)
        face_visible = face.detections is not None and len(face.detections) > 0
        results["face_visible"] = face_visible

    # Pose estimation
    with mp_pose.Pose(static_image_mode=True) as pose_detector:
        pose = pose_detector.process(image_np)
        if pose.pose_landmarks:
            landmarks = pose.pose_landmarks.landmark

            # Example: check horizontal center for nose
            nose = landmarks[0]
            body_centered = 0.4 < nose.x < 0.6
            results["body_centered"] = body_centered

            # Optional: calculate simple eye/head movement as proxy
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            posture_variance = abs(left_shoulder.y - right_shoulder.y)
            results["posture_alignment_percent"] = round((1 - min(posture_variance, 0.5)) * 100, 2)
        else:
            results["body_centered"] = False
            results["posture_alignment_percent"] = 0.0

    return results

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['frame']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    metrics = analyze_frame(frame)

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    return jsonify({"status": "ok", "metrics": metrics})

@app.route('/metrics', methods=['GET'])
def get_metrics():
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "no data"}), 404

if __name__ == '__main__':
    os.makedirs("shared", exist_ok=True)
    app.run(host='0.0.0.0', port=8000, debug=True)
