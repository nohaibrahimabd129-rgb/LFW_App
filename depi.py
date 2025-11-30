
import streamlit as st
import cv2
import face_recognition
import numpy as np
from PIL import Image
import pickle

# Load saved model
with open("face_model.pkl", "rb") as f:
    data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]

# Page Title
st.title("✨ Face Recognition System")
st.write("Choose mode: Live Camera or Upload Image")

# Radio Mode
mode = st.radio("Select Mode:", ["🎥 Live Camera", "🖼 Upload Image"])

# Live Camera Mode
if mode == "🎥 Live Camera":
    start_camera = st.checkbox("Start Camera")

    if start_camera:
        cap = cv2.VideoCapture(0)
        frame_view = st.empty()
        stop = st.button("⛔ Stop Camera")

        while start_camera and not stop:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = frame[:, :, ::-1]
            locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, locations)

            for (top, right, bottom, left), encoding in zip(locations, encodings):
                matches = face_recognition.compare_faces(known_encodings, encoding)
                name = "Unknown"
                if True in matches:
                    name = known_names[matches.index(True)]

                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            frame_view.image(frame, channels="BGR")

        cap.release()

# Upload Image Mode
elif mode == "🖼 Upload Image":
    uploaded = st.file_uploader("Upload Image:", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = np.array(Image.open(uploaded))
        st.image(img, caption="Input Image", width=600)

        rgb = img[:, :, ::-1]
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        if len(encodings) > 0:
            for (top, right, bottom, left), encoding in zip(locations, encodings):
                matches = face_recognition.compare_faces(known_encodings, encoding)
                name = "Unknown"
                if True in matches:
                    name = known_names[matches.index(True)]

                cv2.rectangle(img, (left, top), (right, bottom), (0,255,0), 2)
                cv2.putText(img, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            st.image(img, caption=f"Detected: {name}", width=600)
        else:
            st.warning("⚠ No face detected!")
