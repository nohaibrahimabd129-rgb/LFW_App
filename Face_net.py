import streamlit as st
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from keras_facenet import FaceNet
import joblib

# Load models
embedder = FaceNet()
svm_model = joblib.load("svm_model.pkl")
detector = MTCNN()

# -----------------------------
# FACE EXTRACTION
# -----------------------------
def extract_face(image, required_size=(160, 160)):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img)

    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]['box']
    face = img[y:y + h, x:x + w]
    face = cv2.resize(face, required_size)

    return face, faces[0]['box']


# -----------------------------
# GET EMBEDDING
# -----------------------------
def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    face_pixels = np.expand_dims(face_pixels, axis=0)
    embedding = embedder.embeddings(face_pixels)
    return embedding[0]


# -----------------------------
# FACE RECOGNITION PIPELINE
# -----------------------------
def recognize_face(image):
    face, box = extract_face(image)

    if face is None:
        return None, None, None

    embedding = get_embedding(face)
    embedding = embedding.reshape(1, -1)

    prediction = svm_model.predict(embedding)
    label = prediction[0]
    confidence = np.max(svm_model.predict_proba(embedding))

    return label, confidence, box


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Face Recognition System 👤✨")
mode = st.radio("Select Mode:", ["Upload Image", "Live Camera"])


# ============================
# MODE 1: UPLOAD IMAGE
# ============================
if mode == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = np.array(Image.open(uploaded_image))

        # Recognize face
        label, confidence, box = recognize_face(image)

        # Prepare output image
        output_image = image.copy()
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Side-by-side display
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.image(output_image, caption="Detected Face", use_container_width=True)

        # Text result under images
        if label is None:
            st.error("❌ No face detected!")
        else:
            st.success(f"Identified Person: **{label}**")
            st.info(f"Confidence Score: **{confidence:.2f}**")


# ============================
# MODE 2: LIVE CAMERA
# ============================
if mode == "Live Camera":
    st.write("Turn on your camera and capture an image:")
    cam = st.camera_input("Camera")

    if cam is not None:
        image = np.array(Image.open(cam))

        label, confidence, box = recognize_face(image)

        output_image = image.copy()
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0,255,0), 2)

        # Side-by-side view
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Captured Image", use_container_width=True)

        with col2:
            st.image(output_image, caption="Detected Face", use_container_width=True)

        # Text result
        if label is None:
            st.error("❌ No face detected!")
        else:
            st.success(f"Identified Person: **{label}**")
            st.info(f"Confidence Score: **{confidence:.2f}**")
