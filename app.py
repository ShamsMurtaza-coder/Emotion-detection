import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Emotion Detector", page_icon="😊", layout="centered")
st.title("😊 Real-Time Facial Emotion Detector")
st.markdown("Upload a photo or use your webcam to detect emotions!")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/emotion_model.h5')

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = load_model()
face_cascade = load_face_cascade()

EMOTIONS = ['Angry 😠', 'Disgust 🤢', 'Fear 😨', 'Happy 😄', 'Sad 😢', 'Surprise 😮', 'Neutral 😐']

def predict_emotion(face_img):
    face = cv2.resize(face_img, (48, 48))
    if len(face.shape) == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=[0, -1])
    preds = model.predict(face, verbose=0)[0]
    return preds

def process_image(image):
    img_array = np.array(image.convert('RGB'))
    img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces     = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    result_img = img_bgr.copy()
    all_preds  = []
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        preds    = predict_emotion(face_roi)
        emotion  = EMOTIONS[np.argmax(preds)]
        conf     = np.max(preds) * 100
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result_img, f"{emotion} {conf:.0f}%",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        all_preds.append((emotion, preds))
    return cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), all_preds, len(faces)

tab1, tab2 = st.tabs(["📷 Upload Image", "📸 Use Webcam"])

with tab1:
    uploaded = st.file_uploader("Choose an image", type=['jpg','jpeg','png'])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Detecting emotions..."):
            result_img, predictions, face_count = process_image(image)
        if face_count == 0:
            st.warning("⚠️ No faces detected. Try a clearer front-facing photo.")
        else:
            st.success(f"✅ Detected {face_count} face(s)!")
            st.image(result_img, caption="Result", use_column_width=True)
            for i, (emotion, preds) in enumerate(predictions):
                st.markdown(f"**Face {i+1}: {emotion}**")
                for emo, prob in zip(EMOTIONS, preds):
                    st.progress(float(prob), text=f"{emo}: {prob*100:.1f}%")

with tab2:
    st.info("📸 Click below to take a photo and detect your emotion!")
    cam_image = st.camera_input("Take a photo")
    if cam_image:
        image = Image.open(cam_image)
        with st.spinner("Detecting emotions..."):
            result_img, predictions, face_count = process_image(image)
        if face_count == 0:
            st.warning("⚠️ No face detected. Try again with better lighting.")
        else:
            st.success(f"✅ {face_count} face(s) detected!")
            st.image(result_img, use_column_width=True)
            for i, (emotion, preds) in enumerate(predictions):
                st.markdown(f"**Face {i+1}: {emotion}**")
                for emo, prob in zip(EMOTIONS, preds):
                    st.progress(float(prob), text=f"{emo}: {prob*100:.1f}%")

st.markdown("---")
st.markdown("*Built with TensorFlow, OpenCV & Streamlit — BTech AI & ML Project*")