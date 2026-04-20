import streamlit as st
import numpy as np
import cv2
import os
import mediapipe as mp
import tensorflow as tf
import math

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI Hand Recognition", layout="wide")

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# -----------------------------
# MODEL LOAD
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5")

model = load_model()

# -----------------------------
# 🔥 FIXED LABEL LOADING (IMPORTANT)
# -----------------------------
def load_labels():
    with open("labels.txt", "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]

    # remove empty lines (VERY IMPORTANT FIX)
    labels = [l for l in labels if l != ""]

    return labels

CLASS_NAMES = load_labels()

st.write("DEBUG Labels:", CLASS_NAMES)

# -----------------------------
# MEDIA PIPE
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# -----------------------------
# MODEL A (unchanged example)
# -----------------------------
def model_a_predict(lm):
    thumb = lm[4].x < lm[3].x
    index = lm[8].y < lm[6].y
    middle = lm[12].y < lm[10].y
    ring = lm[16].y < lm[14].y
    pinky = lm[20].y < lm[18].y

    fingers = (thumb, index, middle, ring, pinky)

    gesture_map = {
        (False, False, False, False, False): "0",
        (False, True, False, False, False): "1",
        (False, True, True, False, False): "2",
        (False, True, True, True, False): "3",
        (False, True, True, True, True): "4",
        (True, True, True, True, True): "5",
    }

    return gesture_map.get(fingers, "Unknown")

# -----------------------------
# PREPROCESS (important fix option)
# -----------------------------
def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# -----------------------------
# UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------------- MODEL B ----------------
    preds = model.predict(preprocess(rgb), verbose=0)[0]

    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    # 🔥 SAFE INDEX FIX (IMPORTANT)
    if idx >= len(CLASS_NAMES):
        label = "OUT_OF_RANGE"
    else:
        label = CLASS_NAMES[idx]

    # ---------------- 🔥 SHIFT DEBUG ----------------
    st.write("RAW preds:", preds)
    st.write("Predicted index:", idx)
    st.write("Confidence:", conf)
    st.write("Mapped label:", label)

    # ---------------- DISPLAY ----------------
    st.image(rgb, caption="Input")

    st.markdown("## 🟣 Model B Output")
    st.markdown(f"### {label}")
    st.progress(conf)
