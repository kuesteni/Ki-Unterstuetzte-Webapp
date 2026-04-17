import streamlit as st
import numpy as np
import cv2
import os
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
import base64
import math

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI Hand Recognition", layout="wide")

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# -----------------------------
# LANGUAGE STATE
# -----------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "EN"

def toggle_lang():
    st.session_state.lang = "DE" if st.session_state.lang == "EN" else "EN"

lang = st.session_state.lang

TEXT = {
    "EN": {
        "title": "🤖 AI Hand Recognition System",
        "upload": "Upload Image",
        "modelA": "Model A (Feature AI)",
        "modelB": "Model B (Neural Network)",
        "final": "Final Result",
        "nohand": "No Hand Detected",
    },
    "DE": {
        "title": "🤖 KI Hand Erkennungssystem",
        "upload": "Bild hochladen",
        "modelA": "Modell A (Feature KI)",
        "modelB": "Modell B (Neuronales Netz)",
        "final": "Endergebnis",
        "nohand": "Keine Hand erkannt",
    }
}

T = TEXT[lang]

# -----------------------------
# UI STYLE
# -----------------------------
st.markdown("""
<style>
body { background: #0b1220; }

.main-title {
    font-size: 46px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg,#00ffd5,#4f8cff,#c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.card {
    background: rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
}

.section-title {
    font-size: 14px;
    opacity: 0.8;
}

.big-number {
    font-size: 34px;
    font-weight: 800;
    text-align: center;
}

.result-box {
    background: #0a0a0a;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
}

.result-number {
    font-size: 80px;
    font-weight: 900;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
col1, col2 = st.columns([8, 1])

with col1:
    st.markdown(f'<div class="main-title">{T["title"]}</div>', unsafe_allow_html=True)

with col2:
    st.button("🇩🇪 / 🇬🇧", on_click=toggle_lang)

# -----------------------------
# MODEL LOAD
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5")

model = load_model()

def load_labels():
    with open("labels.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

CLASS_NAMES = load_labels()

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
# MODEL A
# -----------------------------
def model_a_feature_vector(lm):
    thumb = lm[4]
    index = lm[8]
    middle = lm[12]
    ring = lm[16]
    pinky = lm[20]

    fingers = (
        thumb.x < lm[3].x,
        index.y < lm[6].y,
        middle.y < lm[10].y,
        ring.y < lm[14].y,
        pinky.y < lm[18].y
    )

    return str(sum(fingers))

# -----------------------------
# HELPERS
# -----------------------------
def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# -----------------------------
# UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(T["upload"], type=["jpg","png","jpeg"])

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------------- MODEL A ----------------
    result = hands.process(rgb)
    model_a_label = T["nohand"]
    img_a = img.copy()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img_a, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            model_a_label = model_a_feature_vector(hand_landmarks.landmark)

    # ---------------- MODEL B ----------------
    preds = model.predict(preprocess(rgb), verbose=0)[0]
    idx = np.argmax(preds)
    conf = float(preds[idx])

    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown"

    # ---------------- 🔥 FIXED FUSION LOGIC ----------------
    try:
        model_a_value = int(model_a_label)
    except:
        model_a_value = None

    if conf > 0.85:
        final = label
    elif model_a_value is not None:
        final = str(model_a_value)
    else:
        final = label

    # ---------------- IMAGE ----------------
    img_b = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------------- UI ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>🔵 {T['modelA']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-number'>{model_a_label}</div>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB), use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>🟣 {T['modelB']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-number'>{label}</div>", unsafe_allow_html=True)
        st.write(f"Index: {idx}")
        st.write(f"Confidence: {conf:.2f}")
        st.progress(conf)
        st.image(img_b, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>🟢 {T['final']}</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-box">
            <div class="result-number">
                {final}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.image(img_b, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
