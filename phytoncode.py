import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
import base64
import os
import math

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Hand", layout="wide")

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# ---------------- STYLE (MATCH SCREENSHOT) ----------------
st.markdown("""
<style>

body {
    background: #0b1220;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 20px;
    background: linear-gradient(90deg,#00ffd5,#4f8cff,#c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* GRID CARDS */
.card {
    background: rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(18px);
    height: 520px;
}

/* HEADER STRIP LIKE IMAGE */
.card-header {
    font-size: 13px;
    font-weight: 600;
    padding: 8px 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    display: inline-block;
}

.blue { background: rgba(59,130,246,0.2); color:#60a5fa; }
.purple { background: rgba(168,85,247,0.2); color:#c084fc; }
.green { background: rgba(34,197,94,0.2); color:#4ade80; }

.big {
    font-size: 34px;
    font-weight: 800;
    text-align: center;
    margin: 10px 0;
}

.small {
    font-size: 13px;
    opacity: 0.7;
    text-align: center;
}

/* CONF BAR */
.conf {
    height: 10px;
    border-radius: 10px;
}

/* FINAL BOX (BLACK TILE) */
.final-box {
    background: #0a0a0a;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    margin-top: 10px;
}

.final-number {
    font-size: 70px;
    font-weight: 900;
    color: white;
}

img {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="title">🤖 AI Hand Recognition System</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5")

model = load_model()

labels = open("labels.txt").read().splitlines()

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# ---------------- MODEL A ----------------
def model_a(lm):
    fingers = [
        lm[4].x < lm[3].x,
        lm[8].y < lm[6].y,
        lm[12].y < lm[10].y,
        lm[16].y < lm[14].y,
        lm[20].y < lm[18].y
    ]
    return str(sum(fingers))

# ---------------- PREPROCESS ----------------
def prep(img):
    img = cv2.resize(img, (224,224))
    return np.expand_dims(img/255.0, axis=0)

# ---------------- INPUT ----------------
file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if file:

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------------- MODEL A ----------------
    res = hands.process(rgb)
    out_a = "No Hand"

    img_a = img.copy()

    if res.multi_hand_landmarks:
        for h in res.multi_hand_landmarks:
            out_a = model_a(h.landmark)

    # ---------------- MODEL B ----------------
    preds = model.predict(prep(rgb), verbose=0)[0]
    idx = np.argmax(preds)
    conf = float(preds[idx])
    label = labels[idx] if idx < len(labels) else "?"

    final = label if conf > 0.85 else "Uncertain"

    # ---------------- UI ----------------
    col1, col2, col3 = st.columns(3)

    # -------- MODEL A --------
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header blue">Model A (Feature AI)</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="big">{out_a}</div>', unsafe_allow_html=True)
        st.image(img_a, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------- MODEL B --------
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header purple">Model B (Neural Network)</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="big">{label}</div>', unsafe_allow_html=True)

        st.write(f"Index: {idx}")
        st.write(f"Confidence: {conf:.2f}")
        st.progress(conf)

        st.image(rgb, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------- FINAL --------
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header green">Final Result</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="final-box">
            <div class="final-number">{final}</div>
        </div>
        """, unsafe_allow_html=True)

        st.image(rgb, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
