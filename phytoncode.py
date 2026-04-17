import streamlit as st
import numpy as np
import cv2
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
import base64
import math

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Hand Pro", layout="wide")

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
# CSS
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 46px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg,#00ffd5,#3b82f6,#a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
}
.card {
    background: rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.08);
    transition: 0.3s;
}
.card:hover {
    transform: translateY(-3px);
}
.metric {
    font-size: 34px;
    font-weight: 800;
    text-align: center;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

colL, colR = st.columns([6, 1])

with colL:
    st.markdown(f'<div class="main-title">{T["title"]}</div>', unsafe_allow_html=True)

with colR:
    st.button("🇩🇪 / 🇬🇧", on_click=toggle_lang)

# -----------------------------
# MODEL B
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
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# -----------------------------
# MODEL A (FIXED)
# -----------------------------
def model_a_feature_vector(lm):
    def dist(a, b):
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

    wrist      = lm[0]
    thumb_tip  = lm[4]
    index_tip  = lm[8]
    middle_tip = lm[12]
    ring_tip   = lm[16]
    pinky_tip  = lm[20]

    index_mcp  = lm[5]
    middle_mcp = lm[9]
    ring_mcp   = lm[13]
    pinky_mcp  = lm[17]

    palm_size = dist(wrist, middle_mcp) + 1e-6
    spread = dist(index_mcp, pinky_mcp) / palm_size
    thumb_index_dist = dist(thumb_tip, index_tip)

    index_up  = index_tip.y  < lm[6].y
    middle_up = middle_tip.y < lm[10].y
    ring_up   = ring_tip.y   < lm[14].y
    pinky_up  = pinky_tip.y  < lm[18].y
    thumb_up  = thumb_tip.x  < lm[3].x

    fingers = (thumb_up, index_up, middle_up, ring_up, pinky_up)

    # ---------------- FIXED LOGIC ----------------
    if fingers == (False, False, False, False, False):
        return "0"
    if fingers == (False, True, False, False, False):
        return "1"
    if fingers == (False, True, True, False, False):
        return "2"
    if fingers == (False, True, True, True, False):
        return "3"
    if fingers == (False, True, True, True, True):
        return "4"
    if fingers == (True, True, True, True, True):
        return "5"

    if thumb_up and index_up and not middle_up and not ring_up:
        return "6"
    if thumb_up and index_up and middle_up and not ring_up:
        return "7"
    if index_up and middle_up and ring_up and pinky_up and spread > 1.4:
        return "8"
    if index_up and middle_up and ring_up and pinky_up and thumb_up:
        return "9"

    if not any([index_up, middle_up, ring_up, pinky_up]) and thumb_up:
        return "A"

    if index_up and middle_up and not ring_up:
        return "V"

    if pinky_up:
        return "I"

    if thumb_up and index_up:
        return "L"

    return "Unknown"

# -----------------------------
# HELPERS
# -----------------------------
def draw_prediction_label(img, text):
    img_copy = img.copy()
    cv2.rectangle(img_copy, (10, 10), (300, 80), (0, 0, 0), -1)
    cv2.putText(img_copy, text, (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 255, 255), 3, cv2.LINE_AA)
    return img_copy

def create_symbol_image(symbol):
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.putText(canvas, str(symbol),
                (80, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                5,
                (255, 255, 255),
                8,
                cv2.LINE_AA)
    return canvas

# -----------------------------
# PREPROCESS
# -----------------------------
IMG_SIZE = 224

def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# -----------------------------
# SPEECH
# -----------------------------
def speak(text):
    tts = gTTS(text=text, lang="en")
    tts.save("speech.mp3")
    audio = open("speech.mp3", "rb").read()
    b64 = base64.b64encode(audio).decode()

    st.markdown(f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """, unsafe_allow_html=True)

# -----------------------------
# UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(T["upload"], type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # MODEL A
    result = hands.process(rgb)
    model_a_label = T["nohand"]

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            model_a_label = model_a_feature_vector(hand_landmarks.landmark)

    # MODEL B
    preds = model.predict(preprocess(rgb), verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown"
    final = f"{label} ({conf:.2f})" if conf > 0.85 else "Uncertain"

    if conf > 0.85:
        speak(label)

    annotated_img = cv2.cvtColor(
        draw_prediction_label(img, f"{label} ({conf:.2f})"),
        cv2.COLOR_BGR2RGB
    )

    symbol_img = create_symbol_image(label if conf > 0.85 else "?")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # UI
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### 🔵 {T['modelA']}")
        st.markdown(f"<div class='metric'>{model_a_label}</div>", unsafe_allow_html=True)
        st.image(img_rgb, width=160)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### 🟣 {T['modelB']}")
        st.markdown(f"<div class='metric'>{label}</div>", unsafe_allow_html=True)
        st.write(f"Index: {idx}")
        st.write(f"Confidence: {conf:.2f}")
        st.progress(conf)
        st.image(annotated_img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### 🟢 {T['final']}")
        st.markdown(f"<div class='metric'>{final}</div>", unsafe_allow_html=True)
        st.image(symbol_img, caption="AI Symbol View", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.image(img_rgb, caption="Original Input", use_column_width=True)
