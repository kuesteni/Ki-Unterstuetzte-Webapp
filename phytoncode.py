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
# UI
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg,#00ffd5,#0080ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.card {
    background: rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    backdrop-filter: blur(10px);
}
.metric {
    font-size: 26px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

colL, colR = st.columns([6,1])

with colL:
    st.markdown(f'<div class="main-title">{T["title"]}</div>', unsafe_allow_html=True)

with colR:
    st.button("🇩🇪 / 🇬🇧", on_click=toggle_lang)

# -----------------------------
# MODEL B (TensorFlow)
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
# MODEL A (ADVANCED FEATURE + ANGLES + DISTANCES)
# -----------------------------
def model_a_feature_vector(lm):
    def dist(a, b):
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

    def angle(a, b, c):
        ab = np.array([a.x - b.x, a.y - b.y])
        cb = np.array([c.x - b.x, c.y - b.y])

        cosine = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    # landmarks
    wrist = lm[0]

    thumb_tip = lm[4]
    index_tip = lm[8]
    middle_tip = lm[12]
    ring_tip = lm[16]
    pinky_tip = lm[20]

    index_mcp = lm[5]
    middle_mcp = lm[9]
    ring_mcp = lm[13]
    pinky_mcp = lm[17]

    palm_size = dist(wrist, middle_mcp)
    spread = dist(index_mcp, pinky_mcp) / (palm_size + 1e-6)

    thumb_index_dist = dist(thumb_tip, index_tip)
    thumb_palm_dist = dist(thumb_tip, wrist)

    index_up = index_tip.y < lm[6].y
    middle_up = middle_tip.y < lm[10].y
    ring_up = ring_tip.y < lm[14].y
    pinky_up = pinky_tip.y < lm[18].y
    thumb_up = thumb_tip.x < lm[3].x

    fingers = (thumb_up, index_up, middle_up, ring_up, pinky_up)

    # ---------------- NUMBERS ----------------
    if not any(fingers):
        return "0"
    if fingers == (0,1,0,0,0):
        return "1"
    if fingers == (0,1,1,0,0):
        return "2"
    if fingers == (0,1,1,1,0):
        return "3"
    if fingers == (0,1,1,1,1):
        return "4"
    if fingers == (1,1,1,1,1):
        return "5"

    # improved 6–9
    if thumb_up and index_up and not middle_up and not ring_up:
        return "6"

    if thumb_up and index_up and middle_up and not ring_up:
        return "7"

    if index_up and middle_up and ring_up and pinky_up and spread > 1.4:
        return "8"

    if index_up and middle_up and ring_up and pinky_up and thumb_up:
        return "9"

    # ---------------- LETTERS ----------------
    if not any([index_up, middle_up, ring_up, pinky_up]) and thumb_up:
        return "A"

    if not thumb_up and all([index_up, middle_up, ring_up, pinky_up]):
        return "B"

    if thumb_index_dist > palm_size * 0.6 and spread < 1.2:
        return "C"

    if thumb_up and index_up:
        return "L"

    if index_up and middle_up and not ring_up:
        return "V"

    if pinky_up:
        return "I"

    if thumb_up and pinky_up:
        return "Y"

    if index_up and middle_up and spread < 1.1:
        return "U"

    if thumb_index_dist < palm_size * 0.3:
        return "F"

    return str(sum(fingers))

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
uploaded_file = st.file_uploader(T["upload"], type=["jpg","png","jpeg"])

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -------------------------
    # MODEL A
    # -------------------------
    result = hands.process(rgb)

    model_a_label = T["nohand"]

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            model_a_label = model_a_feature_vector(hand_landmarks.landmark)

    # -------------------------
    # MODEL B
    # -------------------------
    preds = model.predict(preprocess(rgb), verbose=0)[0]

    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown"

    final = f"{label} ({conf:.2f})" if conf > 0.85 else "Uncertain"

    if conf > 0.85:
        speak(label)

    # -------------------------
    # UI
    # -------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### 🔵 {T['modelA']}")
        st.markdown(f"<div class='metric'>{model_a_label}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### 🟣 {T['modelB']}")
        st.markdown(f"<div class='metric'>{label}</div>", unsafe_allow_html=True)
        st.write(f"Index: {idx}")
        st.write(f"Confidence: {conf:.2f}")
        st.progress(conf)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### 🟢 {T['final']}")
        st.markdown(f"<div class='metric'>{final}</div>", unsafe_allow_html=True)

    st.image(img, channels="BGR")
