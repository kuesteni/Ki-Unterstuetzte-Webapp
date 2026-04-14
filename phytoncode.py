import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
import base64

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
        "modelA": "Model A (Heuristic AI)",
        "modelB": "Model B (Neural Network)",
        "final": "Final Result",
        "nohand": "No Hand Detected",
        "hand": "Hand Detected"
    },
    "DE": {
        "title": "🤖 KI Hand Erkennungssystem",
        "upload": "Bild hochladen",
        "modelA": "Modell A (Heuristik KI)",
        "modelB": "Modell B (Neuronales Netz)",
        "final": "Endergebnis",
        "nohand": "Keine Hand erkannt",
        "hand": "Hand erkannt"
    }
}

T = TEXT[lang]

# -----------------------------
# UI STYLE
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

# -----------------------------
# HEADER
# -----------------------------
colL, colR = st.columns([6,1])

with colL:
    st.markdown(f'<div class="main-title">{T["title"]}</div>', unsafe_allow_html=True)

with colR:
    st.button("🇩🇪 / 🇬🇧", on_click=toggle_lang)

# -----------------------------
# MODEL B (AI)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5")

model = load_model()

# -----------------------------
# CLASSES
# -----------------------------
CLASS_MAP = {
    0:"0",1:"1",2:"2",3:"3",4:"4",
    5:"5",6:"6",7:"7",8:"8",9:"9",
    10:"A",11:"B",12:"C",13:"L",14:"V",
    15:"O",16:"I",17:"Y",18:"U",19:"F"
}

# -----------------------------
# MEDIA PIPE
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -----------------------------
# 🔥 STABILE HEURISTIK (FIX)
# -----------------------------
def get_finger_states(hand_landmarks):
    lm = hand_landmarks.landmark

    fingers = []

    # Thumb (robust)
    fingers.append(1 if lm[4].x < lm[3].x else 0)

    # Index
    fingers.append(1 if lm[8].y < lm[6].y else 0)

    # Middle
    fingers.append(1 if lm[12].y < lm[10].y else 0)

    # Ring
    fingers.append(1 if lm[16].y < lm[14].y else 0)

    # Pinky
    fingers.append(1 if lm[20].y < lm[18].y else 0)

    return fingers

def gesture_from_fingers(f):
    # mapping
    if f == [0,0,0,0,0]:
        return "Fist (0)"

    if f == [1,1,1,1,1]:
        return "Open Hand (5)"

    if f == [0,1,1,0,0]:
        return "V / Peace"

    if f == [0,1,0,0,0]:
        return "1 Finger"

    if f == [1,0,0,0,0]:
        return "Thumbs"

    return f"Unknown {f}"

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

    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -------------------------
    # MODEL A (FIXED HEURISTIC)
    # -------------------------
    result = hands.process(rgb)

    model_a_label = "No Hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = get_finger_states(hand_landmarks)
            model_a_label = gesture_from_fingers(fingers)

    # -------------------------
    # MODEL B (AI)
    # -------------------------
    preds = model.predict(preprocess(rgb), verbose=0)[0]

    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    label = CLASS_MAP.get(idx, "Unknown")

    final = f"{label} | Class {idx} | {conf:.2f}" if conf > 0.85 else "Uncertain"

    if conf > 0.85:
        speak(label)

    # -----------------------------
    # DASHBOARD
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### 🔵 {T['modelA']}")
        st.markdown(f"<div class='metric'>{model_a_label}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### 🟣 {T['modelB']}")
        st.markdown(f"<div class='metric'>🧠 {label}</div>", unsafe_allow_html=True)

        st.write(f"🔢 Class Index: {idx}")
        st.write(f"📊 Confidence: {conf:.2f}")
        st.progress(float(conf))

        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### 🟢 {T['final']}")
        st.markdown(f"<div class='metric'>{final}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.image(img, channels="BGR")
