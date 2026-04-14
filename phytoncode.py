import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque, Counter
import pyttsx3
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Hand Translator", layout="wide")

# -----------------------------
# CUSTOM CSS (🔥 FANCY UI)
# -----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.big-text {
    font-size: 48px;
    font-weight: bold;
}

.card {
    background: rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
}

.word-box {
    font-size: 40px;
    font-weight: bold;
    color: #00ffcc;
}

.conf-bar {
    height: 20px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# CONFIG
# -----------------------------
CONFIDENCE_THRESHOLD = 0.85
STABILITY_FRAMES = 10
IMG_SIZE = 224

CLASSES = [
    "0","1","2","3","4","5","6","7","8","9",
    "A","B","C","L","V","O","I","Y","U","F"
]

# -----------------------------
# TTS
# -----------------------------
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# -----------------------------
# MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# -----------------------------
# MEDIAPIPE
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="big-text">🤖 AI Hand Gesture Translator</div>', unsafe_allow_html=True)

col_lang, col_stats = st.columns([1,3])

with col_lang:
    lang = st.radio("🌐 Language", ["EN", "DE"])

with col_stats:
    st.markdown("### 📊 Live System Status")

# -----------------------------
# LAYOUT
# -----------------------------
col1, col2 = st.columns([2,1])

FRAME_WINDOW = col1.image([])

with col2:
    word_placeholder = st.empty()
    confidence_placeholder = st.empty()
    history_placeholder = st.empty()

# -----------------------------
# STATE
# -----------------------------
prediction_buffer = deque(maxlen=STABILITY_FRAMES)
history = deque(maxlen=10)

current_word = ""
last_output = ""

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# -----------------------------
# CAMERA
# -----------------------------
run = st.checkbox("🚀 Start Camera")

cap = cv2.VideoCapture(0)

# -----------------------------
# LOOP
# -----------------------------
while run:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe Overlay
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Prediction
    input_img = preprocess(rgb)
    preds = model.predict(input_img, verbose=0)[0]

    idx = np.argmax(preds)
    confidence = preds[idx]
    prediction = CLASSES[idx]

    # Confidence Bar UI
    confidence_placeholder.progress(float(confidence))

    # Filter
    if confidence > CONFIDENCE_THRESHOLD:
        prediction_buffer.append(prediction)

    # Stabilisierung
    if len(prediction_buffer) == STABILITY_FRAMES:
        most_common = Counter(prediction_buffer).most_common(1)[0][0]

        if most_common != last_output:
            last_output = most_common
            history.append(most_common)

            if most_common == "F":
                current_word = current_word[:-1]

            elif most_common == "Y":
                current_word = ""

            else:
                current_word += most_common
                speak(most_common)

    # -----------------------------
    # UI UPDATE
    # -----------------------------
    word_placeholder.markdown(
        f'<div class="card"><div class="word-box">{current_word}</div></div>',
        unsafe_allow_html=True
    )

    history_placeholder.markdown(
        f'<div class="card">History: {" ".join(history)}</div>',
        unsafe_allow_html=True
    )

    if confidence > CONFIDENCE_THRESHOLD:
        cv2.putText(frame, f"{prediction} ({confidence:.2f})",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()
