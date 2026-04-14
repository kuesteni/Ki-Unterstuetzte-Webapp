import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque, Counter
from gtts import gTTS
import base64
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Hand Dashboard", layout="wide")

# -----------------------------
# UI STYLE
# -----------------------------
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: 800;
    color: #00ffd5;
    text-align: center;
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    margin: 5px;
}

.metric {
    font-size: 26px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# CONFIG
# -----------------------------
CONF_THRESHOLD = 0.85
FRAME_STABILITY = 8
IMG_SIZE = 224

CLASSES = [
    "0","1","2","3","4","5","6","7","8","9",
    "A","B","C","L","V","O","I","Y","U","F"
]

# -----------------------------
# SPEECH (CLOUD SAFE)
# -----------------------------
def speak(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts.save("speech.mp3")

    audio_file = open("speech.mp3", "rb")
    audio_bytes = audio_file.read()

    b64 = base64.b64encode(audio_bytes).decode()

    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# -----------------------------
# MODEL B (Teachable Machine)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# -----------------------------
# MEDIA PIPE (Model A)
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# -----------------------------
# TITLE
# -----------------------------
st.markdown('<div class="big-title">🤖 AI Hand Comparison Dashboard</div>', unsafe_allow_html=True)

run = st.checkbox("🚀 Start Camera")

frame_placeholder = st.image([])

# -----------------------------
# STATE
# -----------------------------
buffer = deque(maxlen=FRAME_STABILITY)
history = deque(maxlen=10)

word = ""
last = ""

cap = cv2.VideoCapture(0)

# -----------------------------
# MAIN LOOP
# -----------------------------
while run:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------------------------
    # MODEL A (MediaPipe)
    # -------------------------
    result = hands.process(rgb)

    model_a_status = "❌ No Hand"
    if result.multi_hand_landmarks:
        model_a_status = "🟢 Hand Detected"
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # -------------------------
    # MODEL B (AI CLASSIFIER)
    # -------------------------
    img_input = preprocess(rgb)
    preds = model.predict(img_input, verbose=0)[0]

    idx = np.argmax(preds)
    conf = preds[idx]
    label = CLASSES[idx]

    # -------------------------
    # FILTER
    # -------------------------
    if conf > CONF_THRESHOLD:
        buffer.append(label)

    # -------------------------
    # STABLE OUTPUT
    # -------------------------
    if len(buffer) == FRAME_STABILITY:
        stable = Counter(buffer).most_common(1)[0][0]

        if stable != last:
            last = stable
            history.append(stable)

            # DELETE
            if stable == "F":
                word = word[:-1]

            # RESET
            elif stable == "Y":
                word = ""

            else:
                word += stable

                # 🔊 SOUND (CLOUD SAFE)
                speak(stable, "en")

    # -------------------------
    # DASHBOARD UI
    # -------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔵 Model A")
        st.markdown(f"<div class='metric'>{model_a_status}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🟣 Model B")
        st.markdown(f"<div class='metric'>{label}</div>", unsafe_allow_html=True)
        st.progress(float(conf))
        st.markdown(f"Confidence: {conf:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🟢 Word")
        st.markdown(f"<div class='metric'>{word}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------
    # HISTORY
    # -------------------------
    st.write("📜 History:", " ".join(history))

    # -------------------------
    # FRAME
    # -------------------------
    if conf > CONF_THRESHOLD:
        cv2.putText(frame, f"{label} {conf:.2f}",
                    (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

    frame_placeholder.image(frame, channels="BGR")

cap.release()
