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
st.set_page_config(page_title="AI Hand Detection Pro", layout="wide")

# -----------------------------
# UI
# -----------------------------
st.markdown("""
<style>
.title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    color: #00ffd5;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 15px;
    margin: 10px;
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
IMG_SIZE = 224
CONF_THRESHOLD = 0.85

CLASSES = [
    "0","1","2","3","4","5","6","7","8","9",
    "A","B","C","L","V","O","I","Y","U","F"
]

# -----------------------------
# SPEECH (CLOUD SAFE)
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
# MODEL B (Teachable Machine)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5")

model = load_model()

# -----------------------------
# MEDIA PIPE (MODEL A)
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# -----------------------------
# TITLE
# -----------------------------
st.markdown('<div class="title">🤖 AI Hand Detection & Comparison System</div>', unsafe_allow_html=True)

# -----------------------------
# UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📸 Upload Hand Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # -------------------------
    # IMPROVED IMAGE PROCESSING
    # -------------------------
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)  # contrast boost
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -------------------------
    # MODEL A (MediaPipe)
    # -------------------------
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        model_a = "🟢 Hand Detected"

        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        model_a = "❌ No Hand Detected (try clearer image)"

    # -------------------------
    # MODEL B (ML)
    # -------------------------
    input_img = preprocess(rgb)
    preds = model.predict(input_img, verbose=0)[0]

    idx = np.argmax(preds)
    conf = preds[idx]
    label = CLASSES[idx]

    if conf > CONF_THRESHOLD:
        final = label
        speak(label)
    else:
        final = "Uncertain"

    # -------------------------
    # DASHBOARD
    # -------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔵 Model A (MediaPipe)")
        st.markdown(f"<div class='metric'>{model_a}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🟣 Model B (AI Model)")
        st.markdown(f"<div class='metric'>{label}</div>", unsafe_allow_html=True)
        st.progress(float(conf))
        st.write(f"Confidence: {conf:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🟢 Final Result")
        st.markdown(f"<div class='metric'>{final}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------
    # IMAGE OUTPUT
    # -------------------------
    st.image(img, channels="BGR", caption="Processed Image")
