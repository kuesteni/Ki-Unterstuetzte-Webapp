import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time

# =========================
# CONFIG
# =========================
MODEL_PATH = "model/keras_model.h5"
LABELS_PATH = "model/labels.txt"

CONFIDENCE_THRESHOLD = 0.9
COOLDOWN_TIME = 1.5  # Sekunden zwischen Buchstaben

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_tm_model():
    model = load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

model, labels = load_tm_model()

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# SESSION STATE
# =========================
if "sentence" not in st.session_state:
    st.session_state.sentence = ""

if "last_added_time" not in st.session_state:
    st.session_state.last_added_time = 0

# =========================
# FUNCTIONS
# =========================
def predict_tm(image):
    img = cv2.resize(image, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)
    index = np.argmax(prediction)
    confidence = float(prediction[0][index])

    return labels[index], confidence


def count_fingers(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip in tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

# =========================
# UI
# =========================
st.title("✋ KI Handzeichen Übersetzer")
st.write("Live-Erkennung mit MediaPipe + Teachable Machine")

run = st.checkbox("Kamera starten")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Säule A (MediaPipe)")
    mp_output = st.empty()

with col2:
    st.subheader("Säule B (Teachable Machine)")
    tm_output = st.empty()

st.subheader("📝 Erkannter Text")
text_output = st.empty()

if st.button("Text löschen"):
    st.session_state.sentence = ""

frame_window = st.image([])

cap = cv2.VideoCapture(0)

# =========================
# MAIN LOOP
# =========================
while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Kamera nicht gefunden")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    gesture_text = "Keine Hand"
    finger_count = 0

    # ===== Säule A =====
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_count = count_fingers(hand_landmarks)
            gesture_text = f"{finger_count} Finger erkannt"

    # ===== Säule B =====
    label, confidence = predict_tm(frame)

    # Stabilisierung + Cooldown
    current_time = time.time()

    if confidence > CONFIDENCE_THRESHOLD:
        tm_output.markdown(f"**{label} ({confidence:.2f})**")

        if current_time - st.session_state.last_added_time > COOLDOWN_TIME:
            st.session_state.sentence += label
            st.session_state.last_added_time = current_time
    else:
        tm_output.markdown("Unsicher...")

    # ===== UI Output =====
    mp_output.markdown(f"**{gesture_text}**")
    text_output.markdown(f"### {st.session_state.sentence}")

    frame_window.image(frame, channels="BGR")

cap.release()
