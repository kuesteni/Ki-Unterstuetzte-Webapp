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
COOLDOWN_TIME = 1.5  # Sekunden zwischen Zeichen

# =========================
# MODEL LADEN
# =========================
@st.cache_resource
def load_model_and_labels():
    model = load_model(MODEL_PATH)

    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    return model, labels

model, labels = load_model_and_labels()

# =========================
# MEDIAPIPE
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

if "last_time" not in st.session_state:
    st.session_state.last_time = 0

# =========================
# FUNKTIONEN
# =========================
def predict(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)
    index = np.argmax(prediction)
    confidence = float(prediction[0][index])

    # FIX: richtiges Label aus "10 A"
    parts = labels[index].split(" ")
    label = parts[1] if len(parts) > 1 else parts[0]

    return label, confidence


def count_fingers(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Daumen
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # andere Finger
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
st.write("MediaPipe + Teachable Machine Echtzeit-Erkennung")

run = st.checkbox("Kamera starten")

frame_placeholder = st.image([])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Säule A (Handtracking)")
    mp_out = st.empty()

with col2:
    st.subheader("Säule B (KI Modell)")
    tm_out = st.empty()

st.subheader("📝 Ergebnis")
text_out = st.empty()

if st.button("Text löschen"):
    st.session_state.sentence = ""

# Kamera
cap = cv2.VideoCapture(0)

# =========================
# LOOP
# =========================
while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Keine Kamera gefunden")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    # ===== Säule A =====
    finger_text = "Keine Hand"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = count_fingers(hand_landmarks)
            finger_text = f"{fingers} Finger erkannt"

    # ===== Säule B =====
    label, confidence = predict(frame)

    now = time.time()

    if confidence > CONFIDENCE_THRESHOLD:
        tm_out.markdown(f"**{label} ({confidence:.2f})**")

        if now - st.session_state.last_time > COOLDOWN_TIME:
            st.session_state.sentence += label
            st.session_state.last_time = now
    else:
        tm_out.markdown("Unsicher...")

    # ===== OUTPUT =====
    mp_out.markdown(f"**{finger_text}**")
    text_out.markdown(f"### {st.session_state.sentence}")

    frame_placeholder.image(frame, channels="BGR")

cap.release()
