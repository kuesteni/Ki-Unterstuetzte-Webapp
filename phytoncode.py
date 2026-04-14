import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque, Counter
import pyttsx3
import tensorflow as tf

# -----------------------------
# CONFIG
# -----------------------------
CONFIDENCE_THRESHOLD = 0.85
STABILITY_FRAMES = 10

# Klassen (Teachable Machine Reihenfolge!)
CLASSES = [
    "0","1","2","3","4","5","6","7","8","9",
    "A","B","C","L","V","O","I","Y","U","F"
]

# -----------------------------
# TTS Setup
# -----------------------------
engine = pyttsx3.init()

def speak(text, lang="en"):
    engine.say(text)
    engine.runAndWait()

# -----------------------------
# Load Model (Teachable Machine)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# -----------------------------
# MediaPipe Setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)

# -----------------------------
# UI
# -----------------------------
st.title("🤖 Handzeichen Übersetzer")

# Sprache umschalten
col1, col2 = st.columns([10,1])

with col2:
    lang = st.radio("🌐", ["EN", "DE"])

st.markdown("### Live Kamera")

run = st.checkbox("Start Kamera")

FRAME_WINDOW = st.image([])

# -----------------------------
# STATE
# -----------------------------
prediction_buffer = deque(maxlen=STABILITY_FRAMES)
current_word = ""
last_output = ""
last_time = time.time()

# -----------------------------
# Kamera
# -----------------------------
cap = cv2.VideoCapture(0)

def preprocess_landmarks(hand_landmarks):
    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return np.array(data).reshape(1, -1)

# -----------------------------
# MAIN LOOP
# -----------------------------
while run:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    prediction = None
    confidence = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Zeichne Punkte
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Feature Extraction
            input_data = preprocess_landmarks(hand_landmarks)

            # ML Prediction
            preds = model.predict(input_data, verbose=0)[0]
            idx = np.argmax(preds)
            confidence = preds[idx]

            if confidence > CONFIDENCE_THRESHOLD:
                prediction = CLASSES[idx]
                prediction_buffer.append(prediction)

    # -----------------------------
    # Stabilisierung
    # -----------------------------
    if len(prediction_buffer) == STABILITY_FRAMES:
        most_common = Counter(prediction_buffer).most_common(1)[0][0]

        if most_common != last_output:
            last_output = most_common

            # DELETE Geste (z. B. "F")
            if most_common == "F":
                current_word = current_word[:-1]

            else:
                current_word += most_common

                # Sound
                if lang == "DE":
                    speak(most_common, "de")
                else:
                    speak(most_common, "en")

    # -----------------------------
    # Anzeige
    # -----------------------------
    if prediction and confidence > CONFIDENCE_THRESHOLD:
        cv2.putText(frame, f"{prediction} ({confidence:.2f})",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

    # Wort anzeigen
    cv2.putText(frame, f"Word: {current_word}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255,0,0), 2)

    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()
