import cv2
import torch
import numpy as np
import sqlite3
import time
import queue
import vosk
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import threading
import os

# =====================================================
# THREAD LOCK
# =====================================================
db_lock = threading.Lock()

DEVICE = torch.device("cpu")

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_MODEL_PATH = os.path.join(BASE_DIR, "models/mobilenetv3_rafdb_ts825.pt")
TEXT_MODEL_PATH = os.path.join(BASE_DIR, "models/minilm_emotion_model")
VOSK_MODEL_PATH = os.path.join(BASE_DIR, "models/vosk-model-small-en-in-0.4")

TEXT_DB = os.path.join(BASE_DIR, "emotion_data_text_final.db")
FACE_DB = os.path.join(BASE_DIR, "emotion_data_face.db")

# =====================================================
# CONFIG
# =====================================================
SPEECH_SAMPLE_RATE = 16000
RECORD_SECONDS = 4

FACE_CLASSES = ["Surprise","Fear","Disgust","Happy","Sad","Anger","Neutral"]
TEXT_CLASSES = ["happy","sad","angry","fear","surprise","neutral"]

QUESTIONS = [
    "How do you feel about your work environment?",
    "Do you feel valued and appreciated at your workplace?",
    "How satisfied are you with your work-life balance?",
    "Do you enjoy your daily tasks and responsibilities?",
    "Would you recommend your workplace to others?"
]

# =====================================================
# DATABASE INIT
# =====================================================
def init_text_db():
    conn = sqlite3.connect(TEXT_DB)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS emotion_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            happy REAL,
            sad REAL,
            angry REAL,
            fear REAL,
            surprise REAL,
            neutral REAL
        )
    """)

    conn.commit()
    conn.close()


def init_face_db():
    conn = sqlite3.connect(FACE_DB)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS emotion_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            Surprise REAL,
            Fear REAL,
            Disgust REAL,
            Happy REAL,
            Sad REAL,
            Anger REAL,
            Neutral REAL
        )
    """)

    conn.commit()
    conn.close()


init_text_db()
init_face_db()

# =====================================================
# MODELS
# =====================================================
face_model = torch.jit.load(FACE_MODEL_PATH, map_location="cpu")
face_model.eval()

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)

text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)

text_model = torch.quantization.quantize_dynamic(
    text_model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

text_model.eval()

# =====================================================
# SCORING
# =====================================================
def compute_A0(e):
    return (
        1.0 * e["happy"] +
        0.7 * e["surprise"] +
        0.5 * e["neutral"] +
        0.2 * e["fear"] +
        0.1 * e["sad"]
    )


def compute_V0(e):
    return (
        1.0 * e["Happy"] +
        0.7 * e["Surprise"] +
        0.5 * e["Neutral"] +
        0.3 * e["Disgust"] +
        0.2 * e["Fear"] +
        0.1 * e["Sad"]
    )

# =====================================================
# TEXT PROCESSING
# =====================================================
def process_text_answer(text_input):

    conn = sqlite3.connect(TEXT_DB)
    cur = conn.cursor()

    encoding = tokenizer(
        text_input,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = text_model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )

        probs = torch.softmax(outputs.logits, dim=1)[0]

    emotions = {
        TEXT_CLASSES[i]: probs[i].item()
        for i in range(6)
    }

    timestamp = datetime.now().isoformat()

    with db_lock:
        cur.execute("""
            INSERT INTO emotion_log
            (timestamp, happy, sad, angry, fear, surprise, neutral)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            emotions["happy"],
            emotions["sad"],
            emotions["angry"],
            emotions["fear"],
            emotions["surprise"],
            emotions["neutral"]
        ))

        conn.commit()

    conn.close()

    return compute_A0(emotions)

# =====================================================
# VIDEO PROCESSING
# =====================================================
def process_video_answer():

    import sounddevice as sd

    cap = cv2.VideoCapture(0)

    conn = sqlite3.connect(FACE_DB)
    cur = conn.cursor()

    vosk_model = vosk.Model(VOSK_MODEL_PATH)
    rec = vosk.KaldiRecognizer(vosk_model, SPEECH_SAMPLE_RATE)

    audio_queue = queue.Queue()
    all_audio = bytearray()

    def audio_callback(indata, frames, time_info, status):
        audio_queue.put(bytes(indata))

    frame_count = 0
    start = time.time()

    with sd.RawInputStream(
        samplerate=SPEECH_SAMPLE_RATE,
        blocksize=2000,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):

        while time.time() - start < RECORD_SECONDS:

            ret, frame = cap.read()

            if not ret:
                continue

            frame_count += 1

            if frame_count % 3 != 0:
                continue

            frame = cv2.resize(frame, (160,160))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            tensor = torch.from_numpy(frame).permute(2,0,1).float() / 255.0
            tensor = tensor.unsqueeze(0)

            with torch.no_grad():
                probs = torch.softmax(face_model(tensor), dim=1)[0]

            emotions = {
                FACE_CLASSES[i]: probs[i].item()
                for i in range(7)
            }

            with db_lock:
                cur.execute("""
                    INSERT INTO emotion_log
                    (timestamp, Surprise, Fear, Disgust, Happy, Sad, Anger, Neutral)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    emotions["Surprise"],
                    emotions["Fear"],
                    emotions["Disgust"],
                    emotions["Happy"],
                    emotions["Sad"],
                    emotions["Anger"],
                    emotions["Neutral"]
                ))

                conn.commit()

            while not audio_queue.empty():
                all_audio.extend(audio_queue.get())

    cap.release()
    conn.close()

    # =================================================
    # SPEECH TO TEXT
    # =================================================
    rec_text = ""

    if len(all_audio) > 0:
        rec.AcceptWaveform(bytes(all_audio))
        rec_text = json.loads(rec.FinalResult()).get("text", "")

    if not rec_text.strip():
        rec_text = "neutral"

    # speech text also uses text model
    return process_text_answer(rec_text)
