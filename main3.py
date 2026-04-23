import streamlit as st
import torch
import sqlite3
import os
import base64
import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import datetime

# =====================================================
# PERFORMANCE (Raspberry Pi)
# =====================================================
torch.set_num_threads(2)

# =====================================================
# IMPORT BACKEND
# =====================================================
from ayush_drive import (
    QUESTIONS,
    process_text_answer,
    process_video_answer
)

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GIF_DIR = os.path.join(BASE_DIR, "gifs")

TEXT_DB = os.path.join(BASE_DIR, "emotion_data_text_final.db")
FACE_DB = os.path.join(BASE_DIR, "emotion_data_face.db")

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Happiness Index System",
    layout="centered"
)

# =====================================================
# GIF DISPLAY
# =====================================================
def show_gif(filename, width=260):

    path = os.path.join(GIF_DIR, filename)

    if os.path.exists(path):

        with open(path, "rb") as f:
            data = f.read()

        b64 = base64.b64encode(data).decode()

        st.markdown(
            f"""
            <div style="text-align:center;">
                <img src="data:image/gif;base64,{b64}" width="{width}">
            </div>
            """,
            unsafe_allow_html=True
        )

# =====================================================
# FINAL GIF BASED ON SCORE
# =====================================================
def final_gif(score):

    val = score * 100

    if val < 30:
        return "anger.gif"
    elif val < 40:
        return "sad.gif"
    elif val < 60:
        return "neutral.gif"
    else:
        return "happy.gif"

# =====================================================
# GRAPH FUNCTION
# =====================================================
def show_graph(values, title):

    if len(values) == 0:
        st.warning("No graph data available.")
        return

    x = [1,2,3,4,5]

    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(x, values, marker='o', linewidth=2, markersize=8)

    ax.set_title(title)
    ax.set_xlabel("Question Number")
    ax.set_ylabel("Happiness Index")

    ax.set_xlim(1,5)
    ax.set_ylim(0,100)

    ax.set_xticks([1,2,3,4,5])
    ax.set_yticks([10,20,30,40,50,60,70,80,90,100])

    ax.grid(True)

    plt.tight_layout()

    st.pyplot(fig)

# =====================================================
# TEXT GRAPH DATA (last 5 rows)
# =====================================================
def get_last_5_text_scores():

    if not os.path.exists(TEXT_DB):
        return []

    try:
        conn = sqlite3.connect(TEXT_DB)

        query = """
        SELECT happy, sad, angry, fear, surprise, neutral
        FROM emotion_log
        ORDER BY rowid DESC
        LIMIT 5
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        df = df.iloc[::-1].reset_index(drop=True)

        values = []

        for _, row in df.iterrows():

            hi = (
                1.0 * row["happy"] +
                0.7 * row["surprise"] +
                0.5 * row["neutral"] +
                0.2 * row["fear"] +
                0.1 * row["sad"]
            )

            values.append(float(hi) * 100)

        return values

    except:
        return []

# =====================================================
# VIDEO GRAPH DATA (latest session)
# =====================================================
def get_last_face_scores():

    if not os.path.exists(FACE_DB):
        return []

    try:
        conn = sqlite3.connect(FACE_DB)

        query = """
        SELECT timestamp, Surprise, Fear, Disgust, Happy, Sad, Anger, Neutral
        FROM emotion_log
        ORDER BY timestamp ASC
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        if len(df) == 0:
            return []

        sessions = []
        current = []

        prev_time = None

        for _, row in df.iterrows():

            t = datetime.fromisoformat(row["timestamp"])

            if prev_time is None:
                current.append(row)

            else:
                gap = (t - prev_time).total_seconds()

                if gap > 60:
                    sessions.append(current)
                    current = [row]
                else:
                    current.append(row)

            prev_time = t

        if current:
            sessions.append(current)

        latest = pd.DataFrame(sessions[-1]).reset_index(drop=True)

        N = len(latest)

        if N == 0:
            return []

        step = N / 10.0

        multipliers = [1,3,5,7,9]

        values = []

        for m in multipliers:

            idx = math.ceil(step * m) - 1

            if idx >= N:
                idx = N - 1

            row = latest.iloc[idx]

            hi = (
                1.0 * row["Happy"] +
                0.7 * row["Surprise"] +
                0.5 * row["Neutral"] +
                0.3 * row["Disgust"] +
                0.2 * row["Fear"] +
                0.1 * row["Sad"]
            )

            values.append(float(hi) * 100)

        return values

    except:
        return []

# =====================================================
# SESSION STATE
# =====================================================
if "mode" not in st.session_state:
    st.session_state.mode = None

if "current_q" not in st.session_state:
    st.session_state.current_q = 0

if "hi_scores" not in st.session_state:
    st.session_state.hi_scores = []

if "completed" not in st.session_state:
    st.session_state.completed = False

# =====================================================
# TITLE
# =====================================================
st.title("Happiness Index System 📈")

# =====================================================
# HOME PAGE
# =====================================================
if st.session_state.mode is None:

    show_gif("speaking.gif")

    st.subheader("Select Mode")

    c1, c2 = st.columns(2)

    if c1.button("TEXT MODE"):
        st.session_state.mode = "text"
        st.rerun()

    if c2.button("VIDEO MODE"):
        st.session_state.mode = "video"
        st.rerun()

    st.stop()

# =====================================================
# FINAL RESULT PAGE
# =====================================================
if st.session_state.completed:

    avg_hi = sum(st.session_state.hi_scores) / len(st.session_state.hi_scores)

    st.success("Assessment Completed")

    st.subheader(f"Final Happiness Index: {avg_hi*100:.2f}")

    show_gif(final_gif(avg_hi))

    # TEXT MODE GRAPH
    if st.session_state.mode == "text":

        st.markdown("### Text Emotion Trend")

        values = get_last_5_text_scores()

        show_graph(values, "HI vs Question Number (Text)")

    # VIDEO MODE GRAPH
    elif st.session_state.mode == "video":

        st.markdown("### Facial Emotion Trend")

        values = get_last_face_scores()

        show_graph(values, "HI vs Question Number (Video)")

    if st.button("Restart"):

        st.session_state.mode = None
        st.session_state.current_q = 0
        st.session_state.hi_scores = []
        st.session_state.completed = False

        st.rerun()

    st.stop()

# =====================================================
# QUESTION PAGE
# =====================================================
q = st.session_state.current_q

show_gif("surprise.gif")

st.write(f"Question {q+1} of {len(QUESTIONS)}")
st.subheader(QUESTIONS[q])

# =====================================================
# TEXT MODE
# =====================================================
if st.session_state.mode == "text":

    answer = st.text_area(
        "Your Answer",
        key=f"textbox_{q}"
    )

    if st.button("Submit Answer"):

        if answer.strip() == "":
            st.warning("Please enter an answer.")
            st.stop()

        with st.spinner("Processing..."):
            hi = process_text_answer(answer)

        st.session_state.hi_scores.append(hi)

        st.session_state.current_q += 1

        if st.session_state.current_q >= len(QUESTIONS):
            st.session_state.completed = True

        st.rerun()

# =====================================================
# VIDEO MODE
# =====================================================
elif st.session_state.mode == "video":

    st.info("Uses Raspberry Pi camera + microphone")

    if st.button("Start Recording"):

        with st.spinner("Recording + Processing..."):
            hi = process_video_answer()

        st.session_state.hi_scores.append(hi)

        st.session_state.current_q += 1

        if st.session_state.current_q >= len(QUESTIONS):
            st.session_state.completed = True

        st.rerun()
