import streamlit as st
import tempfile
import os
import whisper
import torch
import threading
import time
import subprocess
import re

st.set_page_config(
    page_title="Transcription Web App",
    page_icon="🎧",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align: center;'>Transcription Web App</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Supports Taglish, Tagalog, and English.</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Developed by: Jairo Devon Daquioag</p>",
    unsafe_allow_html=True
)
st.write("This site is still under development — only the basic features are available.")
st.write("For concerns and suggestions, please contact the developer at daquioagjairo30@gmail.com")


uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "ogg"])


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Loading Whisper Large-v3 model on {device.upper()}... (this may take a while)")
    model = whisper.load_model("large-v3", device=device)
    st.success("Model loaded successfully!")
    return model, device


if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    st.write(f"**File:** {uploaded_file.name} ({uploaded_file.size / (1024 * 1024):.2f} MB)")

    model, device = load_model()

    # Get audio duration using ffprobe
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", temp_path
    ]
    duration = float(subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip())

    # Estimate transcription time
    speed_factor = 0.25 if device == "cuda" else 2.0  # rough scaling
    estimated_seconds = duration * speed_factor

    # Display initial ETA
    est_minutes = int(estimated_seconds // 60)
    est_seconds = int(estimated_seconds % 60)
    eta_text = st.empty()
    eta_text.info(f"Estimated transcription time: {est_minutes} min {est_seconds} sec")

    result_container = {"text": ""}

    def transcribe_audio():
        # FP16 on GPU for safe speedup
        fp16 = True if device == "cuda" else False
        options = dict(
            fp16=fp16,
            verbose=False,
            condition_on_previous_text=False,
            suppress_blank=False,
            temperature=0,
            language=None,
            word_timestamps=True
        )
        result = model.transcribe(temp_path, **options)
        result_container["text"] = result["text"]

    # Start transcription in background thread
    thread = threading.Thread(target=transcribe_audio)
    thread.start()

    start_time = time.time()

    # Update ETA live while transcription is running
    while thread.is_alive():
        elapsed = time.time() - start_time
        remaining = max(0, estimated_seconds - elapsed)
        est_minutes = int(remaining // 60)
        est_seconds = int(remaining % 60)
        eta_text.info(f"Estimated time remaining: {est_minutes} min {est_seconds} sec")
        time.sleep(1)

    thread.join()
    eta_text.success("✅ Transcription completed!")

    transcript = result_container["text"]

    # Normalize laughter/filler patterns (verbatim)
    laugh_patterns = [
        r"\b(fancy|pansy|fanci|fansy|fancy po|pansipo|hahaha|haha|hehe|he he|ha ha|hah)\b",
    ]
    for pattern in laugh_patterns:
        transcript = re.sub(pattern, "(laughs)", transcript, flags=re.IGNORECASE)

    transcript = re.sub(r"(\(laughs\)\s*){2,}", "(laughs) ", transcript)
    transcript = re.sub(r"\s+", " ", transcript).strip()

    st.text_area("Transcript", transcript, height=400)

    os.remove(temp_path)

    st.download_button(
        label="📥 Download Transcript",
        data=transcript,
        file_name="transcript.txt",
        mime="text/plain"
    )