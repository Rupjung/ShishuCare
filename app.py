import streamlit as st
import os
import soundfile as sf
import tempfile
import numpy as np
import librosa
import time
import logging
import subprocess

from util import predict_audio
from auth import login_user, register_user
from db import insert_result

# ------------------------- CONFIGURATION -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

st.set_page_config("👶 Infant Cry Detector", layout="wide")

# ------------------------- FFmpeg CONFIG -------------------------
# Replace with your actual ffmpeg.exe path
FFMPEG_PATH = r"C:\Users\LOQ\ffmpeg-6.0-full_build\bin\ffmpeg.exe"

def has_ffmpeg() -> bool:
    try:
        subprocess.run(
            [FFMPEG_PATH, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except Exception:
        return False

HAS_FFMPEG = has_ffmpeg()

# ------------------------- UTILITIES -------------------------
def _validate_readable_wav(path: str) -> bool:
    """Try loading with librosa to ensure the WAV is readable."""
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        return isinstance(y, np.ndarray) and y.size > 0 and isinstance(sr, (int, np.integer))
    except Exception as e:
        logger.error(f"WAV validation failed: {e}")
        return False

# ------------------------- AUDIO PROCESSING -------------------------
def convert_to_wav(uploaded_file):
    """Convert audio/video (m4a, mp4, mp3, wav, ogg, flac, aac) to mono 22.05kHz WAV."""
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    temp_in = tempfile.mktemp(suffix=file_ext if file_ext else ".bin")
    temp_out = tempfile.mktemp(suffix=".wav")

    try:
        # Save uploaded file
        with open(temp_in, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Use FFmpeg first if available
        if HAS_FFMPEG:
            cmd = [
                FFMPEG_PATH, "-y", "-i", temp_in,
                "-vn",           # drop video if present
                "-ac", "1",      # mono
                "-ar", "22050",  # 22.05 kHz
                "-f", "wav",
                temp_out
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0 and os.path.exists(temp_out) and _validate_readable_wav(temp_out):
                return temp_out
            else:
                logger.error("FFmpeg failed:\n" + result.stderr.decode("utf-8", errors="ignore"))

        # If WAV, try pure Python standardization
        if file_ext == ".wav":
            try:
                y, sr = librosa.load(temp_in, sr=None, mono=True)
                if sr != 22050:
                    y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                    sr = 22050
                sf.write(temp_out, y, sr)
                if _validate_readable_wav(temp_out):
                    return temp_out
            except Exception as e:
                logger.error(f"Pure-Python WAV re-write failed: {e}")

        # Last fallback using librosa
        try:
            y, sr = librosa.load(temp_in, sr=22050, mono=True)
            sf.write(temp_out, y, 22050)
            if _validate_readable_wav(temp_out):
                return temp_out
        except Exception as e:
            logger.error(f"Librosa fallback failed: {e}")

        return None

    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return None
    finally:
        if temp_in and os.path.exists(temp_in):
            try: os.unlink(temp_in)
            except: pass

# ------------------------- AUTH PAGES -------------------------
def login_page():
    st.title("🔑 Login")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if not username or not password:
                st.error("Please enter both username and password")
                return
            user = login_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid credentials")

def signup_page():
    st.title("📝 Create Account")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        new_user = st.text_input("Choose Username")
        new_pass = st.text_input("Choose Password", type="password")
        confirm_pass = st.text_input("Confirm Password", type="password")
        if st.button("Register", use_container_width=True):
            if not new_user or not new_pass:
                st.error("Please fill all fields")
                return
            if new_pass != confirm_pass:
                st.error("Passwords don't match!")
                return
            if len(new_pass) < 6:
                st.error("Password must be at least 6 characters")
                return
            success = register_user(new_user, new_pass)
            if success:
                st.success("Account created! Please login.")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Username already exists")

# ------------------------- DASHBOARD -------------------------
def dashboard():
    st.title(f"👶 Baby Cry Detection - Welcome {st.session_state.username}")

    if not HAS_FFMPEG:
        st.warning("⚠️ FFmpeg not detected. m4a/mp4 may fail. Install FFmpeg and ensure it is on PATH.")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("### 📤 Upload Audio File for Analysis")
        st.info("Supported formats: WAV, MP3, MP4, M4A, OGG, FLAC, AAC")

        audio_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "mp4", "m4a", "ogg", "flac", "aac"]
        )

        if audio_file:
            st.audio(audio_file)
            if st.button("🔍 Analyze Audio", use_container_width=True):
                wav_path = None
                try:
                    with st.spinner("Converting audio to WAV format..."):
                        wav_path = convert_to_wav(audio_file)
                        if not wav_path:
                            st.error("❌ Audio conversion failed. Check FFmpeg path or try another file.")
                            return

                    with st.spinner("Analyzing audio..."):
                        label, confidence = predict_audio(wav_path)
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                        st.markdown("---")
                        st.markdown("### 📊 Analysis Results")
                        if label.lower() == 'cry':
                            if confidence > 0.8:
                                st.error("🚨 **BABY CRY DETECTED!**")
                            elif confidence > 0.6:
                                st.warning("⚠️ **Possible Baby Cry**")
                            else:
                                st.info("🤔 **Uncertain Cry Detection**")
                        else:
                            st.success(f"✅ Sound Classified as: {label.title()}")

                        st.markdown(f"**Confidence:** {confidence:.2%}")
                        st.markdown(f"**Timestamp:** {timestamp}")

                        if st.session_state.logged_in:
                            try:
                                insert_result(st.session_state.username, label, confidence)
                                st.success("✅ Result saved to your history!")
                            except Exception as e:
                                st.warning(f"Could not save to database: {str(e)}")

                        st.markdown("### 📈 Confidence Level")
                        st.progress(min(max(float(confidence), 0.0), 1.0))

                finally:
                    if wav_path and os.path.exists(wav_path):
                        try: os.unlink(wav_path)
                        except: pass

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        logout()

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("Logged out successfully!")
    time.sleep(1)
    st.rerun()

# ------------------------- APP ENTRY -------------------------
def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""

    if not st.session_state.logged_in:
        st.sidebar.title("Navigation")
        auth_page = st.sidebar.radio("Choose an option", ["Login", "Sign Up"])
        if auth_page == "Login":
            login_page()
        else:
            signup_page()
    else:
        dashboard()

if __name__ == "__main__":
    main()
