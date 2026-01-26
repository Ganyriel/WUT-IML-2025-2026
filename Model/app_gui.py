"""
SPEAKER VERIFICATION SYSTEM - GUI with Gradio
Graphical interface for real-time voice classification
Authorized vs Unauthorized

REQUIRED FILES in the same folder:
- speaker_verification_model.h5 (your trained model)
- app_gui.py (this file)
"""

import gradio as gr
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
MODEL_PATH = "speaker_verification_model.h5"
SAMPLE_RATE = 22050
MAX_DURATION = 3  # seconds
MEL_N_MELS = 128

# ==================== AUDIO UTILITIES ====================

def normalize_audio(y):
    return y / (np.max(np.abs(y)) + 1e-6)

def trim_audio(y):
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    return y_trimmed

def pad_audio(y, sr=SAMPLE_RATE, max_duration=MAX_DURATION):
    target_length = sr * max_duration
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    return y

def audio_to_spectrogram(y, sr=SAMPLE_RATE, n_mels=MEL_N_MELS):
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, fmax=8000
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db[..., np.newaxis]

def preprocess_audio(audio_input):
    """
    Preprocess audio input from Gradio
    audio_input: (sample_rate, audio_array)
    """
    if audio_input is None:
        raise ValueError("❌ Please upload or record an audio sample")

    sr, y = audio_input

    # Ensure float32 (FIXES THE ERROR)
    y = y.astype(np.float32)

    # Normalize if coming from int WAV
    if np.max(np.abs(y)) > 1.0:
        y = normalize_audio(y)

    # Resample if needed
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Remove silence
    y = trim_audio(y)

    # Pad or truncate
    y = pad_audio(y)

    return y

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return load_model(MODEL_PATH)

# ==================== MAIN PREDICTION ====================

def predict_voice(audio_input):
    try:
        audio_array = preprocess_audio(audio_input)
        spectrogram = audio_to_spectrogram(audio_array)
        X = np.expand_dims(spectrogram, axis=0)

        model = load_trained_model()
        prob = model.predict(X, verbose=0)[0][0]

        label = int(prob > 0.5)
        confidence = prob if label == 1 else (1 - prob)
        confidence_pct = confidence * 100

        if label == 1:
            result_text = f"""
# ✅ ACCESS GRANTED

**Status:** Authorized Speaker  
**Confidence:** {confidence_pct:.2f}%  
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
✔ Verification successful  
✔ Door unlocked  
✔ Access granted
"""
            actions = "✔ Verification successful\n✔ Door unlocked\n✔ Access granted"
        else:
            result_text = f"""
# ❌ ACCESS DENIED

**Status:** Unauthorized Speaker  
**Confidence:** {confidence_pct:.2f}%  
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
✖ Verification failed  
✖ Door remains locked  
✖ Security log updated
"""
            actions = "✖ Verification failed\n✖ Door locked\n✖ Attempt logged"

        return (
            result_text,
            confidence_pct,
            float(prob),
            str(label),
            actions
        )

    except Exception as e:
        return (
            f"### ❌ Error\n{str(e)}",
            0,
            0.0,
            "Error",
            "Contact system administrator"
        )

# ==================== GRADIO INTERFACE ====================

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="Voice Access Control System") as demo:

        gr.Markdown("# 🔐 Speaker Verification System\n### Secure Voice Authentication")

        audio_input = gr.Audio(
            label="Upload a WAV file or record your voice",
            type="numpy",
            sources=["upload", "microphone"]
        )

        submit_btn = gr.Button("🔍 VERIFY VOICE", variant="primary")

        result_output = gr.Markdown()
        confidence_out = gr.Number(label="Confidence (%)", precision=2)
        prob_out = gr.Number(label="Raw Probability", precision=4)
        label_out = gr.Textbox(label="Prediction (0=Unauthorized, 1=Authorized)")
        actions_out = gr.Textbox(label="Recommended Actions", lines=3)

        submit_btn.click(
            fn=predict_voice,
            inputs=audio_input,
            outputs=[
                result_output,
                confidence_out,
                prob_out,
                label_out,
                actions_out
            ]
        )

    return demo

# ==================== MAIN ====================

if __name__ == "__main__":
    print("🔐 Starting Speaker Verification System")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        exit(1)

    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
