import os
import sys
from datetime import datetime

import gradio as gr
import numpy as np
import librosa
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..", "Model")))
import model_utils as mu

MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "results", "model_AdamW_Plateau.keras"))

CAPTURE_SECONDS = 5
SILENCE_TOP_DB = 35
MIN_NON_SILENT_SECONDS = 0.6

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
_, EXPECTED_MELS, EXPECTED_WIDTH, EXPECTED_CH = model.input_shape
if EXPECTED_CH != 1:
    raise ValueError(f"Unexpected model input shape: {model.input_shape}")

_ = model.predict(np.zeros((1, EXPECTED_MELS, EXPECTED_WIDTH, 1), dtype=np.float32), verbose=0)

def _mono_float32(y):
    y = np.asarray(y)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.5:
        y = y / (peak + 1e-6)
    return y

def _non_silent_seconds(y, sr):
    intervals = librosa.effects.split(y, top_db=SILENCE_TOP_DB)
    if len(intervals) == 0:
        return 0.0
    return float(sum((e - s) for s, e in intervals) / sr)

def _make_model_input(y_proc, sr):
    mel = librosa.feature.melspectrogram(
        y=y_proc, sr=sr, n_mels=mu.N_MELS, fmax=mu.FMAX, hop_length=mu.HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    if mel_db.shape[0] != EXPECTED_MELS:
        raise ValueError(f"n_mels mismatch: got {mel_db.shape[0]} expected {EXPECTED_MELS}")

    if mel_db.shape[1] < EXPECTED_WIDTH:
        mel_db = np.pad(mel_db, ((0, 0), (0, EXPECTED_WIDTH - mel_db.shape[1])), mode="constant")
    elif mel_db.shape[1] > EXPECTED_WIDTH:
        mel_db = mel_db[:, :EXPECTED_WIDTH]

    return mel_db[np.newaxis, ..., np.newaxis]

def _decide_mapping_on_validation():
    return None

def predict(audio_input):
    if audio_input is None:
        return "No audio."

    sr, y = audio_input
    y = _mono_float32(y)

    if sr != mu.SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=mu.SR)
        sr = mu.SR

    cap_n = int(CAPTURE_SECONDS * sr)
    if len(y) < cap_n:
        y = np.pad(y, (0, cap_n - len(y)))
    else:
        y = y[:cap_n]

    ns = _non_silent_seconds(y, sr)
    if ns < MIN_NON_SILENT_SECONDS:
        return f"Too silent ({ns:.2f}s non-silent). Record again closer to mic."

    y_proc = mu.preprocess_audio(y, sr=sr)
    X = _make_model_input(y_proc, sr)

    prob = float(model.predict(X, verbose=0)[0][0])

    status = "ACCEPTED" if prob > 0.5 else "REJECTED"
    conf = prob if prob > 0.5 else (1.0 - prob)

    ts = datetime.now().strftime("%H:%M:%S")
    return (
        f"{status}\n"
        f"time={ts}\n"
        f"P(class=1)={prob:.4f}\n"
        f"confidence={conf*100:.2f}%\n"
        f"non_silent={ns:.2f}s\n"
        f"capture={CAPTURE_SECONDS:.1f}s\n"
        f"model_segment={mu.DURATION:.1f}s"
    )

with gr.Blocks(title="Speaker Verification") as demo:
    gr.Markdown(f"# Speaker Verification\nRecord ~{CAPTURE_SECONDS}s then Predict.")
    audio = gr.Audio(type="numpy", sources=["microphone", "upload"], label="Record / Upload")
    btn = gr.Button("Predict", variant="primary")
    out = gr.Textbox(label="Result", lines=8)
    btn.click(fn=predict, inputs=[audio], outputs=[out])

if __name__ == "__main__":
    demo.launch(show_error=True)
