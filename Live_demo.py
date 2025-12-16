import sys
import os
import numpy as np
import tensorflow as tf
import librosa
import sounddevice as sd
import threading
from datetime import datetime

# --- ROBUST ENVIRONMENT DETECTION ---
IS_JUPYTER = False
try:
    # Check if running in a standard IPython/Jupyter environment
    from IPython import get_ipython
    if get_ipython() is not None:
        from IPython.display import clear_output
        IS_JUPYTER = True
except ImportError:
    pass

# Add Helpers to path
sys.path.append(os.path.abspath('Model'))  # Adjusted path for script execution
import model_utils as mu

# --- CONFIGURATION ---
SR = mu.SR  # 22050
DURATION = mu.DURATION  # 3.0
N_MELS = mu.N_MELS  # 128
FMAX = mu.FMAX  # 8000
HOP_LENGTH = mu.HOP_LENGTH  # 512

UPDATE_INTERVAL = 0.5
BUFFER_SIZE = int(SR * DURATION)
MODEL_PATH = "Model/results/model_Baseline_Adam.keras"

# --- 1. LOAD MODEL ---
if not os.path.exists(MODEL_PATH):
    # Try alternate path if running from root
    MODEL_PATH = "Model/results/model_Baseline_Adam.keras"
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found. Checked path: {MODEL_PATH}")

print(f"Loading model from: {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# Warmup
dummy_input = np.zeros((1, 128, 130, 1))
_ = model.predict(dummy_input, verbose=0)
print("Model loaded and ready!")


# --- 2. AUDIO BUFFER ---
class AudioRingBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float32)
        self.lock = threading.Lock()

    def extend(self, new_data):
        with self.lock:
            new_data = new_data.flatten()
            n = len(new_data)
            if n >= self.size:
                self.buffer = new_data[-self.size:]
            else:
                self.buffer = np.roll(self.buffer, -n)
                self.buffer[-n:] = new_data

    def get(self):
        with self.lock:
            return self.buffer.copy()


audio_buffer = AudioRingBuffer(BUFFER_SIZE)


# --- 3. CALLBACK ---
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_buffer.extend(indata)


# --- 4. PREDICTION LOOP ---
def run_continuous_monitoring():
    print("\n" + "=" * 50)
    print("   CONTINUOUS VOICE SECURITY SYSTEM - ACTIVE   ")
    print("=" * 50)
    print(f"Listening... (Press Ctrl+C to quit)")

    stream = sd.InputStream(
        samplerate=SR,
        channels=1,
        callback=callback,
        blocksize=int(SR * UPDATE_INTERVAL)
    )

    with stream:
        try:
            while True:
                # 1. Get audio
                raw_audio = audio_buffer.get()

                # Default values
                reset = "\033[0m"
                color = "\033[0m"

                # Check for silence/initialization
                if np.max(np.abs(raw_audio)) < 0.005:
                    label = "Silence / Initializing..."
                    confidence = 0.0
                    color = "\033[90m"  # Gray
                    bar = "[Waiting for sound...]"
                else:
                    # 2. Process
                    processed_audio = mu.preprocess_audio(raw_audio, sr=SR)

                    mel_spec = librosa.feature.melspectrogram(
                        y=processed_audio, sr=SR, n_mels=N_MELS, fmax=FMAX, hop_length=HOP_LENGTH
                    )
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                    # Resize
                    target_width = 130
                    if mel_spec_db.shape[1] < target_width:
                        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, target_width - mel_spec_db.shape[1])))
                    else:
                        mel_spec_db = mel_spec_db[:, :target_width]

                    model_input = mel_spec_db.reshape(1, N_MELS, target_width, 1)

                    # 3. Predict
                    prediction_prob = model.predict(model_input, verbose=0)[0][0]

                    # 4. Logic
                    if prediction_prob > 0.5:
                        label = "ACCEPTED (ACCESS GRANTED)"
                        confidence = prediction_prob
                        color = "\033[92m"  # Green
                    else:
                        label = "REJECTED (ACCESS DENIED) "
                        confidence = 1 - prediction_prob
                        color = "\033[91m"  # Red

                    # Visual bar
                    bar_len = 20
                    filled = int(confidence * bar_len)
                    bar = "[" + "=" * filled + " " * (bar_len - filled) + "]"

                # --- DISPLAY ---

                # Clear screen logic
                if IS_JUPYTER:
                    # Jupyter Notebook method
                    clear_output(wait=True)
                else:
                    # ANSI Escape Codes for Terminal / PyCharm Console
                    # \033[2J clears the entire screen
                    # \033[H moves cursor to top-left
                    print("\033[2J\033[H", end="")
                    # Terminal clear command (cls for Windows, clear for Linux/Mac)
                    os.system('cls' if os.name == 'nt' else 'clear')

                print("=" * 50)
                print(f"📡 STATUS: MONITORING MICROPHONE")
                print(f"⏰ {datetime.now().strftime('%H:%M:%S.%f')[:-4]}")
                print("-" * 50)
                print(f"{color}RESULT: {label} {reset}")
                print(f"CONFIDENCE: {confidence:.1%} {bar}")
                print("-" * 50)

                sd.sleep(int(UPDATE_INTERVAL * 1000))

        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user.")


if __name__ == "__main__":
    run_continuous_monitoring()