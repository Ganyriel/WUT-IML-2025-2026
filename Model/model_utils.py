import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models

# --- GLOBAL CONFIGURATION ---
SR = 22050
DURATION = 3.0  # Seconds
N_MELS = 128
HOP_LENGTH = 512
FMAX = 8000
TARGET_LENGTH = int(SR * DURATION)


def load_manifest(base_path):
    """
    Loads the manifest CSV file and adds a 'speaker' column extracted from the file path.

    Args:
        base_path (str): Path to the root folder containing the 'manifest.csv'.

    Returns:
        pd.DataFrame: DataFrame containing file paths, labels, and speaker IDs.
    """
    manifest_path = os.path.join(base_path, 'manifest.csv')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    df = pd.read_csv(manifest_path)
    # Extract speaker ID from path (e.g., accepted/p001/file.wav -> p001)
    df['speaker'] = df['path'].apply(lambda p: os.path.basename(os.path.dirname(p)))
    return df


def preprocess_audio(y, sr=SR, target_length=TARGET_LENGTH):
    """
    Normalizes, trims, and pads/truncates the audio signal to a fixed length.
    This function is used during both training data loading and live inference.

    Args:
        y (np.array): Raw audio signal.
        sr (int): Sampling rate.
        target_length (int): Desired length of the audio array (samples).

    Returns:
        np.array: Preprocessed audio signal.
    """
    # 1. Amplitude normalization
    y = y / (np.max(np.abs(y)) + 1e-6)

    # 2. Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # 3. Pad or Truncate to target length
    if len(y) < target_length:
        # Pad with zeros at the end
        y = np.pad(y, (0, target_length - len(y)))
    else:
        # Truncate
        y = y[:target_length]

    return y


def load_and_process_data(df, base_path, sr=SR):
    """
    Loads audio files listed in the DataFrame, processes them, and generates mel-spectrograms.

    Args:
        df (pd.DataFrame): Dataframe with 'path', 'label', and 'speaker' columns.
        base_path (str): Root directory of the dataset.
        sr (int): Sampling rate.

    Returns:
        tuple: (X, y, speakers) where X contains spectrograms, y contains labels.
    """
    audio_data = []
    labels = []
    speakers_list = []
    failed = 0

    print(f"Loading {len(df)} audio files...")

    for idx, row in df.iterrows():
        full_path = os.path.join(base_path, row['path'])
        try:
            # Load audio using librosa
            y, _ = librosa.load(full_path, sr=sr, duration=DURATION)

            # Apply preprocessing
            y = preprocess_audio(y, sr=sr)

            audio_data.append(y)
            labels.append(row['label'])
            speakers_list.append(row['speaker'])
        except Exception as e:
            failed += 1

    print(f"Successfully loaded: {len(audio_data)}, Failed: {failed}")

    # Convert to Mel-Spectrograms
    print("Generating Mel-Spectrograms...")
    spectrograms = []
    for y in audio_data:
        # Generate Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX, hop_length=HOP_LENGTH)
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # Add channel dimension (Height, Width, 1) for CNN input
        spectrograms.append(mel_spec_db[..., np.newaxis])

    return np.array(spectrograms), np.array(labels), np.array(speakers_list)


def get_recording_group_key(path, speaker):
    """
    Helper function to group recordings by their original session ID to prevent data leakage during splitting.
    Given a relative path like 'accepted/p226/p226_003_000.wav', return a deterministic group key.

    For most speakers: group by (speaker, original_recording_id) where
    original_recording_id is the middle number in 'p226_003_000'.

    For accepted/p001: treat each file as its own group (no special grouping).
    """

    norm_path = path.replace("\\", "/")
    basename = os.path.basename(norm_path)
    stem, _ = os.path.splitext(basename)
    parts = stem.split("_")

    # Manual recordings (e.g., p001) - treat each file independently
    if "/accepted/" in norm_path and speaker == "p001":
        return f"{speaker}|{stem}"

    # Auto-split pattern: pXXX_###_### (middle number is the recording ID)
    if len(parts) >= 3 and parts[0] == speaker and parts[1].isdigit() and parts[2].isdigit():
        original_id = parts[1]
        return f"{speaker}|{original_id}"

    return f"{speaker}|{stem}"  # Fallback


def split_dataset(X, y, speakers, df, train_ratio=0.7, val_ratio=0.15):
    """
    Splits the dataset into train, validation, and test sets.
    Ensures that segments from the same original recording stay in the same split.

    Args:
        X (np.array): Input features (spectrograms).
        y (np.array): Labels.
        speakers (np.array): Speaker IDs.
        df (pd.DataFrame): Original dataframe (used for paths).

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test)
    """
    # Group indices by speaker and recording session
    groups_per_speaker = {}
    for idx, spk in enumerate(speakers):
        path = df.iloc[idx]["path"]
        group_key = get_recording_group_key(path, spk)

        if spk not in groups_per_speaker:
            groups_per_speaker[spk] = {}
        if group_key not in groups_per_speaker[spk]:
            groups_per_speaker[spk][group_key] = []
        groups_per_speaker[spk][group_key].append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for spk in sorted(groups_per_speaker.keys()):
        group_dict = groups_per_speaker[spk]
        # Sort for deterministic results
        grouped_items = sorted(group_dict.items(), key=lambda kv: kv[0])

        all_indices = [i for _, idxs in grouped_items for i in idxs]
        n_total = len(all_indices)

        target_train = int(round(train_ratio * n_total))
        target_val = int(round(val_ratio * n_total))
        n_train = n_val = 0

        for group_key, idxs in grouped_items:
            gsize = len(idxs)
            if n_train + gsize <= target_train:
                train_idx.extend(idxs)
                n_train += gsize
            elif n_val + gsize <= target_val:
                val_idx.extend(idxs)
                n_val += gsize
            else:
                test_idx.extend(idxs)

    # Convert indices to numpy arrays for indexing
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    # Return data AND corresponding DataFrames (metadata)
    return (
        X[train_idx], X[val_idx], X[test_idx],
        y[train_idx], y[val_idx], y[test_idx],
        df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy()
    )


def make_lr_callbacks(use_plateau=False):
    callbacks = []
    if use_plateau:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ))
    return callbacks

def build_model(
    input_shape,
    optimizer_name='adam',
    dropout_rate=0.0,
    learning_rate=0.001,
    lr_schedule=None,          # None | "cosine" | "exp"
    weight_decay=0.0,          # only used for AdamW
    steps_per_epoch=None,      # needed for cosine
    total_epochs=None,         # needed for cosine
    adapt_data=None            # pass X_train to adapt() normalization
):
    """
    Builds a CNN model with configurable hyperparameters + optional LR schedules and AdamW weight decay.

    Args:
        input_shape (tuple): Shape of the input data (H, W, C).
        optimizer_name (str): 'adam', 'sgd', or 'adamw'.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Base learning rate (or initial LR for schedules).
        lr_schedule (str|None): None | "cosine" | "exp". (Plateau is a callback, not here.)
        weight_decay (float): Weight decay for AdamW (e.g. 1e-4).
        steps_per_epoch (int|None): Required for cosine schedule.
        total_epochs (int|None): Required for cosine schedule.
        adapt_data (np.ndarray|tf.Tensor|None): If provided, adapt() the Normalization layer.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # Internal normalization layer (optionally adapted)
    norm = layers.Normalization()
    model.add(norm)

    # Convolutional Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))

    # Convolutional Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))

    # Convolutional Block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))

    # Classifier Head
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    if dropout_rate > 0:
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    # --- Learning rate: constant or schedule ---
    lr = learning_rate
    if lr_schedule == "cosine":
        if steps_per_epoch is None or total_epochs is None:
            raise ValueError("lr_schedule='cosine' requires steps_per_epoch and total_epochs.")
        lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=steps_per_epoch * total_epochs
        )
    elif lr_schedule == "exp":
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )
    elif lr_schedule is not None:
        raise ValueError("lr_schedule must be None, 'cosine', or 'exp'.")

    # --- Optimizer ---
    opt_name = optimizer_name.lower()
    if opt_name == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif opt_name == 'adamw':
        opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    else:
        print(f"Unknown optimizer {optimizer_name}, defaulting to Adam.")
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    # adapt normalization if data provided ---

    if adapt_data is not None:
        norm.adapt(adapt_data)

    return model
