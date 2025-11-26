"""Model training script: loads preprocessed patches from artifacts and trains CAE, saving model + history.
Run standalone or import and call main().
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

ARTIFACT_DIR = os.environ.get("CAE_ARTIFACT_DIR", "artifacts")
MODEL_PATH = os.environ.get("CAE_MODEL_PATH", "cae_model.h5")
BATCH_SIZE = int(os.environ.get("CAE_BATCH_SIZE", 16))  # lowered default batch size to reduce memory
EPOCHS = int(os.environ.get("CAE_EPOCHS", 10))  # lowered default epochs per user request
MAX_TRAIN_PATCHES = int(os.environ.get("CAE_MAX_TRAIN_PATCHES", 0))  # optional secondary cap
VALIDATION_SPLIT = float(os.environ.get("CAE_VALIDATION_SPLIT", 0.1))

def build_cae(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(8, (3, 3), activation='relu', padding='same', name='encoded')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(input_img, decoded)

def main():
    print("[Model] Starting training phase")
    x_train_path = os.path.join(ARTIFACT_DIR, "X_train.npy")
    if not os.path.isfile(x_train_path):
        raise FileNotFoundError(f"{x_train_path} not found. Run preprocessing first.")
    # Memory-friendly load
    X_train = np.load(x_train_path, mmap_mode=None)
    original_count = X_train.shape[0]
    if MAX_TRAIN_PATCHES > 0 and original_count > MAX_TRAIN_PATCHES:
        print(f"[Model] Capping training patches from {original_count} to {MAX_TRAIN_PATCHES}")
        sel = np.random.choice(original_count, MAX_TRAIN_PATCHES, replace=False)
        X_train = X_train[sel]
    print(f"[Model] Training patches: {X_train.shape[0]} of original {original_count}")
    print(f"[Model] Patch shape: {X_train.shape[1:]}  dtype: {X_train.dtype}")
    # Enable GPU memory growth if possible
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            print(f"[Model] Enabled memory growth on {len(gpus)} GPU(s)")
    except Exception as e:
        print(f"[Model] GPU memory growth setup failed: {e}")
    # Clear previous TF graphs
    K.clear_session()
    input_shape = X_train.shape[1:]
    cae_model = build_cae(input_shape)
    cae_model.compile(optimizer='adam', loss='mse')
    cae_model.summary()
    history = cae_model.fit(
        X_train, X_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        shuffle=True
    )
    cae_model.save(MODEL_PATH)
    hist_path = os.path.join(ARTIFACT_DIR, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump({"loss": history.history.get("loss", []), "val_loss": history.history.get("val_loss", [])}, f, indent=2)
    print(f"Model saved to {MODEL_PATH}; history saved to {hist_path}")

if __name__ == "__main__":
    main()