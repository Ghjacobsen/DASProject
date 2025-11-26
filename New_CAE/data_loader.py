import tensorflow as tf
import os
from tensorflow.keras.utils import image_dataset_from_directory
import config as CFG
from pathlib import Path
import random

def load_train_validation_data():
    """
    Loads labeled training images, splits them into training and validation sets,
    and applies simple data augmentation.
    """
    print(f"Loading and preparing data from: {getattr(CFG, 'TRAIN_DIR', 'train')}")
    
    # 1. Load data and split into train/validation
    # The subdirectories '0_NOISE' and '1_SHIP' define the classes.
    train_ds = image_dataset_from_directory(
        getattr(CFG, 'TRAIN_DIR', 'train'),
        labels='inferred',
        label_mode='binary',
        image_size=getattr(CFG, 'IMAGE_SIZE', (256, 256)),
        interpolation='bilinear',
        batch_size=getattr(CFG, 'BATCH_SIZE', 32),
        shuffle=True,
        seed=getattr(CFG, 'RANDOM_SEED', 42),
        validation_split=getattr(CFG, 'VALIDATION_SPLIT', 0.2),
        subset='training'
    )

    val_ds = image_dataset_from_directory(
        getattr(CFG, 'TRAIN_DIR', 'train'),
        labels='inferred',
        label_mode='binary',
        image_size=getattr(CFG, 'IMAGE_SIZE', (256, 256)),
        interpolation='bilinear',
        batch_size=getattr(CFG, 'BATCH_SIZE', 32),
        shuffle=True,
        seed=getattr(CFG, 'RANDOM_SEED', 42),
        validation_split=getattr(CFG, 'VALIDATION_SPLIT', 0.2),
        subset='validation'
    )
    
    # 2. Rescaling and Augmentation
    # Mild, physically-consistent augmentation (no flips/large rotations)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.RandomTranslation(0.05, 0.02),
        tf.keras.layers.RandomContrast(0.1),
    ])

    # Function to apply preprocessing (rescale, then augment for training only)
    def process(image, label, is_training):
        # 1. Rescale pixel values from 0-255 to 0-1
        image = tf.cast(image / 255.0, tf.float32)
        # 2. Apply augmentation if in training mode
        if is_training:
            image = data_augmentation(image)
        return image, label

    # Apply preprocessing to datasets
    train_ds = train_ds.map(lambda x, y: process(x, y, True), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: process(x, y, False), num_parallel_calls=tf.data.AUTOTUNE)

    # Optimization: Prefetching data improves pipeline speed
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    print(f"Training Samples: {len(train_ds) * getattr(CFG, 'BATCH_SIZE', 32)}")
    print(f"Validation Samples: {len(val_ds) * getattr(CFG, 'BATCH_SIZE', 32)}")
    
    return train_ds, val_ds

def ensure_test_split(move_count_per_class=None, seed=None):
    """
    Ensure a small test holdout exists by moving N samples from each class
    directory under TRAIN_DIR into TEST_DIR (mirrors class subfolders).
    Idempotent: will only move up to the shortfall if test already has files.
    """
    rng = random.Random(seed if seed is not None else getattr(CFG, 'RANDOM_SEED', 42))
    train_root = Path(getattr(CFG, 'TRAIN_DIR', 'train'))
    test_root = Path(getattr(CFG, 'TEST_DIR', 'test'))
    test_root.mkdir(exist_ok=True)

    n_per = move_count_per_class if move_count_per_class is not None else getattr(CFG, 'TEST_SAMPLES_PER_CLASS', 5)
    classes = ['0_NOISE', '1_SHIP']
    for cls in classes:
        src = train_root / cls
        dst = test_root / cls
        dst.mkdir(parents=True, exist_ok=True)
        if not src.is_dir():
            raise FileNotFoundError(f"Missing class folder: {src}")

        # Current counts
        dst_files = sorted([p for p in dst.glob('*') if p.is_file()])
        shortfall = max(0, n_per - len(dst_files))
        if shortfall == 0:
            continue

        candidates = sorted([p for p in src.glob('*') if p.is_file()])
        if len(candidates) == 0:
            raise FileNotFoundError(f"No files available in {src} to move for test split.")

        pick = rng.sample(candidates, k=min(shortfall, len(candidates)))
        for p in pick:
            p.rename(dst / p.name)
        print(f"Moved {len(pick)} files from {src} -> {dst}")

def load_test_data():
    """
    Load the held-out test images from TEST_DIR with inferred binary labels,
    return dataset (rescaled), filenames and numeric labels.
    """
    test_root = Path(getattr(CFG, 'TEST_DIR', 'test'))
    if not test_root.exists():
        raise FileNotFoundError(f"Test directory not found: {test_root}")

    ds = image_dataset_from_directory(
        str(test_root),
        labels='inferred',
        label_mode='binary',
        image_size=getattr(CFG, 'IMAGE_SIZE', (256, 256)),
        interpolation='bilinear',
        batch_size=getattr(CFG, 'BATCH_SIZE', 32),
        shuffle=False
    )

    ds = ds.map(lambda x, y: (tf.cast(x / 255.0, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Build a deterministic file list and labels
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    paths = []
    for root, _, files in os.walk(test_root):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))
    paths.sort()
    filenames = [os.path.basename(p) for p in paths]
    labels = [1 if ('1_SHIP' in Path(p).parts) else 0 for p in paths]
    return ds, filenames, labels

def load_unseen_data():
    """
    Loads the final, unlabeled 'unseen_data' for anomaly prediction.
    """
    unseen_root = Path(getattr(CFG, 'UNSEEN_DIR', 'unseen_data'))
    print(f"Loading unseen data from: {unseen_root}")
    
    # Use image_dataset_from_directory without validation_split to load all files
    # The 'labels=None' indicates that this data is unlabeled.
    unseen_ds = image_dataset_from_directory(
        str(unseen_root),
        labels=None,
        image_size=getattr(CFG, 'IMAGE_SIZE', (256, 256)),
        interpolation='bilinear',
        batch_size=getattr(CFG, 'BATCH_SIZE', 32),
        shuffle=False # Do not shuffle for prediction to match filenames
    )
    
    # Rescale pixel values from 0-255 to 0-1
    unseen_ds = unseen_ds.map(lambda x: tf.cast(x / 255.0, tf.float32))
    
    # Optimization
    unseen_ds = unseen_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Build a deterministic, sorted list of underlying files (matching Keras order: alpha-sorted)
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    paths = []
    for root, _, files in os.walk(unseen_root):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))
    # Sort by full path for deterministic order
    paths.sort()
    # Return only basenames for reporting
    filenames = [os.path.basename(p) for p in paths]

    return unseen_ds, filenames