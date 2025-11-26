import tensorflow as tf
import os
from tensorflow.keras.utils import image_dataset_from_directory
import config as CFG
from pathlib import Path
import random
import numpy as np
from sklearn.model_selection import train_test_split

def _list_labeled_paths(root: Path):
    """Return (paths, labels) from labeled_data/0_NOISE and labeled_data/1_SHIP."""
    classes = [('0_NOISE', 0), ('1_SHIP', 1)]
    paths, labels = [], []
    for sub, lab in classes:
        d = root / sub
        if not d.is_dir():
            raise FileNotFoundError(f"Missing class folder: {d}")
        for p in sorted(d.glob('*')):
            if p.is_file():
                paths.append(str(p))
                labels.append(lab)
    return np.array(paths), np.array(labels)

def load_and_split_data():
    """
    Tactical split:
    - Identify the single Day A ship image (filename starts with '00' and label==1) -> always in final TEST_SET.
    - Remaining data (all noise + Day B ships) is stratified into TRAIN/VAL/TEST per fractions.
    - Returns train_ds, val_ds, test_ds, test_paths, test_labels.
    Augmentation is applied only to TRAIN_DS.
    """
    labeled_root = Path(getattr(CFG, 'LABELED_DATA_DIR', 'labeled_data'))
    print(f"Loading labeled data from: {labeled_root}")
    paths, labels = _list_labeled_paths(labeled_root)

    # Identify Day A (prefix '00') and Day B ('12' or '13') by filename prefix
    def prefix(fname):
        base = os.path.basename(fname)
        return base[:2]

    is_day_a_ship = np.array([(prefix(p) == '00') and (lab == 1) for p, lab in zip(paths, labels)])
    day_a_ship_paths = paths[is_day_a_ship]
    day_a_ship_labels = labels[is_day_a_ship]
    if len(day_a_ship_paths) == 0:
        print("Warning: no Day A ship images found. Proceeding without forced test inclusion.")
    else:
        print(f"Day A ship images found: {len(day_a_ship_paths)}")

    # Remaining data excludes the Day A ship image
    remain_mask = ~is_day_a_ship
    remain_paths = paths[remain_mask]
    remain_labels = labels[remain_mask]

    # Stratified split: first split off TEST fraction
    test_frac = float(getattr(CFG, 'TEST_SIZE_FRACTION', 0.20))
    val_frac = float(getattr(CFG, 'VALIDATION_SIZE_FRACTION', 0.15))

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        remain_paths, remain_labels, test_size=test_frac, stratify=remain_labels, random_state=getattr(CFG, 'RANDOM_SEED', 42))

    # Now split train/val from trainval
    # val_frac relative to remaining pool
    val_rel = val_frac / (1.0 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_rel, stratify=y_trainval, random_state=getattr(CFG, 'RANDOM_SEED', 42))

    # Final TEST must include at least one Day A ship (if present). Remaining Day A ships go to validation.
    X_test_final = list(X_test)
    y_test_final = list(y_test)
    if len(day_a_ship_paths) > 0:
        # Ensure at least one Day A ship in test
        X_test_final.append(day_a_ship_paths[0])
        y_test_final.append(1)
        # Place any additional Day A ships into validation set
        if len(day_a_ship_paths) > 1:
            X_val = list(X_val)
            y_val = list(y_val)
            for extra in day_a_ship_paths[1:]:
                X_val.append(extra)
                y_val.append(1)
            X_val = np.array(X_val)
            y_val = np.array(y_val)

    # Build tf.data datasets from file paths
    image_size = getattr(CFG, 'IMAGE_SIZE', (256, 256))
    batch_size = getattr(CFG, 'BATCH_SIZE', 32)

    def make_ds(file_paths, labels_arr, augment=False):
        # Create a tf.data.Dataset from (path, label)
        ds_paths = tf.data.Dataset.from_tensor_slices(file_paths)
        ds_labels = tf.data.Dataset.from_tensor_slices(labels_arr.astype('float32'))
        ds = tf.data.Dataset.zip((ds_paths, ds_labels))

        def _load(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=getattr(CFG, 'CHANNELS', 3))
            img = tf.image.resize(img, image_size, method='bilinear')
            img = tf.cast(img, tf.float32) / 255.0
            if augment:
                img = tf.keras.Sequential([
                    tf.keras.layers.RandomZoom(0.05),
                    tf.keras.layers.RandomTranslation(0.05, 0.02),
                    tf.keras.layers.RandomContrast(0.1),
                ])(img)
            return img, tf.expand_dims(label, axis=-1)

        ds = ds.map(lambda p, y: _load(p, y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        # Optional caching; only safe if full dataset consumed each epoch
        if getattr(CFG, 'CACHE_DATASETS', False):
            ds = ds.cache()
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_ds(np.array(X_train), np.array(y_train), augment=True)
    val_ds   = make_ds(np.array(X_val),   np.array(y_val),   augment=False)
    test_ds  = make_ds(np.array(X_test_final), np.array(y_test_final), augment=False)

    return train_ds, val_ds, test_ds, X_test_final, y_test_final
    
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

# ensure_test_split and load_test_data are no longer used in tactical split; keeping for legacy if referenced elsewhere.

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