"""Preprocessing script: loads DAS HDF5 data, builds train/test sets, extracts patches, saves artifacts.
Run standalone or import and call main().
"""

import os
import sys
import glob
import bisect
import h5py
import json
import pickle
import numpy as np
import gc
from sklearn.preprocessing import MinMaxScaler  # retained for compatibility if env forces scaler usage
from tqdm import tqdm

# Ensure parent directory (project root) is on path for Helpers
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PARENT_DIR not in sys.path:
    sys.path.append(_PARENT_DIR)
from Helpers import extract_metadata

# Configuration (can be overridden by environment variables if desired)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_DEFAULT_BASE_DIR = os.path.join(_PROJECT_ROOT, "data", "GC_data")
raw_base_dir = os.environ.get("CAE_BASE_DIR")
BASE_DIR = raw_base_dir if raw_base_dir else _DEFAULT_BASE_DIR  # robust absolute path
print(f"[Preprocessing] Working dir: {os.getcwd()}")
print(f"[Preprocessing] Project root inferred: {_PROJECT_ROOT}")
print(f"[Preprocessing] Using BASE_DIR: {BASE_DIR}")
if not os.path.isdir(BASE_DIR):
    raise FileNotFoundError(f"BASE_DIR does not exist: {BASE_DIR}. Override with CAE_BASE_DIR or fix directory structure.")
TEST_SUBDIR = os.environ.get("CAE_TEST_SUBDIR", "20250702")
DMIN_KM = float(os.environ.get("CAE_DMIN_KM", 20.0))  # default narrowed start km
DMAX_KM = float(os.environ.get("CAE_DMAX_KM", 22.0))
SAMPLE_STEP = int(os.environ.get("CAE_SAMPLE_STEP", 1))
CHANNEL_STEP = int(os.environ.get("CAE_CHANNEL_STEP", 1))
TIME_PATCH_STEPS = int(os.environ.get("CAE_TIME_PATCH", 128))
DISTANCE_PATCH_STEPS = int(os.environ.get("CAE_DISTANCE_PATCH", 128))
PATCH_OVERLAP = float(os.environ.get("CAE_PATCH_OVERLAP", 0.1))
ARTIFACT_DIR = os.environ.get("CAE_ARTIFACT_DIR", "artifacts")
MAX_TRAIN_FILES = int(os.environ.get("CAE_MAX_TRAIN_FILES", 30))  # cap number of files
MAX_TEST_FILES = int(os.environ.get("CAE_MAX_TEST_FILES", 10))
MAX_TRAIN_TIME_STEPS = int(os.environ.get("CAE_MAX_TRAIN_TIME_STEPS", 60000))  # cap time steps post-load
MAX_TEST_TIME_STEPS = int(os.environ.get("CAE_MAX_TEST_TIME_STEPS", 60000))
USE_FLOAT16 = os.environ.get("CAE_USE_FLOAT16", "1") == "1"
MAX_TRAIN_PATCHES = int(os.environ.get("CAE_MAX_TRAIN_PATCHES", 0))  # 0 means no cap
MAX_TEST_PATCHES = int(os.environ.get("CAE_MAX_TEST_PATCHES", 0))  # 0 means no cap
NORMALIZE_CHUNK_ROWS = int(os.environ.get("CAE_NORMALIZE_CHUNK_ROWS", 30000))  # chunk size for normalization; 0 disables chunking
STREAM_MODE = os.environ.get("CAE_STREAM_MODE", "0") == "1"  # if enabled, avoid building large combined matrices

os.makedirs(ARTIFACT_DIR, exist_ok=True)

def discover_files():
    all_files = sorted(glob.glob(os.path.join(BASE_DIR, "**", "*.hdf5"), recursive=True))
    test_files = sorted(glob.glob(os.path.join(BASE_DIR, TEST_SUBDIR, "*.hdf5")))
    train_files = [p for p in all_files if TEST_SUBDIR not in os.path.relpath(p, BASE_DIR).split(os.path.sep)]
    return train_files, test_files

def build_distance_mask(ref_file):
    _, dt, dx, channels_info, _ = extract_metadata(ref_file)
    if np.isscalar(channels_info):
        channel_indices = np.arange(int(channels_info))
    else:
        channel_indices = np.asarray(channels_info)
    distance_array_km = channel_indices * dx / 1000.0
    dist_mask = (distance_array_km >= DMIN_KM) & (distance_array_km <= DMAX_KM)
    if dist_mask.sum() == 0:
        raise RuntimeError(f"No channels found in km range {DMIN_KM}-{DMAX_KM}.")
    return dist_mask, dx, dt, distance_array_km

def load_and_combine(paths, dist_mask, sample_step=1, channel_step=1, max_time_steps=None, return_lengths=False):
    arrays = []
    lengths = []
    total_rows = 0
    for p in tqdm(paths, desc="Loading files"):
        with h5py.File(p, "r") as f:
            data = f['data'][::sample_step, dist_mask][:, ::channel_step]
        original_len = data.shape[0]
        if max_time_steps is not None and total_rows + original_len > max_time_steps:
            remaining = max_time_steps - total_rows
            if remaining <= 0:
                break
            data = data[:remaining]
            original_len = data.shape[0]
        arrays.append(data)
        lengths.append(original_len)
        total_rows += original_len
        if max_time_steps is not None and total_rows >= max_time_steps:
            break
    if not arrays:
        raise RuntimeError("No arrays loaded.")
    stacked = np.vstack(arrays) if len(arrays) > 1 else arrays[0]
    if return_lengths:
        return stacked, lengths
    return stacked

def create_patches_and_labels(data_matrix, time_window, dist_window, overlap_factor, is_test_set=False):
    patches = []
    labels = []
    time_steps, dist_channels = data_matrix.shape
    time_step_size = max(1, int(time_window * (1 - overlap_factor)))
    dist_step_size = max(1, int(dist_window * (1 - overlap_factor)))
    for t in range(0, time_steps - time_window + 1, time_step_size):
        for d in range(0, dist_channels - dist_window + 1, dist_step_size):
            patch = data_matrix[t:t + time_window, d:d + dist_window]
            if patch.shape == (time_window, dist_window):
                patches.append(patch)
                if is_test_set:
                    labels.append(1 if np.max(patch) > 0.8 else 0)
                else:
                    labels.append(0)
    if not patches:
        return np.empty((0, time_window, dist_window), dtype=np.float32), np.empty((0,), dtype=int)
    return np.asarray(patches), np.asarray(labels)

def main():
    train_files, test_files = discover_files()
    if MAX_TRAIN_FILES > 0 and len(train_files) > MAX_TRAIN_FILES:
        print(f"Capping training files from {len(train_files)} to {MAX_TRAIN_FILES}")
        train_files = train_files[:MAX_TRAIN_FILES]
    if MAX_TEST_FILES > 0 and len(test_files) > MAX_TEST_FILES:
        print(f"Capping test files from {len(test_files)} to {MAX_TEST_FILES}")
        test_files = test_files[:MAX_TEST_FILES]
    print(f"Found {len(train_files)} training files and {len(test_files)} test files.")
    if not train_files:
        raise RuntimeError("No training files found — cannot proceed.")
    ref_file = train_files[0] if train_files else test_files[0]
    dist_mask, dx, dt, distance_array_km = build_distance_mask(ref_file)
    print(f"Selected km window {DMIN_KM}-{DMAX_KM} km -> {dist_mask.sum()} channels")

    if STREAM_MODE:
        print("[Preprocessing] STREAM_MODE enabled: two-pass min/max then per-file patch extraction.")
        # First pass: compute global min/max without storing large arrays
        global_min = None
        global_max = None
        train_lengths = []
        used_time_steps = 0
        for p in train_files:
            with h5py.File(p, 'r') as f:
                data = f['data'][:, dist_mask][:, ::CHANNEL_STEP]
            if MAX_TRAIN_TIME_STEPS and used_time_steps + data.shape[0] > MAX_TRAIN_TIME_STEPS:
                data = data[:MAX_TRAIN_TIME_STEPS - used_time_steps]
            cur_min = data.min()
            cur_max = data.max()
            global_min = cur_min if global_min is None else min(global_min, cur_min)
            global_max = cur_max if global_max is None else max(global_max, cur_max)
            train_lengths.append(data.shape[0])
            used_time_steps += data.shape[0]
            if MAX_TRAIN_TIME_STEPS and used_time_steps >= MAX_TRAIN_TIME_STEPS:
                break
        if global_min is None:
            raise RuntimeError("No training data read during min/max pass.")
        min_val, max_val = global_min, global_max
        range_val = max_val - min_val if max_val != min_val else 1.0
        print(f"[Preprocessing] Global min={min_val}, max={max_val}")
        # Second pass: generate normalized patches per file
        train_patches_list = []
        total_train_patches = 0
        used_time_steps = 0
        for file_idx, p in enumerate(train_files):
            with h5py.File(p, 'r') as f:
                data = f['data'][:, dist_mask][:, ::CHANNEL_STEP]
            if MAX_TRAIN_TIME_STEPS and used_time_steps + data.shape[0] > MAX_TRAIN_TIME_STEPS:
                data = data[:MAX_TRAIN_TIME_STEPS - used_time_steps]
            used_time_steps += data.shape[0]
            # normalize current data chunk
            norm_chunk = (data.astype(np.float32) - min_val) / range_val
            if USE_FLOAT16:
                norm_chunk = norm_chunk.astype(np.float16)
            # create patches inside this file (no cross-file windows)
            file_patches, _labels = create_patches_and_labels(norm_chunk, TIME_PATCH_STEPS, DISTANCE_PATCH_STEPS, PATCH_OVERLAP, is_test_set=False)
            if file_patches.size:
                train_patches_list.append(file_patches)
                total_train_patches += file_patches.shape[0]
            if MAX_TRAIN_PATCHES > 0 and total_train_patches >= MAX_TRAIN_PATCHES:
                print(f"[Preprocessing] Train patch cap reached ({total_train_patches} >= {MAX_TRAIN_PATCHES}).")
                break
            if MAX_TRAIN_TIME_STEPS and used_time_steps >= MAX_TRAIN_TIME_STEPS:
                break
        if not train_patches_list:
            X_train_raw = np.empty((0, TIME_PATCH_STEPS, DISTANCE_PATCH_STEPS), dtype=np.float32)
        else:
            X_train_raw = np.concatenate(train_patches_list, axis=0)
        # No combined_train retained; free memory
        normalized_train = None
        gc.collect()
    else:
        combined_train, train_lengths = load_and_combine(train_files, dist_mask, sample_step=SAMPLE_STEP, channel_step=CHANNEL_STEP, max_time_steps=MAX_TRAIN_TIME_STEPS, return_lengths=True)
        print("Combined train shape (raw):", combined_train.shape, combined_train.dtype)
        if combined_train.shape[0] > MAX_TRAIN_TIME_STEPS:
            print(f"Cropping train time steps from {combined_train.shape[0]} to {MAX_TRAIN_TIME_STEPS}")
            combined_train = combined_train[:MAX_TRAIN_TIME_STEPS]
        # Manual min-max stats
        min_val = combined_train.min()
        max_val = combined_train.max()
        range_val = max_val - min_val if max_val != min_val else 1.0
        # Chunked normalization to reduce peak memory
        if NORMALIZE_CHUNK_ROWS > 0:
            rows = combined_train.shape[0]
            dtype_target = np.float16 if USE_FLOAT16 else np.float32
            normalized_train = np.empty((rows, combined_train.shape[1]), dtype=dtype_target)
            print(f"Normalizing train in chunks of {NORMALIZE_CHUNK_ROWS} rows -> target dtype {dtype_target}")
            for start in range(0, rows, NORMALIZE_CHUNK_ROWS):
                end = min(start + NORMALIZE_CHUNK_ROWS, rows)
                block = combined_train[start:end]
                tmp = (block.astype(np.float32) - min_val) / range_val
                if USE_FLOAT16:
                    tmp = tmp.astype(np.float16)
                normalized_train[start:end] = tmp
            del tmp, block  # release references
        else:
            normalized_train = (combined_train.astype(np.float32) - min_val) / range_val
            if USE_FLOAT16:
                normalized_train = normalized_train.astype(np.float16)
        X_train_raw, _ = create_patches_and_labels(normalized_train, TIME_PATCH_STEPS, DISTANCE_PATCH_STEPS, PATCH_OVERLAP, is_test_set=False)
        if MAX_TRAIN_PATCHES > 0 and X_train_raw.shape[0] > MAX_TRAIN_PATCHES:
            print(f"Capping train patches from {X_train_raw.shape[0]} to {MAX_TRAIN_PATCHES}")
            sel = np.random.choice(X_train_raw.shape[0], MAX_TRAIN_PATCHES, replace=False)
            X_train_raw = X_train_raw[sel]
        # free large arrays
        del combined_train, normalized_train
        gc.collect()

    if test_files:
        combined_test, test_lengths = load_and_combine(test_files, dist_mask, sample_step=SAMPLE_STEP, channel_step=CHANNEL_STEP, max_time_steps=MAX_TEST_TIME_STEPS, return_lengths=True)
        print("Combined test shape (raw):", combined_test.shape, combined_test.dtype)
        if combined_test.shape[0] > MAX_TEST_TIME_STEPS:
            print(f"Cropping test time steps from {combined_test.shape[0]} to {MAX_TEST_TIME_STEPS}")
            combined_test = combined_test[:MAX_TEST_TIME_STEPS]
        if NORMALIZE_CHUNK_ROWS > 0:
            rows_t = combined_test.shape[0]
            dtype_target = np.float16 if USE_FLOAT16 else np.float32
            normalized_test = np.empty((rows_t, combined_test.shape[1]), dtype=dtype_target)
            print(f"Normalizing test in chunks of {NORMALIZE_CHUNK_ROWS} rows -> target dtype {dtype_target}")
            for start in range(0, rows_t, NORMALIZE_CHUNK_ROWS):
                end = min(start + NORMALIZE_CHUNK_ROWS, rows_t)
                block = combined_test[start:end]
                tmp = (block.astype(np.float32) - min_val) / range_val
                if USE_FLOAT16:
                    tmp = tmp.astype(np.float16)
                normalized_test[start:end] = tmp
            del tmp, block
        else:
            normalized_test = (combined_test.astype(np.float32) - min_val) / range_val
            if USE_FLOAT16:
                normalized_test = normalized_test.astype(np.float16)
        print("Combined test shape (normalized):", normalized_test.shape, normalized_test.dtype)
    else:
        normalized_test = None
        print("No test files found in test subdir.")

    # Build cumulative boundaries for train (not currently saved, but could be extended)
    train_boundaries = [0]
    for L in train_lengths:
        train_boundaries.append(train_boundaries[-1] + L)

    # Prepare test set (streaming if enabled)
    if STREAM_MODE and test_files:
        print("[Preprocessing] STREAM_MODE test processing.")
        test_lengths = []
        used_test_time = 0
        test_patches_list = []
        patch_file_indices = []
        y_test_simulated_list = []
        for file_idx, p in enumerate(test_files):
            with h5py.File(p, 'r') as f:
                data = f['data'][:, dist_mask][:, ::CHANNEL_STEP]
            if MAX_TEST_TIME_STEPS and used_test_time + data.shape[0] > MAX_TEST_TIME_STEPS:
                data = data[:MAX_TEST_TIME_STEPS - used_test_time]
            used_test_time += data.shape[0]
            norm_chunk = (data.astype(np.float32) - min_val) / range_val
            if USE_FLOAT16:
                norm_chunk = norm_chunk.astype(np.float16)
            # create patches for this file
            file_patches, file_labels = create_patches_and_labels(norm_chunk, TIME_PATCH_STEPS, DISTANCE_PATCH_STEPS, PATCH_OVERLAP, is_test_set=True)
            if file_patches.size:
                test_patches_list.append(file_patches)
                y_test_simulated_list.append(file_labels)
                patch_file_indices.extend([file_idx] * file_patches.shape[0])
            if MAX_TEST_PATCHES > 0 and sum(x.shape[0] for x in test_patches_list) >= MAX_TEST_PATCHES:
                print("[Preprocessing] Test patch cap reached.")
                break
            if MAX_TEST_TIME_STEPS and used_test_time >= MAX_TEST_TIME_STEPS:
                break
        if test_patches_list:
            X_test_raw = np.concatenate(test_patches_list, axis=0)
            y_test_simulated = np.concatenate(y_test_simulated_list, axis=0)
            patch_file_indices = np.asarray(patch_file_indices, dtype=np.int32)
        else:
            X_test_raw = np.empty((0, TIME_PATCH_STEPS, DISTANCE_PATCH_STEPS), dtype=np.float32)
            y_test_simulated = np.empty((0,), dtype=int)
            patch_file_indices = np.empty((0,), dtype=np.int32)
        normalized_test = None
        gc.collect()
    elif not STREAM_MODE and normalized_test is not None:
        # Build cumulative boundaries for test files to map patch start indices to file index
        test_boundaries = [0]
        for L in test_lengths:
            test_boundaries.append(test_boundaries[-1] + L)

        # Wrapper to capture file mapping
        patch_file_indices = []
        def create_patches_and_labels_with_mapping(data_matrix, time_window, dist_window, overlap_factor):
            patches = []
            labels = []
            time_steps, dist_channels = data_matrix.shape
            time_step_size = max(1, int(time_window * (1 - overlap_factor)))
            dist_step_size = max(1, int(dist_window * (1 - overlap_factor)))
            for t in range(0, time_steps - time_window + 1, time_step_size):
                for d in range(0, dist_channels - dist_window + 1, dist_step_size):
                    patch = data_matrix[t:t + time_window, d:d + dist_window]
                    if patch.shape == (time_window, dist_window):
                        patches.append(patch)
                        labels.append(1 if np.max(patch) > 0.8 else 0)
                        # Determine file index by locating which boundary t falls into
                        file_idx = bisect.bisect_right(test_boundaries, t) - 1
                        patch_file_indices.append(file_idx)
            if not patches:
                return np.empty((0, time_window, dist_window), dtype=np.float32), np.empty((0,), dtype=int)
            return np.asarray(patches), np.asarray(labels)

        X_test_raw, y_test_simulated = create_patches_and_labels_with_mapping(normalized_test, TIME_PATCH_STEPS, DISTANCE_PATCH_STEPS, PATCH_OVERLAP)
        # Ensure mapping converted before any slicing
        patch_file_indices = np.asarray(patch_file_indices, dtype=np.int32)
        if MAX_TEST_PATCHES > 0 and X_test_raw.shape[0] > MAX_TEST_PATCHES:
            print(f"Capping test patches from {X_test_raw.shape[0]} to {MAX_TEST_PATCHES}")
            sel_t = np.random.choice(X_test_raw.shape[0], MAX_TEST_PATCHES, replace=False)
            X_test_raw = X_test_raw[sel_t]
            # adjust mapping arrays accordingly (guard length match)
            if patch_file_indices.shape[0] == sel_t.shape[0]:
                patch_file_indices = patch_file_indices[sel_t]
            else:
                patch_file_indices = patch_file_indices[sel_t]
            if y_test_simulated.shape[0] == sel_t.shape[0]:
                y_test_simulated = y_test_simulated[sel_t]
    else:
        X_test_raw, y_test_simulated = np.empty((0, TIME_PATCH_STEPS, DISTANCE_PATCH_STEPS)), np.empty((0,), dtype=int)
        patch_file_indices = np.empty((0,), dtype=np.int32)

    X_train = X_train_raw[..., np.newaxis].astype(np.float32 if not USE_FLOAT16 else np.float16)
    X_test = X_test_raw[..., np.newaxis].astype(np.float32 if not USE_FLOAT16 else np.float16)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test_simulated shape:", y_test_simulated.shape)

    # Persist artifacts
    np.save(os.path.join(ARTIFACT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(ARTIFACT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(ARTIFACT_DIR, "y_test_simulated.npy"), y_test_simulated)
    # Save mapping artifacts
    with open(os.path.join(ARTIFACT_DIR, "test_files.json"), "w") as f:
        json.dump([os.path.basename(p) for p in test_files], f, indent=2)
    np.save(os.path.join(ARTIFACT_DIR, "test_patch_file_indices.npy"), patch_file_indices)
    # Save normalization metadata instead of sklearn scaler
    norm_meta = {"min_val": float(min_val), "max_val": float(max_val), "range_val": float(range_val), "dtype": str(X_train.dtype)}
    with open(os.path.join(ARTIFACT_DIR, "normalization.json"), "w") as f:
        json.dump(norm_meta, f, indent=2)
    config = {
        "BASE_DIR": BASE_DIR,
        "TEST_SUBDIR": TEST_SUBDIR,
        "DMIN_KM": DMIN_KM,
        "DMAX_KM": DMAX_KM,
        "SAMPLE_STEP": SAMPLE_STEP,
        "CHANNEL_STEP": CHANNEL_STEP,
        "TIME_PATCH_STEPS": TIME_PATCH_STEPS,
        "DISTANCE_PATCH_STEPS": DISTANCE_PATCH_STEPS,
        "PATCH_OVERLAP": PATCH_OVERLAP,
        "dx": dx,
        "dt": dt,
        "MAX_TRAIN_FILES": MAX_TRAIN_FILES,
        "MAX_TEST_FILES": MAX_TEST_FILES,
        "MAX_TRAIN_TIME_STEPS": MAX_TRAIN_TIME_STEPS,
        "MAX_TEST_TIME_STEPS": MAX_TEST_TIME_STEPS,
        "USE_FLOAT16": USE_FLOAT16,
        "MAX_TRAIN_PATCHES": MAX_TRAIN_PATCHES,
        "MAX_TEST_PATCHES": MAX_TEST_PATCHES,
        "STREAM_MODE": STREAM_MODE,
        "channels_selected": int(dist_mask.sum())
    }
    with open(os.path.join(ARTIFACT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Artifacts saved in {ARTIFACT_DIR}")

if __name__ == "__main__":
    main()