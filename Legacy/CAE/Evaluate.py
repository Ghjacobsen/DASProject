"""Evaluation script: loads artifacts + trained model and produces metrics + anomaly rankings.
Run standalone or import and call main().
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.models import load_model

ARTIFACT_DIR = os.environ.get("CAE_ARTIFACT_DIR", "artifacts")
MODEL_PATH = os.environ.get("CAE_MODEL_PATH", "cae_model.h5")
PERCENTILE = float(os.environ.get("CAE_THRESHOLD_PERCENTILE", 99.9))
TOP_K = int(os.environ.get("CAE_TOP_K", 5))

# Resolve relative paths based on script location to avoid silent early exit when run from another CWD.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(ARTIFACT_DIR):
    ARTIFACT_DIR = os.path.join(SCRIPT_DIR, ARTIFACT_DIR)
if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(SCRIPT_DIR, MODEL_PATH)

def _log_start():
    print("[Evaluate] CWD:", os.getcwd())
    print("[Evaluate] Script dir:", SCRIPT_DIR)
    print("[Evaluate] Using ARTIFACT_DIR:", ARTIFACT_DIR)
    print("[Evaluate] Using MODEL_PATH:", MODEL_PATH)
    print(f"[Evaluate] Threshold percentile: {PERCENTILE}  TOP_K: {TOP_K}")

def main():
    _log_start()
    x_train_path = os.path.join(ARTIFACT_DIR, "X_train.npy")
    x_test_path = os.path.join(ARTIFACT_DIR, "X_test.npy")
    y_test_path = os.path.join(ARTIFACT_DIR, "y_test_simulated.npy")
    if not os.path.isdir(ARTIFACT_DIR):
        print(f"[Evaluate] Artifact directory not found: {ARTIFACT_DIR}")
        return
    if not os.path.isfile(MODEL_PATH):
        print(f"[Evaluate] Model file not found: {MODEL_PATH}")
        return
    if not os.path.isfile(x_train_path) or not os.path.isfile(x_test_path):
        print("[Evaluate] Required arrays missing (X_train.npy / X_test.npy). Run preprocessing & training first.")
        return

    X_train = np.load(x_train_path)
    X_test = np.load(x_test_path)
    y_test_simulated = np.load(y_test_path) if os.path.isfile(y_test_path) else np.empty((0,), dtype=int)

    # Optional mapping artifacts for file-level aggregation
    test_files = []
    patch_file_indices = None
    test_files_json = os.path.join(ARTIFACT_DIR, "test_files.json")
    patch_indices_npy = os.path.join(ARTIFACT_DIR, "test_patch_file_indices.npy")
    if os.path.isfile(test_files_json) and os.path.isfile(patch_indices_npy):
        try:
            with open(test_files_json, "r") as f:
                test_files = json.load(f)
            patch_file_indices = np.load(patch_indices_npy)
            print(f"[Evaluate] Loaded mapping: {len(test_files)} test files, {patch_file_indices.shape[0]} patch indices.")
        except Exception as e:
            print(f"[Evaluate] Warning: failed to load mapping artifacts: {e}")

    cae_model = load_model(MODEL_PATH)
    X_train_pred = cae_model.predict(X_train, verbose=0)
    train_error = np.mean(np.square(X_train - X_train_pred), axis=(1, 2, 3))
    if len(X_test) > 0:
        X_test_pred = cae_model.predict(X_test, verbose=0)
        test_error = np.mean(np.square(X_test - X_test_pred), axis=(1, 2, 3))
    else:
        X_test_pred = np.empty((0,))
        test_error = np.empty((0,))

    if train_error.size == 0:
        print("[Evaluate] Empty training error array; aborting.")
        return
    anomaly_threshold = np.percentile(train_error, PERCENTILE)
    print(f"Train errors: mean={train_error.mean():.6e}, std={train_error.std():.6e}")
    print(f"Anomaly threshold (train {PERCENTILE}th percentile): {anomaly_threshold:.6e}")

    metrics = {}
    if len(test_error) > 0 and len(y_test_simulated) == len(test_error) and len(y_test_simulated) > 0:
        y_pred_binary = (test_error > anomaly_threshold).astype(int)
        metrics = {
            "precision": float(precision_score(y_test_simulated, y_pred_binary, zero_division=0)),
            "recall": float(recall_score(y_test_simulated, y_pred_binary, zero_division=0)),
            "f1": float(f1_score(y_test_simulated, y_pred_binary, zero_division=0)),
            "accuracy": float(accuracy_score(y_test_simulated, y_pred_binary))
        }
        print("\nEvaluation metrics (simulated labels):")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    # File-level ranking (if mapping available)
    top_files = []
    if patch_file_indices is not None and len(test_error) == patch_file_indices.shape[0] and test_files:
        per_file_errors = {}
        for patch_idx, file_idx in enumerate(patch_file_indices):
            err = float(test_error[patch_idx])
            if file_idx not in per_file_errors:
                per_file_errors[file_idx] = []
            per_file_errors[file_idx].append(err)
        file_scores = []
        for fi, errs in per_file_errors.items():
            arr = np.asarray(errs)
            score_p95 = float(np.percentile(arr, 95))
            score_max = float(arr.max())
            file_scores.append({
                "file": test_files[fi],
                "p95_error": score_p95,
                "max_error": score_max,
                "num_patches": int(len(errs))
            })
        # Rank by p95 (stable) then max as tie-breaker
        file_scores.sort(key=lambda x: (x["p95_error"], x["max_error"]), reverse=True)
        top_files = file_scores[:TOP_K]
        print(f"\nTop {len(top_files)} anomalous test files (by p95 patch error):")
        for rank, entry in enumerate(top_files, 1):
            print(f"{rank}. {entry['file']} | p95={entry['p95_error']:.6e} | max={entry['max_error']:.6e} | patches={entry['num_patches']}")
    else:
        if len(test_error) > 0:
            print("[Evaluate] Patch-file mapping unavailable; skipping file-level ranking.")

    top_anomalies = []
    if len(test_error) > 0:
        top_idx = np.argsort(test_error)[::-1][:min(TOP_K, len(test_error))]
        print(f"\nTop {len(top_idx)} anomalous patch indices (in X_test):")
        for rank, idx in enumerate(top_idx, 1):
            label = int(y_test_simulated[idx]) if idx < len(y_test_simulated) else None
            score = float(test_error[idx])
            print(f"{rank}. idx={idx}, score={score:.6e}, simulated_label={label}")
            top_anomalies.append({"rank": rank, "index": int(idx), "score": score, "simulated_label": label})

        # Plot top anomaly original vs reconstruction
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        best = top_idx[0]
        orig = X_test[best].squeeze()
        recon = X_test_pred[best].squeeze()
        vmax = max(orig.max(), recon.max())
        axes[0].imshow(orig, aspect='auto', cmap='viridis', vmax=vmax)
        axes[0].set_title('Original (top anomalous)')
        axes[1].imshow(recon, aspect='auto', cmap='viridis', vmax=vmax)
        axes[1].set_title('Reconstruction')
        plt.tight_layout()
        plot_path = os.path.join(ARTIFACT_DIR, "top_anomaly_plot.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Saved plot to {plot_path}")
    else:
        print("No test patches to evaluate.")

    results = {
        "threshold_percentile": PERCENTILE,
        "anomaly_threshold": float(anomaly_threshold),
        "train_error_mean": float(train_error.mean()),
        "train_error_std": float(train_error.std()),
        "metrics": metrics,
        "top_anomalies": top_anomalies,
        "top_files": top_files
    }
    try:
        with open(os.path.join(ARTIFACT_DIR, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        np.save(os.path.join(ARTIFACT_DIR, "train_error.npy"), train_error)
        np.save(os.path.join(ARTIFACT_DIR, "test_error.npy"), test_error)
        print(f"Results saved in {ARTIFACT_DIR}")
    except Exception as e:
        print(f"[Evaluate] Failed to save results: {e}")

if __name__ == "__main__":
    main()