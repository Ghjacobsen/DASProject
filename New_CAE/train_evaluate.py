import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import itertools
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from pathlib import Path

import config as CFG
try:
    CFG_PATH = Path(CFG.__file__).resolve()
except Exception:
    CFG_PATH = None

# --- Deterministic Seeding ---
import random
os.environ.setdefault('PYTHONHASHSEED', str(getattr(CFG, 'RANDOM_SEED', 42)))
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
random.seed(getattr(CFG, 'RANDOM_SEED', 42))
np.random.seed(getattr(CFG, 'RANDOM_SEED', 42))
tf.random.set_seed(getattr(CFG, 'RANDOM_SEED', 42))
from data_loader import load_and_split_data

# --- Custom F1-Score Metric (since Keras doesn't have a native one) ---
# We define a custom metric to track F1-score during training
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
        
# --- Model Definition ---
def build_cnn_model(learning_rate, dropout_rate, num_conv_layers):
    """
    Builds a flexible CNN model based on hyperparameter inputs.
    """
    model = Sequential()
    channels = 1 if getattr(CFG, 'GRAYSCALE_INPUT', False) else getattr(CFG, 'CHANNELS', 3)
    input_shape = (*getattr(CFG, 'IMAGE_SIZE', (256, 256)), channels)
    
    # Initial Conv Block (slightly wider + 5x5 kernel helps diagonal patterns)
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rate))

    # Dynamic Conv Blocks based on num_conv_layers
    for i in range(num_conv_layers - 1):
        # Increase deeper capacity for large inputs (128, 256, ...)
        filters = 128 * (2 ** i)
        model.add(Conv2D(filters, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    
    # Dense Layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    # Output layer (Binary Classification: Ship or Noise)
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    # Loss: focal or BCE
    if getattr(CFG, 'USE_FOCAL_LOSS', False):
        alpha = float(getattr(CFG, 'FOCAL_ALPHA', 0.25))
        gamma = float(getattr(CFG, 'FOCAL_GAMMA', 2.0))
        def focal_loss(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            w = alpha * tf.pow(1.0 - pt, gamma)
            return tf.reduce_mean(w * tf.keras.losses.binary_crossentropy(y_true, y_pred))
        loss_fn = focal_loss
    else:
        loss_fn = 'binary_crossentropy'
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), F1Score()]
    )
    return model

# --- Main Hyperparameter Search and Training ---
def run_hyperparameter_search():
    """
    Executes the grid search, trains models, and writes best-per-run rows to a global registry.
    """
    print("--- Starting Data Load (Tactical Split) ---")
    train_ds, val_ds, test_ds, test_paths, test_labels = load_and_split_data()
    results_dir = Path(getattr(CFG, 'RESULTS_DIR', 'results'))
    results_dir.mkdir(exist_ok=True)

    best_f1 = -1.0
    best_params = {}
    best_threshold = 0.5
    best_epoch = None
    registry_rows = []

    # Setup combinations for grid search
    keys, values = zip(*getattr(CFG, 'HYPERPARAMETER_SPACE', {}).items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n--- Starting Hyperparameter Search ({len(combinations)} total runs) ---")

    # Timestamp id for this entire search
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    runs_dir = Path('runs'); runs_dir.mkdir(exist_ok=True)
    
    results = []
    
    best_run_row = None
    for run_num, params in enumerate(combinations):
        print(f"\n[Run {run_num + 1}/{len(combinations)}] Testing parameters: {params}")
        tf.keras.backend.clear_session()
        
        # 1. Build and Compile Model
        model = build_cnn_model(**params)
        
        # 2. Callbacks for Stability and Best Performance Tracking
        callbacks = [
            EarlyStopping(monitor='val_f1_score', patience=getattr(CFG, 'PATIENCE', 5), mode='max', verbose=0),
            ModelCheckpoint(
                filepath=str(Path(getattr(CFG, 'RESULTS_DIR', 'results')) / 'temp_best_model.h5'),
                monitor='val_f1_score',
                mode='max',
                save_best_only=True
            )
        ]
        
        # 3. Train Model
        history = model.fit(
            train_ds,
            epochs=getattr(CFG, 'EPOCHS_PER_RUN', 15),
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1 # Use 1 for progress bar logging
        )
        
        # 4. Evaluate and Log Results
        
        # Load the best weights from the current run (saved by ModelCheckpoint)
        model.load_weights(str(results_dir / 'temp_best_model.h5')) 
        
        # Evaluate on the validation set at default 0.5 for reference
        metrics = model.evaluate(val_ds, verbose=0)
        val_f1_at_05 = float(metrics[-1])

        # Collect validation labels and probabilities to tune threshold
        y_true = []
        for _, y in val_ds:
            y_true.append(y.numpy().reshape(-1))
        y_true = np.concatenate(y_true, axis=0)
        y_prob = model.predict(val_ds, verbose=0).reshape(-1)

        # Find best threshold maximizing F1
        thresholds = np.linspace(0.05, 0.95, 19)
        best_thr = 0.5
        best_f1_local = -1.0
        best_prec_local = 0.0
        best_rec_local = 0.0
        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, preds)
            if f1 > best_f1_local:
                best_f1_local = f1
                best_thr = float(t)
                best_prec_local = precision_score(y_true, preds)
                best_rec_local = recall_score(y_true, preds)

        # Additional metrics
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            roc_auc = float(roc_auc_score(y_true, y_prob))
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            roc_auc = float('nan'); pr_auc = float('nan')
        
        # Best epoch by Keras val_f1_score history
        hist_f1 = history.history.get('val_f1_score', [])
        best_epoch_idx = int(np.argmax(hist_f1)) if len(hist_f1) else None
        best_epoch_num = int(best_epoch_idx + 1) if best_epoch_idx is not None else None

        run_result = {
            'run_id': run_id,
            'combo_id': run_num + 1,
            'learning_rate': params['learning_rate'],
            'dropout_rate': params['dropout_rate'],
            'num_conv_layers': params['num_conv_layers'],
            'best_epoch': best_epoch_num,
            'val_f1_at_0.5': round(val_f1_at_05, 4),
            'val_f1_tuned': round(best_f1_local, 4),
            'val_precision_tuned': round(best_prec_local, 4),
            'val_recall_tuned': round(best_rec_local, 4),
            'best_threshold': best_thr,
            'roc_auc': round(roc_auc, 4) if roc_auc == roc_auc else None,
            'pr_auc': round(pr_auc, 4) if pr_auc == pr_auc else None,
        }
        results.append(run_result)
        print(f"Validation F1 (tuned) for Run {run_num + 1}: {best_f1_local:.4f} @ threshold {best_thr:.2f}")

        # 5. Track the overall best model
        if best_f1_local > best_f1:
            best_f1 = best_f1_local
            best_params = params
            best_threshold = best_thr
            best_epoch = best_epoch_num
            print(f"*** New CHAMPION (F1: {best_f1:.4f} @ {best_threshold:.2f}) with params: {best_params} ***")
            # Save champion model to results directory
            model.save(str(results_dir / 'champion_model.h5'))
            # Prepare champion row
            best_run_row = {
                'run_id': run_id,
                'combo_id': run_num + 1,
                'learning_rate': params['learning_rate'],
                'dropout_rate': params['dropout_rate'],
                'num_conv_layers': params['num_conv_layers'],
                'best_epoch': best_epoch_num,
                'val_f1_tuned': round(best_f1_local, 4),
                'best_threshold': best_threshold,
                'val_precision_tuned': round(best_prec_local, 4),
                'val_recall_tuned': round(best_rec_local, 4),
                'roc_auc': round(roc_auc, 4) if roc_auc == roc_auc else None,
                'pr_auc': round(pr_auc, 4) if pr_auc == pr_auc else None
            }
            
    # Print and Save Search Summary
    # Print summary table (optional) but only persist champion row
    results_df = pd.DataFrame(results)
    print("\n--- Hyperparameter Search Summary (All Runs) ---")
    print(results_df.to_string(index=False))

    if best_run_row is not None:
        try:
            registry = Path('runs') / 'registry.csv'
            pd.DataFrame([best_run_row]).to_csv(registry, mode='a', header=not registry.exists(), index=False)
            print(f"Champion row appended to {registry}")
        except Exception as e:
            print(f"Warning: failed to update registry with champion row: {e}")
    else:
        print("Warning: No champion row recorded (no runs?)")

    return best_params, best_threshold, test_ds, test_paths, test_labels

# --- Final Anomaly Prediction ---
def evaluate_on_test(best_params, threshold: float, test_ds, test_paths, test_labels):
    """
    Evaluate the champion model on the held-out test set.
    """
    print("\n--- Evaluating Champion on Test Holdout ---")

    results_dir = Path(getattr(CFG, 'RESULTS_DIR', 'results'))
    champion_path = results_dir / 'champion_model.h5'
    if champion_path.exists():
        best_model = tf.keras.models.load_model(str(champion_path), compile=False)
        best_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']),
                          loss='binary_crossentropy',
                          metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), F1Score()])
    else:
        best_model = build_cnn_model(**best_params)
        best_model.load_weights(getattr(CFG, 'MODEL_SAVE_PATH', 'best_ship_detector.h5'))

    probabilities = best_model.predict(test_ds, verbose=0).reshape(-1)
    preds = (probabilities >= threshold).astype(int)
    y_true = np.array(test_labels).astype(int)

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)

    report = pd.DataFrame({
        'filename': [os.path.basename(p) for p in test_paths],
        'true_label': y_true,
        'ship_probability': probabilities,
        'prediction': preds
    }).sort_values(by='ship_probability', ascending=False)
    # CRITICAL TEST: Day A Ship Classification (prefix '00' and true_label==1)
    critical_idx = None
    for i, (fname, tl) in enumerate(zip(report['filename'].tolist(), y_true.tolist())):
        if fname.startswith('00') and tl == 1:
            critical_idx = i
            break
    if critical_idx is not None:
        crit_fname = report['filename'].iloc[critical_idx]
        crit_prob = report['ship_probability'].iloc[critical_idx]
        crit_pred = 'SHIP' if report['prediction'].iloc[critical_idx] == 1 else 'NOISE'
        print(f"CRITICAL TEST: Day A Ship Classification → file={crit_fname}, prob={crit_prob:.4f}, pred={crit_pred}")
    else:
        print("CRITICAL TEST: Day A Ship image not found in test set.")

    out_csv = results_dir / 'champion_test_predictions.csv'
    report.to_csv(out_csv, index=False)

    print(f"Test metrics — Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    print(f"Saved test predictions: {out_csv}")


if __name__ == "__main__":
    # Debug: show which config is being used
    print(f"Using config module: {CFG_PATH}")
    print(f"Config values: IMAGE_SIZE={getattr(CFG,'IMAGE_SIZE', None)}, CHANNELS={getattr(CFG,'CHANNELS', None)}, EPOCHS_PER_RUN={getattr(CFG,'EPOCHS_PER_RUN', None)}")
    # Verify required train folders
    expected_classes = ['0_NOISE', '1_SHIP']
    missing = [c for c in expected_classes if not os.path.isdir(os.path.join(getattr(CFG, 'LABELED_DATA_DIR', 'labeled_data'), c))]
    if missing:
        raise FileNotFoundError(f"Missing class folders in '{getattr(CFG, 'LABELED_DATA_DIR', 'labeled_data')}': {missing}")
        
    try:
        # Create a timestamped run directory under runs/
        runs_root = Path('runs')
        runs_root.mkdir(exist_ok=True)
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_root = runs_root / run_id

        # Step 1: Run the search and find the best parameters and threshold
        best_params, best_threshold, test_ds, test_paths, test_labels = run_hyperparameter_search()
        
        # Step 2: Evaluate champion on the held-out test set
        evaluate_on_test(best_params, best_threshold, test_ds, test_paths, test_labels)

    except Exception as e:
        print(f"\nAn error occurred during the pipeline execution. Ensure you have the 'train' folder with '0_NOISE' and '1_SHIP' subfolders:")
        print(f"Error: {e}")