"""Execution script to run preprocessing, model training, and evaluation sequentially.
Usage:
    python run_all.py

Environment overrides (optional):
    CAE_BASE_DIR, CAE_TEST_SUBDIR, CAE_DMIN_KM, CAE_DMAX_KM,
    CAE_SAMPLE_STEP, CAE_CHANNEL_STEP, CAE_TIME_PATCH, CAE_DISTANCE_PATCH,
    CAE_PATCH_OVERLAP, CAE_ARTIFACT_DIR, CAE_BATCH_SIZE, CAE_EPOCHS,
    CAE_MODEL_PATH, CAE_THRESHOLD_PERCENTILE, CAE_TOP_K
"""

import importlib
import sys
import traceback
import time

STEPS = [
    ("Preprocessing", "Prepocessing", "main"),  # retains original filename spelling
    ("Model Training", "Model", "main"),
    ("Evaluation", "Evaluate", "main"),
]

def run_step(label, module_name, func_name):
    print(f"\n=== {label} ===")
    start = time.time()
    try:
        try:
            mod = importlib.import_module(module_name)
        except ModuleNotFoundError:
            # Fallback for common misspelling: attempt corrected name
            if module_name == "Prepocessing":
                print("[run_all] 'Prepocessing' not found; trying 'Preprocessing'.")
                mod = importlib.import_module("Preprocessing")
            else:
                raise
        func = getattr(mod, func_name)
        func()
    except Exception as e:
        print(f"Error during {label}: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        elapsed = time.time() - start
        print(f"=== {label} completed in {elapsed:.1f}s ===")

def main():
    for label, module_name, func_name in STEPS:
        run_step(label, module_name, func_name)
    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()