Supervised Ship Anomaly Detection (DAS Data)

Purpose
- Detect faint ship signatures in DAS time–distance images using a compact CNN classifier optimized for F1 (balanced Precision/Recall).

What’s Included
- Deterministic training (full seeding) and clear session between runs.
- Tuned decision threshold (max-F1 on validation) used for evaluation.
- Single global registry (`runs/registry.csv`) that logs the best metrics for every hyperparameter combination across searches.
- Champion model: best-performing model saved as `champion_model.h5` and evaluated on a small held-out test split (5 images/class) with results in `champion_test_predictions.csv`.

Setup
- Install dependencies (preferably in a virtual environment):

```cmd
pip install tensorflow pandas scikit-learn matplotlib
```

Directory Layout
```
New_CAE/
	train/
		0_NOISE/   # Labeled noise images (Class 0)
		1_SHIP/    # Labeled ship images (Class 1)
	test/              # Automatically created; contains held-out images
		0_NOISE/
		1_SHIP/
	runs/
		registry.csv  # Accumulated best-per-combination results across searches
	config.py
	data_loader.py
	train_evaluate.py
```

Run
```cmd
python train_evaluate.py
```

Outputs
- `runs/registry.csv`: One row per hyperparameter combination in each search, including:
	- `run_id`, `combo_id`, `learning_rate`, `dropout_rate`, `num_conv_layers`
	- `best_epoch`, `val_f1_at_0.5`, `val_f1_tuned`, `val_precision_tuned`, `val_recall_tuned`, `best_threshold`, `roc_auc`, `pr_auc`
- `champion_model.h5`: Best overall model from the latest search.
- `champion_test_predictions.csv`: Per-image probability and label for the held-out test set using the champion’s tuned threshold.

Pipeline Details
- Data loading: Uses Keras directory datasets with 80/20 split and no shuffling for evaluation. A small test holdout is created by moving 5 samples per class from `train/` into `test/` (idempotent).
- Preprocessing: Images scaled to [0, 1].
- Augmentation (training only): Mild `RandomZoom`, `RandomTranslation`, and `RandomContrast`. No flips (time inversion) and no large rotations to preserve diagonals.
- Model: Compact CNN with a slightly wider first layer and 5x5 kernel to better capture diagonal patterns.
- Metrics: Tracks Accuracy, Precision, Recall, and F1 during training (0.5 threshold). After training, tunes the decision threshold on validation to maximize F1, and reports ROC-AUC and PR-AUC.
- Champion selection: Chooses the hyperparameter combination with the highest tuned validation F1; saves `champion_model.h5` and uses it for the final unseen scoring.

Notes
- If you don’t have `tabulate` installed, the console prints will fall back to plain text.
- Ensure `train/0_NOISE` and `train/1_SHIP` exist before running. The script creates `unseen_data/dummy` if it’s missing.

📒 Run Registry & Logs

- A timestamped run folder is created under `runs/` for each full pipeline execution.
- Example: `runs/20251126_153045/`
- Contents:
	- `config_snapshot.json`: Captures image size, epochs, patience, and search space used.
	- `summary.csv`: Grid-search results across all hyperparameter combinations.
	- `best_model.h5` and `best_params.json`: The best-performing model (also saved to `best_ship_detector.h5` at project root) and its parameters.
	- `unseen_data_predictions.csv`: Copy of the final predictions report for reproducibility.
	- Per-combination subfolders like `run_01_lr0.001_do0.2_layers2/` containing:
		- `history.csv`: Per-epoch training/validation metrics (via Keras CSVLogger).
		- `best_weights.h5`: Best weights for that specific combination (monitored by validation F1).

This lightweight registry preserves all runs, metrics, and artifacts to make comparisons and auditing easy, while keeping the original outputs for quick access.