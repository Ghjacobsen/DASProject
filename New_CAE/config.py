# --- Project Configuration Settings ---

# Data Paths (Ensure these directories exist and contain your images)
LABELED_DATA_DIR = 'labeled_data'  # expects subfolders '0_NOISE' and '1_SHIP'
TEST_DIR = 'test'
MODEL_SAVE_PATH = 'best_ship_detector.h5'
PREDICTIONS_OUTPUT_FILE = 'unseen_data_predictions.csv'
RESULTS_DIR = 'results'  # central folder for champion model, temp best weights, test predictions

# Image Parameters
IMAGE_SIZE = (512, 1024)
CHANNELS = 3  # RGB input for high-fidelity signals
BATCH_SIZE = 2
RANDOM_SEED = 42
TEST_SIZE_FRACTION = 0.20   # 20% of remaining data used for test
VALIDATION_SIZE_FRACTION = 0.15 # 15% of remaining data used for validation
TEST_SAMPLES_PER_CLASS = 5  # legacy; not used in tactical split

EPOCHS_PER_RUN = 10 # Limit epochs per run to reduce memory/time
PATIENCE = 5        # Early stopping patience

# --- Hyperparameter Search Space ---
# The script will iterate through all combinations of these parameters.
HYPERPARAMETER_SPACE = {
    'learning_rate': [0.01,0.003, 0.001, 0.0005],
    'dropout_rate': [0.15],
    'num_conv_layers': [2, 3] # Number of blocks (Conv2D + MaxPool)
}
# Total runs: 4 * 1 * 2 = 8 runs

# Data Pipeline Tweaks
# Set to True only if you consume full datasets each epoch (no steps_per_epoch truncation)
CACHE_DATASETS = False

# Training Tweaks
USE_FOCAL_LOSS = True
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
REDUCE_LR_ON_PLATEAU = False
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 3

# Data Tweaks
GRAYSCALE_INPUT = False  # use RGB channels
OVERSAMPLE_SHIPS_FACTOR = 1  # avoid extra memory on CPU