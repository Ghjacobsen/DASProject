# --- Project Configuration Settings ---

# Data Paths (Ensure these directories exist and contain your images)
LABELED_DATA_DIR = 'labeled_data'  # expects subfolders '0_NOISE' and '1_SHIP'
TEST_DIR = 'test'
MODEL_SAVE_PATH = 'best_ship_detector.h5'
PREDICTIONS_OUTPUT_FILE = 'unseen_data_predictions.csv'
RESULTS_DIR = 'results'  # central folder for champion model, temp best weights, test predictions

# Image Parameters
IMAGE_SIZE = (256, 256)
CHANNELS = 3  # PNGs are often loaded as 3-channel RGB
BATCH_SIZE = 32
RANDOM_SEED = 42
TEST_SIZE_FRACTION = 0.20   # 20% of remaining data used for test
VALIDATION_SIZE_FRACTION = 0.15 # 15% of remaining data used for validation
TEST_SAMPLES_PER_CLASS = 5  # legacy; not used in tactical split

# Training Parameters
EPOCHS_PER_RUN = 15 # Number of epochs for each hyperparameter combination
PATIENCE = 5        # Early stopping patience

# --- Hyperparameter Search Space ---
# The script will iterate through all combinations of these parameters.
HYPERPARAMETER_SPACE = {
    'learning_rate': [0.001, 0.0005],
    'dropout_rate': [0.2, 0.3],
    'num_conv_layers': [2, 3] # Number of blocks (Conv2D + MaxPool)
}
# Total runs: 2 * 2 * 2 = 8 runs   

# Data Pipeline Tweaks
# Set to True only if you consume full datasets each epoch (no steps_per_epoch truncation)
CACHE_DATASETS = False