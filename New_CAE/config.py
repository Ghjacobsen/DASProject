# --- Project Configuration Settings ---

# Data Paths (Ensure these directories exist and contain your images)
TRAIN_DIR = 'train'
TEST_DIR = 'test'
UNSEEN_DIR = 'unseen_data'  # legacy; not used in current pipeline
MODEL_SAVE_PATH = 'best_ship_detector.h5'
PREDICTIONS_OUTPUT_FILE = 'unseen_data_predictions.csv'

# Image Parameters
IMAGE_SIZE = (256, 256)
CHANNELS = 3  # PNGs are often loaded as 3-channel RGB
BATCH_SIZE = 32
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.20 # 20% of 'train' data used for validation
TEST_SAMPLES_PER_CLASS = 5  # number of samples to move to test from each class

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