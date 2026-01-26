import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
VAL_DIR = Path("data/reconstructions/validation")
TEST_DIR = Path("data/reconstructions/anomalous")
OUTPUT_DIR = Path("reports/figures")

# SAMPLES TO TAKE (The "Big Data" Fix)
# We need enough samples to see the 10^-6 frequencies
TARGET_SAMPLES = 1_000_000

# VISUALIZATION SETTINGS
X_LIMIT = 10.0       # Hardcap x-axis at 4.0
USE_LOG_SCALE = True

# FONT SIZES
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18

def get_massive_samples(folder_path, label, n_samples):
    """
    Reads a massive amount of data using efficient slicing.
    Strategies:
    - Validation: Read the WHOLE quiet zone (it's small).
    - Test: Read every 10th time step (it's huge).
    """
    files = sorted(list(folder_path.glob("*.h5")) + list(folder_path.glob("*.hdf5")))
    
    if not files:
        print(f"⚠️ Warning: No files found in {folder_path}")
        return np.array([])

    print(f"[{label}] Scanning {len(files)} files for ~{n_samples} samples...")
    
    pool = []
    
    # Validation Window (Quiet Zone 3-4km)
    VAL_START_KM = 3.0
    VAL_END_KM = 4.0
    
    # Smart Stride Calculation
    # We want to read enough files to hit 1 million samples without reading 50GB.
    # Stride = 10 means "read every 10th row"
    DISK_STRIDE = 10 
    
    for f_path in files:
        try:
            with h5py.File(f_path, 'r') as f:
                # 1. Dimensions
                if 'header/dx' in f:
                    dx = f['header/dx'][()]
                else:
                    dx = 1.02
                
                # 2. Slice Logic
                if "Validation" in label:
                    # Spatial Slice: 3km - 4km
                    col_start = int((VAL_START_KM * 1000) / dx)
                    col_end = int((VAL_END_KM * 1000) / dx)
                    
                    # Safety check
                    _, max_cols = f['data'].shape
                    col_end = min(col_end, max_cols)
                    
                    # Read STRIDED rows, specific columns
                    # This is efficient in HDF5
                    data_chunk = f['data'][::DISK_STRIDE, col_start:col_end]
                    
                else:
                    # Test Data: Full Cable
                    # Read STRIDED rows, all columns
                    data_chunk = f['data'][::DISK_STRIDE, :]

                # Flatten and add to pool
                pool.append(np.abs(data_chunk.flatten()))
                
                # Early stopping check (optional, but good for speed)
                current_count = sum(len(x) for x in pool)
                if current_count > n_samples * 1.5:
                    break
                
        except Exception as e:
            print(f"Error reading {f_path.name}: {e}")

    if not pool:
        return np.array([])

    full_pool = np.concatenate(pool)
    
    # --- EXACT DOWNSAMPLING ---
    # If we collected 2 million, cut it down to exactly 1 million
    if len(full_pool) >= n_samples:
        np.random.seed(42)
        final_samples = np.random.choice(full_pool, n_samples, replace=False)
    else:
        print(f"⚠️ Warning: Only found {len(full_pool)} samples (Target: {n_samples})")
        final_samples = full_pool
        
    return final_samples

def plot_final_histogram(val_errors, test_errors):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating Final Plot...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # BINS: Fine-grained bins for detail
    N_BINS = 100
    plot_range = (0, X_LIMIT) 

    # 1. Plot Ocean Background (Blue)
    # density=True is CRITICAL for 10^-6 scale
    ax.hist(val_errors, bins=N_BINS, range=plot_range, alpha=0.85, label='Ocean Background',
             color='#1f77b4', density=True, log=USE_LOG_SCALE)

    # 2. Plot Test Data (Red)
    ax.hist(test_errors, bins=N_BINS, range=plot_range, alpha=0.6, label='Test Data',
             color='#d62728', density=True, log=USE_LOG_SCALE)

    # --- STYLING ---
    ax.set_xlabel("Reconstruction Error", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Frequency (Density)", fontsize=LABEL_FONTSIZE)
    
    ax.set_xlim(0, X_LIMIT)
    
    # Big Ticks
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE, length=8, width=2)

    # Legend
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=True, fancybox=True, framealpha=0.9)
    
    # Cleanup
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False) 
    
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "final_paper_histogram_log.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Histogram saved to {save_path}")

if __name__ == "__main__":
    val_data = get_massive_samples(VAL_DIR, "Validation", TARGET_SAMPLES)
    test_data = get_massive_samples(TEST_DIR, "Test", TARGET_SAMPLES)
    
    if len(val_data) > 0 and len(test_data) > 0:
        plot_final_histogram(val_data, test_data)
    else:
        print("❌ Error: Could not load data.")