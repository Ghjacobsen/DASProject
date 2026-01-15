import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import random
from datetime import datetime, timedelta

# --- CONFIGURATION ---
START_TIME = 150405
END_TIME   = 150505
# Note: Files are 10s apart, so we step by 10 seconds
FILE_STEP  = 10      

RAW_DIR = Path("../data/raw/inference")
RECON_DIR = Path("../data/reconstructions")
OUTPUT_DIR = Path("../reports/figures")

FS = 800.0          
TARGET_SAMPLES = 40 
THRESHOLD_SIGMA = 3
MORPH_KERNEL = square(5)
OUTLIER_REMOVE_PCT = 0.20 # Remove top 20% furthest points

# ZONES
CAR_ZONE_PX = 1500     
SHIP_START_PX = 3500   
SHIP_END_PX = 4500

# --- HELPER: TIME LOGIC ---
def generate_file_list(start_int, end_int):
    s_str = str(start_int).zfill(6)
    e_str = str(end_int).zfill(6)
    t_start = datetime.strptime(s_str, "%H%M%S")
    t_end = datetime.strptime(e_str, "%H%M%S")
    
    file_list = []
    curr = t_start
    while curr <= t_end:
        fname = curr.strftime("%H%M%S")
        if fname.startswith("0"): fname = fname[1:]
        file_list.append(fname)
        curr += timedelta(seconds=10)
    return file_list

def load_hardcoded_window(start, end):
    file_ids = generate_file_list(start, end)
    full_raw, full_err = [], []
    
    print(f"🧵 Stitching {len(file_ids)} files...")
    
    for fid in file_ids:
        r_path = RAW_DIR / f"{fid}.hdf5"
        if not r_path.exists(): r_path = RAW_DIR / f"{fid}.h5"
        e_path = RECON_DIR / f"residual_{fid}.hdf5"
        if not e_path.exists(): e_path = RECON_DIR / f"residual_{fid}.h5"
        
        if r_path.exists() and e_path.exists():
            try:
                with h5py.File(r_path, 'r') as f:
                    d = f['data'][:]
                    if d.shape[0] < d.shape[1]: d = d.T
                    full_raw.append(d)
                with h5py.File(e_path, 'r') as f:
                    d = f['data'][:]
                    if d.shape[0] < d.shape[1]: d = d.T
                    full_err.append(d)
            except: continue

    if not full_raw: return None, None
    return np.vstack(full_raw), np.vstack(full_err)

def run_final_pca_plot():
    # 1. LOAD DATA
    raw, err = load_hardcoded_window(START_TIME, END_TIME)
    if raw is None: return

    # 2. MASK GENERATION
    print("✨ Generating Mask...")
    mu, sigma = np.mean(err), np.std(err)
    clean_mask = closing((err > (mu + THRESHOLD_SIGMA * sigma)), MORPH_KERNEL)
    regions = regionprops(label(clean_mask))
    
    candidates = {"Car": [], "Ship": []}

    # 3. HARVEST FEATURES
    for props in regions:
        if props.area < 200: continue 
        y_c, x_c = props.centroid
        minr, minc, maxr, maxc = props.bbox
        
        category = None
        if x_c < CAR_ZONE_PX: category = "Car"
        elif SHIP_START_PX < x_c < SHIP_END_PX: category = "Ship"
        if not category: continue

        try:
            blob = raw[minr:maxr, minc:maxc]
            seg_len = min(blob.shape[0], 256)
            f, p = welch(blob, fs=FS, nperseg=seg_len, axis=0)
            psd = np.mean(p, axis=1)
            
            # Normalize & Resample
            psd_norm = (psd - np.min(psd)) / (np.max(psd) - np.min(psd))
            psd_res = np.interp(np.linspace(0, 1, 50), np.linspace(0, 1, len(psd_norm)), psd_norm)
            candidates[category].append(psd_res)
        except: continue

    # 4. RANDOM SAMPLING
    final_features, final_labels = [], []
    for cat in ["Car", "Ship"]:
        pool = candidates[cat]
        selected = random.sample(pool, min(len(pool), TARGET_SAMPLES))
        final_features.extend(selected)
        final_labels.extend([cat] * len(selected))

    # 5. OCEAN SAMPLING
    ocean_count, tries = 0, 0
    while ocean_count < TARGET_SAMPLES and tries < TARGET_SAMPLES*20:
        tries += 1
        ry = np.random.randint(0, raw.shape[0]-128)
        rx = np.random.randint(SHIP_END_PX, raw.shape[1]) 
        if np.sum(clean_mask[ry:ry+128, rx:rx+1]) == 0:
            chunk = raw[ry:ry+128, rx:rx+10]
            f, p = welch(chunk, fs=FS, nperseg=128, axis=0)
            psd = np.mean(p, axis=1)
            psd_norm = (psd - np.min(psd)) / (np.max(psd) - np.min(psd))
            psd_res = np.interp(np.linspace(0, 1, 50), np.linspace(0, 1, len(psd_norm)), psd_norm)
            final_features.append(psd_res)
            final_labels.append("Ocean")
            ocean_count += 1

    # 6. RUN PCA
    X = np.nan_to_num(np.array(final_features))
    y = np.array(final_labels)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 7. CLEAN PLOTTING (With 20% Outlier Removal)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set global font sizes for report quality
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"Car": "#d62728", "Ship": "#1f77b4", "Ocean": "grey"}
    markers = {"Car": "s", "Ship": "o", "Ocean": "x"}
    
    for cat in ["Car", "Ship", "Ocean"]:
        idx = (y == cat)
        points = X_pca[idx]
        if len(points) == 0: continue
        
        # --- OUTLIER REMOVAL LOGIC ---
        # 1. Calculate Centroid
        centroid = np.mean(points, axis=0)
        # 2. Calculate Distances
        distances = np.linalg.norm(points - centroid, axis=1)
        # 3. Determine Cutoff (Remove top 20%)
        cutoff = np.percentile(distances, 100 * (1 - OUTLIER_REMOVE_PCT))
        # 4. Filter
        mask_keep = distances <= cutoff
        clean_points = points[mask_keep]
        
        # Plot
        ax.scatter(clean_points[:, 0], clean_points[:, 1], c=colors[cat], marker=markers[cat], 
                   label=cat, s=150, alpha=0.9, edgecolors='k', linewidth=0.5)

    # Aesthetics
    ax.set_xlabel(f"PC1")
    ax.set_ylabel(f"PC2")
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend on TOP
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=14)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "pca_plot_clean_3.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Report-Ready PCA to {save_path}")
    plt.show()

if __name__ == "__main__":
    run_final_pca_plot()
    