import matplotlib.pyplot as plt
import numpy as np
import h5py
import logging
from scipy.ndimage import binary_opening
from pathlib import Path

def generate_reports(config):
    """
    Loads 'Residual' HDF5 files and their corresponding 'Raw' files.
    Generates side-by-side comparison plots.
    """
    logger = logging.getLogger(__name__)
    
    # Paths
    raw_path = Path(config['paths']['inference_data_path'])
    recon_path = Path(config['paths']['reconstruction_path'])
    fig_path = Path(config['paths']['figure_path'])
    fig_path.mkdir(parents=True, exist_ok=True)
    
    # Find all generated residual files
    res_files = sorted(list(recon_path.glob("residual_*.h5")) + list(recon_path.glob("residual_*.hdf5")))
    
    if not res_files:
        logger.warning(f"No residual files found in {recon_path}. Did evaluate.py run?")
        return

    thresh_sigma = config['visualization']['threshold_sigma']
    
    for res_file in res_files:
        # Deduce original filename: "residual_file1.h5" -> "file1.h5"
        orig_name = res_file.name.replace("residual_", "")
        orig_file = raw_path / orig_name
        
        if not orig_file.exists():
            logger.warning(f"Could not find original raw file for {res_file.name}")
            continue
            
        logger.info(f"Generating report for {orig_name}...")
        
        # Load Data
        with h5py.File(res_file, 'r') as f:
            residual_map = f['data'][:]
            # Load metadata for axes scaling
            dt = f['header/dt'][()]
            dx = f['header/dx'][()]
            
        with h5py.File(orig_file, 'r') as f:
            # Load raw data and applying basic Normalization for display 
            # (Just so it looks comparable to the Z-scored model output)
            raw_map = f['data'][:].astype(np.float32)
            raw_map = (raw_map - np.mean(raw_map)) / (np.std(raw_map) + 1e-6)

        # --- Processing for Visualization ---
        
        # 1. Thresholding
        mean_res = np.mean(residual_map)
        std_res = np.std(residual_map)
        threshold = mean_res + (thresh_sigma * std_res)
        
        # 2. Binary Mask
        binary_map = (residual_map > threshold).astype(int)
        
        # 3. Morphological Cleanup
        clean_map = binary_opening(binary_map, structure=np.ones((3,3)))
        
        # --- Plotting 2x2 Grid ---
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Calculate Extent for Axes (Time vs Distance)
        # extent = [x_min, x_max, y_max, y_min] (Origin usually top-left)
        max_dist = raw_map.shape[1] * dx / 1000.0 # km
        max_time = raw_map.shape[0] * dt # seconds
        extent = [0, max_dist, max_time, 0] 

        # A. Original
        im0 = axs[0, 0].imshow(raw_map, aspect='auto', cmap='ocean', extent=extent, vmin=-3, vmax=3)
        axs[0, 0].set_title(f"Original DAS Data ({orig_name})")
        axs[0, 0].set_ylabel("Time (s)")
        axs[0, 0].set_xlabel("Distance (km)")
        plt.colorbar(im0, ax=axs[0, 0], label="Amplitude (Z)")

        # B. Residual (The Heatmap)
        im1 = axs[0, 1].imshow(residual_map, aspect='auto', cmap='inferno', extent=extent)
        axs[0, 1].set_title("Reconstruction Residual (Error)")
        axs[0, 1].set_xlabel("Distance (km)")
        plt.colorbar(im1, ax=axs[0, 1], label="Error Magnitude")

        # C. Binary Threshold
        axs[1, 0].imshow(binary_map, aspect='auto', cmap='gray', extent=extent)
        axs[1, 0].set_title(f"Binary Mask (> {thresh_sigma}$\sigma$)")
        axs[1, 0].set_xlabel("Distance (km)")
        axs[1, 0].set_ylabel("Time (s)")

        # D. Final Clean Mask
        axs[1, 1].imshow(clean_map, aspect='auto', cmap='gray', extent=extent)
        axs[1, 1].set_title("Final Anomaly Detection (Cleaned)")
        axs[1, 1].set_xlabel("Distance (km)")

        plt.tight_layout()
        save_path = fig_path / f"report_{orig_name}.png"
        plt.savefig(save_path)
        plt.close()