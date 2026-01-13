import torch
import h5py
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from src.dasproject.data import DASDataset

def stitch_patches(patches, original_shape, patch_size):
    """
    Reconstructs the full 2D array from a list of patches.
    Assumes patches were created in a row-major order (Time then Channel)
    without overlap, as defined in DASDataset.
    """
    full_img = np.zeros(original_shape, dtype=np.float32)
    p_time, p_chan = patch_size
    time_dim, channel_dim = original_shape
    
    # Calculate exactly how many steps fit (matching DASDataset logic)
    # We use these bounds to avoid index errors if the file isn't perfectly divisible
    idx = 0
    for t in range(0, time_dim - p_time + 1, p_time):
        for c in range(0, channel_dim - p_chan + 1, p_chan):
            if idx < len(patches):
                # patches[idx] is (1, H, W), we need (H, W)
                full_img[t:t+p_time, c:c+p_chan] = patches[idx][0]
                idx += 1
    return full_img

def run_inference_and_save(model, config):
    """
    Runs inference one file at a time, stitches the result, 
    and saves the 'Residual Map' to disk as a new HDF5 file.
    """
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Paths
    inf_path = Path(config['paths']['inference_data_path'])
    recon_path = Path(config['paths']['reconstruction_path'])
    recon_path.mkdir(parents=True, exist_ok=True)
    
    inf_files = sorted(list(inf_path.glob("*.h5")) + list(inf_path.glob("*.hdf5")))
    
    if not inf_files:
        raise FileNotFoundError(f"No inference files found in {inf_path}")

    logger.info(f"Starting inference on {len(inf_files)} files. Output dir: {recon_path}")
    
    patch_size = tuple(config['data']['patch_size'])
    
    for file_path in inf_files:
        logger.info(f"Processing {file_path.name}...")
        
        # 1. Load ONE file via Dataset (mode='inference' does full scan)
        dataset = DASDataset([file_path], config, mode='inference')
        
        # If file was too small or skipped
        if len(dataset) == 0:
            continue

        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # 2. Collect Patches
        file_residuals = []
        
        with torch.no_grad():
            for img, _ in dataloader:
                img = img.to(device)
                recon = model(img)
                
                # Calculate Absolute Error (Residual) per patch
                # | Input - Reconstruction |
                diff = torch.abs(img - recon).cpu().numpy()
                
                # Store patches
                for i in range(diff.shape[0]):
                    file_residuals.append(diff[i])
        
        # 3. Get Original Dimensions from file to initialize stitcher
        with h5py.File(file_path, 'r') as f:
             # We use the shape of the data we actually loaded
             # Note: If you have different precision in file vs RAM, shape is same
             orig_shape = f['data'].shape 
             
             # Also grab metadata to copy to new file (Optional but good)
             dt = f['header/dt'][()]
             dx = f['header/dx'][()]

        # 4. Stitch
        stitched_residual = stitch_patches(file_residuals, orig_shape, patch_size)
        
        # 5. Save to HDF5
        save_name = recon_path / f"residual_{file_path.name}"
        with h5py.File(save_name, 'w') as f_out:
            f_out.create_dataset('data', data=stitched_residual, compression="gzip")
            # Save metadata so we can plot axes correctly later
            grp = f_out.create_group('header')
            grp.create_dataset('dt', data=dt)
            grp.create_dataset('dx', data=dx)
            
        logger.info(f"Saved residual map to {save_name}")
        
    logger.info("Inference & Stitching Complete.")