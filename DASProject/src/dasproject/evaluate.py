import torch
import h5py
import numpy as np
import time
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from src.dasproject.data import DASDataset
from src.dasproject.utils import get_device

def stitch_patches(patches, original_shape, patch_size):
    """
    Reconstructs the full 2D array from a list of patches.
    """
    full_img = np.zeros(original_shape, dtype=np.float32)
    p_time, p_chan = patch_size
    time_dim, channel_dim = original_shape
    
    idx = 0
    for t in range(0, time_dim - p_time + 1, p_time):
        for c in range(0, channel_dim - p_chan + 1, p_chan):
            if idx < len(patches):
                full_img[t:t+p_time, c:c+p_chan] = patches[idx][0]
                idx += 1
    return full_img

def run_inference_and_save(model, config, input_dir, output_dir):
    """
    Runs inference one file at a time from input_dir, 
    saves the 'Residual Map' to output_dir as HDF5.
    
    Args:
        model: Loaded PyTorch model
        config: Config dictionary
        input_dir (str or Path): Folder containing source .h5/.hdf5 files
        output_dir (str or Path): Folder to save residual .h5 files
    """
    logger = logging.getLogger(__name__)
    device = get_device()
    model.to(device)
    model.eval()

    # Ensure paths are Path objects
    input_path = Path(input_dir)
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Find files
    inf_files = sorted(list(input_path.glob("*.h5")) + list(input_path.glob("*.hdf5")))
    if not inf_files:
        raise FileNotFoundError(f"No inference files found in {input_path}")

    logger.info(f"Starting inference on {len(inf_files)} files.")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {save_path}")
    
    patch_size = tuple(config['data']['patch_size'])
    
    # Statistics containers
    total_processing_times = []
    pure_compute_times = []
    
    for file_path in inf_files:
        logger.info(f"Processing {file_path.name}...")
        
        # --- TIMER START (Total File) ---
        t_file_start = time.perf_counter()
        
        # 1. Load Dataset
        # Note: We pass a list of just ONE file to handle them individually
        dataset = DASDataset([file_path], config, mode='inference')
        
        if len(dataset) == 0:
            logger.warning(f"Skipping {file_path.name} (Dataset empty)")
            continue

        dataloader = DataLoader(dataset, batch_size=config['training']['inference_batch_size'], shuffle=False)
        
        file_residuals = []
        t_compute_accum = 0.0 # Accumulator for pure model time
        
        with torch.no_grad():
            for img, _ in dataloader:
                img = img.to(device)
                
                # --- TIMER START (Pure Compute) ---
                t_batch_start = time.perf_counter()
                
                recon = model(img)
                diff = torch.abs(img - recon) # Keep on GPU for a moment
                
                t_batch_end = time.perf_counter()
                t_compute_accum += (t_batch_end - t_batch_start)
                # --- TIMER END (Pure Compute) ---

                # Move to CPU for storage
                diff_np = diff.cpu().numpy()
                
                for i in range(diff_np.shape[0]):
                    file_residuals.append(diff_np[i])

        # 3. Stitching
        # Get Original Dimensions from file
        with h5py.File(file_path, 'r') as f:
             orig_shape = f['data'].shape 
             dt = f['header/dt'][()]
             dx = f['header/dx'][()]

        stitched_residual = stitch_patches(file_residuals, orig_shape, patch_size)
        
        # 4. Save to HDF5
        save_name = save_path / f"residual_{file_path.name}"
        with h5py.File(save_name, 'w') as f_out:
            f_out.create_dataset('data', data=stitched_residual, compression="gzip")
            grp = f_out.create_group('header')
            grp.create_dataset('dt', data=dt)
            grp.create_dataset('dx', data=dx)
            
        # --- TIMER END (Total File) ---
        t_file_end = time.perf_counter()
        
        total_time = t_file_end - t_file_start
        
        # Store stats
        total_processing_times.append(total_time)
        pure_compute_times.append(t_compute_accum)
        
        logger.info(f"-> Saved: {save_name.name}")
        logger.info(f"   Total Time: {total_time:.3f}s | Pure Compute: {t_compute_accum:.3f}s")

    # Final Summary
    avg_total = np.mean(total_processing_times)
    avg_compute = np.mean(pure_compute_times)
    
    logger.info("-" * 30)
    logger.info(f"Batch Processing Complete.")
    logger.info(f"Avg Total Time per File:  {avg_total:.3f}s")
    logger.info(f"Avg Compute Time per File: {avg_compute:.3f}s")
    logger.info("-" * 30)