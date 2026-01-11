import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import typer
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter

# Configuration Constants
WINDOW_SEC = 60              # Window duration in seconds
OVERLAP_SEC = 40             # Overlap duration in seconds (e.g., 20s overlap = 10s step)
SPATIAL_START_KM = 0.1         # Start of spatial range (clip data before this point)
SPATIAL_LIMIT_KM = 10        # Clip data beyond 10km to remove high-intensity noise [2]
DOWNSAMPLE_TIME = 10         # Decimate time to reduce image height (625Hz is too high for raw images)
DOWNSAMPLE_SPACE = 1         # Keep spatial resolution or decimate if needed
IMAGE_MODE = 'DATA'          # Modes for image saving 'DATA' for raw, 'VISUALIZATION' for colormap
BACKGROUND_WINDOW_SEC = 300  # Window for rolling median background subtraction
TILE_SIZE = 640              # Tile size in pixels (square tiles)

# Masking Configuration (for cable defects)
MASK_ENABLED = True          # Enable/disable spatial masking
MASK_START_KM = 1.0          # Start of masked region (km)
MASK_END_KM = 2.0            # End of masked region (km)

# Bandpass Filter Configuration (for ocean noise removal)
BANDPASS_ENABLED = True      # Enable/disable temporal bandpass filtering
BANDPASS_LOW = 1.0           # Low cutoff frequency (Hz) - removes ocean swell
BANDPASS_HIGH = 10.0         # High cutoff frequency (Hz) - removes high-freq noise
BANDPASS_ORDER = 4           # Filter order (3-4 for stability)

class MyDataset(Dataset):
    """
    DAS Dataset handler. 
    Currently optimized for preprocessing raw HDF5 files into tiled images.
    """
    def __init__(self, data_path: str | Path | None = None) -> None:
        if data_path is None:
            project_root = Path(__file__).resolve().parents[2]
            data_path = project_root / "data" / "raw"
        self.data_path = Path(data_path)
        self.files = list(self.data_path.glob("*.hdf5"))
        if not self.files:
            print(f"Warning: No .hdf5 files found in {self.data_path}")

    def __len__(self) -> int:
        """Return the number of raw files available."""
        return len(self.files)

    def __getitem__(self, index: int):
        """
        Placeholder: In a training pipeline, this would return a processed image/label tensor.
        For this script, we focus on the preprocess() method.
        """
        return self.files[index]

    def extract_metadata(self, file_path):
        """Extract metadata from HDF5 file (Helper function)."""
        with h5py.File(file_path, "r") as hdf5_file:
            start_time = hdf5_file["header/time"][()]
            dt = hdf5_file["header/dt"][()]
            dx = hdf5_file["header/dx"][()]
            channels = hdf5_file["header/channels"][()]
            num_samples = hdf5_file["data"].shape
        return start_time, dt, dx, channels, num_samples

    def preprocess(self, output_folder: str | Path) -> None:
        output_folder = Path(output_folder)
        """
        Preprocess the raw data and save it to the output folder.
        
        Steps:
        1. Load HDF5 (potentially multiple files for long windows).
        2. Spatial Clip (>60km).
        3. Background Subtraction (Median removal).
        4. Normalize (99th percentile).
        5. Slice into windows with overlap.
        6. Tile into squares for CV models.
        """
        output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting preprocessing of {len(self.files)} files...")
        
         # Sort files by timestamp (assumes filenames like HHMMSS.hdf5)
        sorted_files = sorted(self.files, key=lambda x: x.stem)
        
        # Calculate how many files we need per window (each file = 10 seconds)
        files_per_window = max(1, int(np.ceil(WINDOW_SEC / 10.0)))
        files_per_step = max(1, int(np.ceil((WINDOW_SEC - OVERLAP_SEC) / 10.0)))
        
        print(f"Window: {WINDOW_SEC}s ({files_per_window} files), Step: {WINDOW_SEC - OVERLAP_SEC}s ({files_per_step} files)")
        
        # Process files in sliding windows
        file_idx = 0
        window_idx = 0
        
        while file_idx + files_per_window <= len(sorted_files):
            try:
                # Determine chunk of files to load for this window
                chunk_files = sorted_files[file_idx:file_idx + files_per_window]
                
                all_data = []
                dt = None
                dx = None
                
                # Load and concatenate data from chunk files
                for file_path in chunk_files:
                    _, file_dt, file_dx, _, _ = self.extract_metadata(file_path)
                    if dt is None:
                        dt = file_dt
                        dx = file_dx
                        
                    with h5py.File(file_path, "r") as f:
                        data = f["data"][()]
                        all_data.append(data)
                
                # Combine data from all files in chunk
                if len(all_data) > 1:
                    data = np.vstack(all_data)
                else:
                    data = all_data[0]
                
                # Generate filename based on time range
                start_time = chunk_files[0].stem
                end_time = chunk_files[-1].stem
                if len(chunk_files) == 1:
                    base_name = start_time
                else:
                    base_name = f"{start_time}-{end_time}"
                
                print(f"\nWindow {window_idx}: {base_name} ({len(chunk_files)} files)")
                print(f"Combined data shape: {data.shape}")
                
                # 2. Spatial Range Clipping (both start and end)
                min_spatial_idx = int(SPATIAL_START_KM * 1000 / dx)
                max_spatial_idx = int(SPATIAL_LIMIT_KM * 1000 / dx)
                # Ensure indices are within bounds
                min_spatial_idx = max(0, min(min_spatial_idx, data.shape[1]))
                max_spatial_idx = max(min_spatial_idx, min(max_spatial_idx, data.shape[1]))
                data = data[:, min_spatial_idx:max_spatial_idx]
                print(f"  Spatial range: {SPATIAL_START_KM}-{SPATIAL_LIMIT_KM} km (indices {min_spatial_idx}-{max_spatial_idx})")

                # 2.5. Spatial Masking (Cable Defect Removal)
                if MASK_ENABLED:
                    # Adjust mask indices relative to the clipped spatial range
                    mask_start_idx = int((MASK_START_KM - SPATIAL_START_KM) * 1000 / dx)
                    mask_end_idx = int((MASK_END_KM - SPATIAL_START_KM) * 1000 / dx)
                    # Ensure indices are within bounds
                    mask_start_idx = max(0, min(mask_start_idx, data.shape[1]))
                    mask_end_idx = max(0, min(mask_end_idx, data.shape[1]))
                    if mask_start_idx < mask_end_idx:
                        data[:, mask_start_idx:mask_end_idx] = 0
                        print(f"  Masked spatial region: {MASK_START_KM}-{MASK_END_KM} km (indices {mask_start_idx}-{mask_end_idx})")

                # 2.6. Temporal Bandpass Filtering (Ocean Noise Removal)
                if BANDPASS_ENABLED:
                    fs = 1.0 / dt  # Sampling frequency
                    nyq = 0.5 * fs  # Nyquist frequency
                    low = BANDPASS_LOW / nyq
                    high = BANDPASS_HIGH / nyq
                    
                    # Design Butterworth bandpass filter
                    b, a = butter(BANDPASS_ORDER, [low, high], btype='band')
                    
                    # Apply filter to each channel (spatial location)
                    for ch in range(data.shape[1]):
                        data[:, ch] = filtfilt(b, a, data[:, ch])
                    
                    print(f"  Applied Butterworth bandpass filter: {BANDPASS_LOW}-{BANDPASS_HIGH} Hz (order {BANDPASS_ORDER})")

                # 3. Background Subtraction (The 'Pylon' Fix)
                data = data - np.median(data, axis=0)  # Initial median removal

                # 4. Normalization (Hybrid: Per-channel clipping + Global normalization)
                # Per-channel 99th percentile clipping (suppresses noise independently)
                v_max_per_channel = np.percentile(np.abs(data), 99, axis=0)
                data = np.clip(data, -v_max_per_channel, v_max_per_channel)
                
                # Global min-max normalization (creates smooth, consistent images for YOLO)
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)

                # 5. Downsampling
                data = data[::DOWNSAMPLE_TIME, ::DOWNSAMPLE_SPACE]
                
                n_time, n_space = data.shape
                
                # 6. For visualization mode, save the entire window as one image
                if IMAGE_MODE == 'VISUALIZATION':
                    # Save as RGB colormap with 4:3 aspect ratio
                    display_tile = np.abs(data)
                    mx = display_tile.max()
                    if mx > 0:
                        display_tile = display_tile / mx
                        
                        H, W = display_tile.shape
                        target_W = max(1, int(round((4/3) * H)))
                        if W != target_W:
                            x_old = np.arange(W)
                            x_new = np.linspace(0, W - 1, target_W)
                            display_tile = np.vstack([np.interp(x_new, x_old, row) for row in display_tile])
                        
                        save_name = output_folder / f"{base_name}.png"
                        plt.imsave(save_name, display_tile, cmap='ocean', vmin=0, vmax=1)
                
                elif IMAGE_MODE == 'DATA':

                    #Force correct downsampling
                    keep_space = (n_space // TILE_SIZE) * TILE_SIZE
                    if keep_space > 0 and keep_space != n_space:
                        data = data[:, :keep_space]
                        n_space = keep_space

                    # For DATA mode, tile the window
                    current_dt = dt * DOWNSAMPLE_TIME
                    window_samples = int(WINDOW_SEC / current_dt)
                    if window_samples <= 0:
                        window_samples = 1
                    if n_time < window_samples:
                        window_samples = n_time
                    
                    tile_size = TILE_SIZE
                    tile_idx = 0

                    for t_start in range(0, n_time, tile_size):
                        t_end = min(t_start + tile_size, n_time)
                        for s_start in range(0, n_space, tile_size):
                            s_end = min(s_start + tile_size, n_space)

                            tile = data[t_start:t_end, s_start:s_end]

                            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                                pad_t = tile_size - tile.shape[0]
                                pad_s = tile_size - tile.shape[1]
                                tile = np.pad(tile, ((0, pad_t), (0, pad_s)), mode='constant')

                            save_name = output_folder / f"{base_name}_tile{tile_idx}.png"
                            plt.imsave(save_name, tile, cmap='ocean', vmin=0, vmax=1)
                            tile_idx += 1

                # Advance by step size (not full window)
                file_idx += files_per_step
                window_idx += 1
                    
            except Exception as e:
                print(f"Error processing window {window_idx}: {e}")
                file_idx += files_per_step

def preprocess() -> None:

    print("Preprocessing data...")

    if IMAGE_MODE == 'DATA':
        print("Image Mode: DATA (raw images)")
    else:
        print("Image Mode: VISUALIZATION (colormap images)")

    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw"
    output_folder = project_root / "data" / "processed"
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)
    print(f"Processing complete. Images saved to {output_folder}")


if __name__ == "__main__":
    typer.run(preprocess)