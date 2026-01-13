import h5py
import numpy as np
import torch
import logging
import gc
from torch.utils.data import Dataset
from scipy.signal import butter, sosfiltfilt
from pathlib import Path

class DASDataset(Dataset):
    """
    PyTorch Dataset for Distributed Acoustic Sensing (DAS) data.
    """

    def __init__(self, file_paths, config, mode='train'):
        self.logger = logging.getLogger(__name__)
        self.file_paths = file_paths
        self.config = config
        self.mode = mode
        self.patch_size = tuple(config['data']['patch_size'])
        
        self.logger.info(f"Initializing DASDataset with {len(file_paths)} files in {mode} mode.")
        self.patches = self._load_and_process_data()

    def _bandpass_filter(self, data, fs):
        """
        Applies a Butterworth bandpass filter using Second Order Sections (SOS).
        SOS is much more stable than (b, a) filtering when using float32.
        """
        low = self.config['preprocessing']['bandpass_lowcut']
        high = self.config['preprocessing']['bandpass_highcut']
        order = 4
        
        # Nyquist Check
        nyquist = 0.5 * fs
        if high >= nyquist:
            high = nyquist - 1.0

        low = low / nyquist
        high = high / nyquist
        
        # Use 'sos' (Second Order Sections) for float32 stability
        sos = butter(order, [low, high], btype='band', output='sos')
        
        # Explicitly cast coefficients to float32 to encourage float32 output
        sos = sos.astype(np.float32)
        
        # sosfiltfilt returns a new array, we cast it immediately just in case
        return sosfiltfilt(sos, data, axis=0).astype(np.float32)

    def _remove_channel_mean(self, data):
        """
        Subtracts mean in-place to save memory.
        """
        # Calculate mean vector (small)
        mean_profile = np.mean(data, axis=0, dtype=np.float32)
        
        # IN-PLACE subtraction (saves 480MB copy)
        data -= mean_profile
        return data

    def _load_and_process_data(self):
        """
        Loads HDF5 files with strict memory management.
        """
        all_patches = []
        PATCHES_PER_FILE_TRAIN = 50 
        
        for fp in self.file_paths:
            try:
                # Explicit Garbage Collection before big alloc
                gc.collect()
                
                with h5py.File(fp, 'r') as f:
                    if 'data' not in f or 'header' not in f:
                        continue

                    # 1. Physics
                    dt = f['header/dt'][()] 
                    fs = 1.0 / dt
                    
                    # 2. Allocating Float32 Container
                    data_shape = f['data'].shape
                    # Allocates ~480MB for standard file
                    raw_data = np.empty(data_shape, dtype=np.float32)
                    
                    # Direct Read (Bypasses Float64 buffer)
                    f['data'].read_direct(raw_data)
                    
                    # 3. Preprocessing (Strict Float32)
                    if self.config['preprocessing']['enable_bandpass']:
                        # Note: Filter creates new array, old 'raw_data' is GC'd
                        raw_data = self._bandpass_filter(raw_data, fs=fs)
                    
                    if self.config['preprocessing']['enable_channel_mean_subtraction']:
                        # In-place modification
                        raw_data = self._remove_channel_mean(raw_data)
                        
                    # 4. Normalization (In-Place)
                    mean = np.mean(raw_data, dtype=np.float32)
                    std = np.std(raw_data, dtype=np.float32)
                    
                    if std > 0:
                        raw_data -= mean # In-place
                        raw_data /= std  # In-place

                    # 5. Patching
                    time_dim, channel_dim = raw_data.shape
                    p_time, p_chan = self.patch_size
                    
                    if self.mode == 'train':
                        # Random Sampling (Low Memory)
                        for _ in range(PATCHES_PER_FILE_TRAIN):
                            rand_t = np.random.randint(0, time_dim - p_time)
                            rand_c = np.random.randint(0, channel_dim - p_chan)
                            patch = raw_data[rand_t:rand_t+p_time, rand_c:rand_c+p_chan]
                            all_patches.append(np.expand_dims(patch, axis=0))
                            
                    else:
                        # Sequential Scan (Inference)
                        for t in range(0, time_dim - p_time + 1, p_time):
                            for c in range(0, channel_dim - p_chan + 1, p_chan):
                                patch = raw_data[t:t+p_time, c:c+p_chan]
                                all_patches.append(np.expand_dims(patch, axis=0))
                            
            except Exception as e:
                self.logger.error(f"Error processing file {fp}: {e}")

        # Final Cleanup
        del raw_data
        gc.collect()

        if len(all_patches) == 0:
            self.logger.error("No patches generated.")
            return np.array([], dtype=np.float32)

        self.logger.info(f"Loaded {len(all_patches)} patches.")
        return np.array(all_patches, dtype=np.float32)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.patches[idx])
        return x, x