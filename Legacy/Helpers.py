# Import required libraries
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta,timezone
import glob
import os
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import butter, filtfilt, firwin
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
# from matplotlib.colors import LinearSegmentedColomap
from matplotlib.ticker import AutoMinorLocator


#TAGER ALLE DE FUNKTIONER HASSE HAR LAVET.

def extract_metadata(file_path):
    """Extract metadata from HDF5 file"""
    with h5py.File(file_path, "r") as hdf5_file:
        start_time = hdf5_file["header/time"][()]
        dt = hdf5_file["header/dt"][()]
        dx = hdf5_file["header/dx"][()]
        channels = hdf5_file["header/channels"][()]
        # Get the actual number of samples in the file
        num_samples = hdf5_file["data"].shape[0]
    return start_time, dt, dx, channels, num_samples

def fir_bandpass(data, lowcut, highcut, fs, numtaps=101):
    """FIR bandpass filter with Nyquist frequency check"""
    nyquist = 0.5 * fs
    if highcut >= nyquist:
        highcut = nyquist * 0.99
    taps = firwin(numtaps, [lowcut/nyquist, highcut/nyquist], pass_zero=False)
    return filtfilt(taps, [1.0], data, axis=-1)


def find_start_file(file_paths, start_time_str):
    """Find the first file at or after the specified start time (HHMMSS format)"""
    # Convert input time string to datetime.time object
    try:
        input_time = datetime.strptime(start_time_str, "%H%M%S").time()
    except ValueError:
        raise ValueError("Invalid time format. Please use HHMMSS format (e.g., 102030)")

    # Find all files that match the time pattern
    time_files = []
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        if filename.endswith('.hdf5'):
            time_str = filename.split('.')[0]
            try:
                file_time = datetime.strptime(time_str, "%H%M%S").time()
                time_files.append((file_time, file_path))
            except ValueError:
                continue

    if not time_files:
        raise ValueError("No valid time-based HDF5 files found in the directory")

    # Sort files by time
    time_files.sort()

    # Find the first file at or after the requested time
    for file_time, file_path in time_files:
        if file_time >= input_time:
            return file_path, time_files.index((file_time, file_path))

    # If we get here, all files are before requested time - return last file
    return time_files[-1][1], len(time_files) - 1

def compute_spectrogram(signal, fs, Nfft=2048, overlap=0.9, file_time=None):
    """Compute spectrogram using matplotlib's specgram function"""
    Nover = int(np.floor(Nfft * overlap))
    pspect, fspect, tspect = plt.mlab.specgram(
        signal,
        NFFT=Nfft,
        Fs=fs,
        window=np.hanning(Nfft),
        noverlap=Nover,
        mode="psd",
    )
    psd = 10 * np.log10(pspect)
    if file_time:
        tspect_utc = [datetime.fromtimestamp(file_time) + timedelta(seconds=t) for t in tspect]
    else:
        tspect_utc = tspect
    return np.array(tspect_utc), fspect, psd

def load_and_join_data(input_directory, channel_step, start_time=None, end_time=None):
    """
    Load and concatenate data from HDF5 files within a specified time range.

    Parameters
    ----------
    input_directory : str
        Folder containing the HDF5 DAS files.
    channel_step : int
        Spatial downsampling factor.
    start_time, end_time : str, optional
        Start and end times in HHMMSS format. If None, load all available data.

    Returns
    -------
    data : np.ndarray
        Concatenated DAS data.
    time_axis : list
        List of datetime objects for the time axis.
    dt : float
        Sampling interval.
    dx : float
        Spatial sampling interval.
    channels_array : np.ndarray
        Array of channel indices.
    """
    file_paths = sorted(glob.glob(os.path.join(input_directory, '*.hdf5')))

    if not file_paths:
        raise FileNotFoundError(f"No HDF5 files found in {input_directory}")

    # Find the start and end files based on the time window
    start_file_path = file_paths[0]
    end_file_path = file_paths[-1]
    start_file_index = 0
    end_file_index = len(file_paths) - 1

    if start_time:
        try:
            start_file_path, start_file_index = find_start_file(file_paths, start_time)
        except ValueError as e:
            print(f"Warning: Could not find start file for time {start_time}. Loading from the first file. Error: {e}")

    if end_time:
         # For the end time, we need to find files up to and including the one that contains the end time.
         # We can reuse find_start_file but logic needs adjustment to find the last file *before* the end time + 1 file.
         # A simpler approach for now is to iterate from the start file and stop when we pass the end time.
         try:
            end_datetime = datetime.strptime(end_time, "%H%M%S").time()
            temp_end_index = start_file_index
            for i in range(start_file_index, len(file_paths)):
                 filename = os.path.basename(file_paths[i])
                 if filename.endswith('.hdf5'):
                    time_str = filename.split('.')[0]
                    try:
                         file_time = datetime.strptime(time_str, "%H%M%S").time()
                         if file_time <= end_datetime:
                             temp_end_index = i
                         else:
                             break # Stop when we pass the end time
                    except ValueError:
                         continue
            end_file_index = temp_end_index

         except ValueError as e:
             print(f"Warning: Could not parse end time {end_time}. Loading up to the last file. Error: {e}")


    # Ensure end_file_index is not before start_file_index
    end_file_index = max(start_file_index, end_file_index)


    selected_file_paths = file_paths[start_file_index : end_file_index + 1]

    if not selected_file_paths:
        raise ValueError("No files selected based on the provided time window.")

    # Extract metadata from the first selected file
    file_start_time_unix, dt, dx, channels_array, _ = extract_metadata(selected_file_paths[0])

    all_data = []
    time_axis = []
    total_samples = 0

    for file_path in selected_file_paths:
        try:
            current_start_time_unix, current_dt, _, current_channels_array, current_num_samples = extract_metadata(file_path)
            if current_dt != dt:
                print(f"Warning: dt mismatch in {file_path}. Expected {dt}, got {current_dt}. Skipping file.")
                continue
            if not np.array_equal(current_channels_array, channels_array):
                 print(f"Warning: Channel mismatch in {file_path}. Skipping file.")
                 continue


            with h5py.File(file_path, 'r') as f:
                # Load all channels for the current file
                file_data = f['data'][:]

                # Downsample channels
                file_data_downsampled = file_data[:, ::channel_step]


                # Create time axis for this file
                file_start_datetime = datetime.utcfromtimestamp(current_start_time_unix)
                # Calculate times for each sample in the current file
                file_times_seconds = np.arange(current_num_samples) * current_dt
                file_times_datetime = [file_start_datetime + timedelta(seconds=float(t)) for t in file_times_seconds]

                all_data.append(file_data_downsampled)
                time_axis.extend(file_times_datetime)
                total_samples += file_data_downsampled.shape[0]

        except KeyError as e:
            print(f"Skipping file {file_path} due to error: {e}")
        except Exception as e:
            print(f"Skipping file {file_path} due to unexpected error: {e}")


    if not all_data:
        raise ValueError("No data could be loaded from the selected files.")

    # Concatenate data from all files
    data = np.concatenate(all_data, axis=0)


    return data, time_axis, dt, dx, channels_array[::channel_step]

print("Helpers.py loaded successfully.")