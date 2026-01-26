import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta, timezone
from pathlib import Path


# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent
RAW_DIR = SCRIPT_DIR / "../data/raw_old"
OUTPUT_DIR = SCRIPT_DIR / "../reports/figures"
# VISUALIZATION SETTINGS
TIME_DOWNSAMPLE = 10    
SPATIAL_DOWNSAMPLE = 4  

# FORMALIA - FONT SIZES 22
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans']
})

def get_file_metadata(path):
    with h5py.File(path, 'r') as f:
        t0 = f['header/time'][()]
        dt = f['header/dt'][()]
        dx = f['header/dx'][()]
        shape = f['data'].shape
    return t0, dt, dx, shape

def load_raw_only(raw_path):
    """Load and downsample raw HDF5 files, stitching along time."""
    raw_files = sorted(list(raw_path.glob("*.h5")) + list(raw_path.glob("*.hdf5")))
    if not raw_files:
        raise ValueError(f"No raw files found in {raw_path}")

    full_raw = []
    start_timestamp = None
    total_time_samples = 0
    current_dt = 0.0
    current_dx = 0.0

    print(f"Stitching {len(raw_files)} raw files...")

    for raw_file in raw_files:
        try:
            t0, dt, dx, shape = get_file_metadata(raw_file)

            if start_timestamp is None:
                start_timestamp = t0
                current_dt = dt
                current_dx = dx

            with h5py.File(raw_file, 'r') as f_raw:
                r_data = np.empty(shape, dtype=np.float32)
                f_raw['data'].read_direct(r_data)
                r_down = np.abs(r_data[::TIME_DOWNSAMPLE, ::SPATIAL_DOWNSAMPLE])
                full_raw.append(r_down)
                total_time_samples += r_down.shape[0]

        except Exception as e:
            print(f"Error reading {raw_file.name}: {e}")

    if not full_raw:
        raise ValueError("No data loaded!")

    big_raw = np.vstack(full_raw)

    start_datetime = datetime.fromtimestamp(start_timestamp, tz=timezone.utc)
    effective_dt = current_dt * TIME_DOWNSAMPLE
    duration_sec = total_time_samples * effective_dt
    end_datetime = start_datetime + timedelta(seconds=duration_sec)
    effective_dx = (current_dx * 4) * SPATIAL_DOWNSAMPLE

    return big_raw, start_datetime, end_datetime, effective_dx

def add_formal_colorbar(im, ax, label, ticks=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label, rotation=270, labelpad=30, fontsize=22)
    cbar.ax.tick_params(labelsize=22) 
    if ticks is not None: 
        cbar.set_ticks(ticks)
    return cbar

def plot_raw_only(raw, t_start, t_end, dx):
    """Plot only the raw data as a single figure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    max_dist_km = (raw.shape[1] * dx) / 1000.0
    t_start_num = mdates.date2num(t_start)
    t_end_num = mdates.date2num(t_end)
    extent = [0, max_dist_km, t_start_num, t_end_num]
    date_fmt = mdates.DateFormatter('%H:%M')

    vmax_raw = np.percentile(raw, 99.5)

    print(f"Plotting raw figure... (Distance: {max_dist_km:.1f} km)")

    fig, ax = plt.subplots(figsize=(16, 8))

    im = ax.imshow(raw, aspect='auto', cmap='jet', origin='lower', extent=extent, vmin=0, vmax=vmax_raw)
    add_formal_colorbar(im, ax, '|Strain Rate|')
    ax.yaxis_date()
    ax.yaxis.set_major_formatter(date_fmt)
    ax.set_ylabel("Time")
    ax.set_xlabel("Distance (km)")

    path = OUTPUT_DIR / "raw_only_gradient.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved raw plot to {path}")
    plt.close(fig)

if __name__ == "__main__":
    raw_arr, t0, t1, dx = load_raw_only(RAW_DIR)
    if len(raw_arr) > 0:
        plot_raw_only(raw_arr, t0, t1, dx)