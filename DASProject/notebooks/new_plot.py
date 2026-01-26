import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from skimage.morphology import closing, square


# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent
RAW_DIR = SCRIPT_DIR / "../data/raw/inference"
RECON_DIR = SCRIPT_DIR / "../data/reconstructions/anomalous"
OUTPUT_DIR = SCRIPT_DIR / "../reports/figures"
# VISUALIZATION SETTINGS
TIME_DOWNSAMPLE = 10    
SPATIAL_DOWNSAMPLE = 4  
THRESHOLD = 3.7041
MORPH_KERNEL = square(5)  # Morphological closing kernel to remove noise      

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

def load_and_stitch_optimized(raw_path, recon_path):
    raw_files = sorted(list(raw_path.glob("*.h5")) + list(raw_path.glob("*.hdf5")))
    recon_files = sorted(list(recon_path.glob("residual_*.h5")) + list(recon_path.glob("residual_*.hdf5")))
    
    full_raw = []
    full_res = []
    
    start_timestamp = None
    total_time_samples = 0
    current_dt = 0.0
    current_dx = 0.0

    print(f"Stitching {len(recon_files)} files...")
    
    for res_file in recon_files:
        orig_name = res_file.name.replace("residual_", "")
        raw_file = next((p for p in raw_files if p.name == orig_name), None)
        
        if not raw_file: continue
            
        try:
            t0, dt, dx, shape = get_file_metadata(raw_file)
            
            if start_timestamp is None:
                start_timestamp = t0
                current_dt = dt
                current_dx = dx
            
            with h5py.File(raw_file, 'r') as f_raw, h5py.File(res_file, 'r') as f_res:
                r_data = np.empty(shape, dtype=np.float32)
                f_raw['data'].read_direct(r_data)
                r_down = np.abs(r_data[::TIME_DOWNSAMPLE, ::SPATIAL_DOWNSAMPLE])
                full_raw.append(r_down)

                res_data = f_res['data'][:] 
                res_down = res_data[::TIME_DOWNSAMPLE, ::SPATIAL_DOWNSAMPLE]
                full_res.append(res_down)
                
                total_time_samples += r_down.shape[0]

        except Exception as e:
            print(f"Error reading {res_file.name}: {e}")

    if not full_raw:
        raise ValueError("No data loaded!")

    big_raw = np.vstack(full_raw)
    big_res = np.vstack(full_res)
    
    start_datetime = datetime.fromtimestamp(start_timestamp, tz=timezone.utc)
    effective_dt = current_dt * TIME_DOWNSAMPLE
    duration_sec = total_time_samples * effective_dt
    end_datetime = start_datetime + timedelta(seconds=duration_sec)
    effective_dx = (current_dx * 4) * SPATIAL_DOWNSAMPLE
    
    return big_raw, big_res, start_datetime, end_datetime, effective_dx

def add_formal_colorbar(im, ax, label, ticks=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label, rotation=270, labelpad=30, fontsize=22)
    cbar.ax.tick_params(labelsize=22) 
    if ticks is not None: 
        cbar.set_ticks(ticks)
    return cbar

def plot_split_figures(raw, res, t_start, t_end, dx):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    max_dist_km = (raw.shape[1] * dx) / 1000.0
    t_start_num = mdates.date2num(t_start)
    t_end_num = mdates.date2num(t_end)
    extent = [0, max_dist_km, t_start_num, t_end_num]
    date_fmt = mdates.DateFormatter('%H:%M')
    
    vmax_raw = np.percentile(raw, 99.5)
    vmax_res = np.percentile(res, 99.5)

    print(f"Plotting split figures... (Distance: {max_dist_km:.1f} km)")

    # --- FIGURE 1: RAW + RECONSTRUCTION ERROR (2 PLOTS) ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    # (a) RAW
    im1 = ax1.imshow(raw, aspect='auto', cmap='jet', origin='lower', extent=extent, vmin=0, vmax=vmax_raw)
    add_formal_colorbar(im1, ax1, '|Strain Rate|')
    ax1.yaxis_date()
    ax1.yaxis.set_major_formatter(date_fmt)
    ax1.set_ylabel("Time ")
    # Remove text(a) if you want clean plots, or keep it:
    # ax1.text(-0.08, 1.05, '(a)', transform=ax1.transAxes, fontsize=22, fontweight='bold', va='top')

    # (b) ERROR
    im2 = ax2.imshow(res, aspect='auto', cmap='inferno', origin='lower', extent=extent, vmin=0, vmax=vmax_res)
    add_formal_colorbar(im2, ax2, 'Reconstruction Error')
    ax2.yaxis_date()
    ax2.yaxis.set_major_formatter(date_fmt)
    ax2.set_ylabel("Time")
    ax2.set_xlabel("Distance (km)")

    path1 = OUTPUT_DIR / "final_analysis_part1_gradient.png"
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Part 1 to {path1}")
    plt.close(fig1)

    # --- FIGURE 2: BINARY MASK (1 PLOT) ---
    fig2, ax3 = plt.subplots(figsize=(16, 6))
    
    # Strict Thresholding
    binary_map = (res > THRESHOLD).astype(np.uint8)
    
    # Apply morphological closing to remove salt-and-pepper noise
    binary_map = closing(binary_map, MORPH_KERNEL)
    
    # IMPORTANT: Zero out the car zone (0-2km) if desired
    # Assuming dx approx 4m after downsampling, 2km is approx index 500
    # car_zone_idx = int(2000 / dx)
    # binary_map[:, :car_zone_idx] = 0

    im3 = ax3.imshow(binary_map, aspect='auto', cmap='gray', origin='lower', extent=extent, vmin=0, vmax=1)


    ax3.yaxis_date()
    ax3.yaxis.set_major_formatter(date_fmt)
    ax3.set_ylabel("Time")
    ax3.set_xlabel("Distance (km)")

    path2 = OUTPUT_DIR / "final_analysis_part2_mask.png"
    plt.savefig(path2, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Part 2 to {path2}")
    plt.close(fig2)

if __name__ == "__main__":
    raw_arr, res_arr, t0, t1, dx = load_and_stitch_optimized(RAW_DIR, RECON_DIR)
    if len(raw_arr) > 0:
        plot_split_figures(raw_arr, res_arr, t0, t1, dx)