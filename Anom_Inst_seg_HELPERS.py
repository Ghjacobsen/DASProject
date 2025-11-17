"""
DAS Anomaly Detection Helpers — Storebælt project
-------------------------------------------------
This helper file follows the same loading philosophy as the original
das_loader notebook: safe HDF5 reading, correct metadata extraction,
controlled spatial/temporal downsampling, and optional FIR bandpass
filtering.  It then provides utilities for unsupervised anomaly instance
detection with a 2-D convolutional autoencoder.
"""

import os, glob, h5py, json
from datetime import datetime, timedelta, timezone
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from scipy.signal import firwin, filtfilt, decimate
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
#  Basic utilities
# ---------------------------------------------------------------------

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
#  File / metadata utilities
# ---------------------------------------------------------------------

def extract_metadata(file_path):
    """Extract basic metadata from HDF5 file header."""
    with h5py.File(file_path, "r") as f:
        start_time = f["header/time"][()]
        dt = f["header/dt"][()]
        dx = f["header/dx"][()]
        channels = f["header/channels"][()]
        num_samples = f["data"].shape[0]
    return start_time, dt, dx, channels, num_samples


def list_files(data_dir, pattern="*.hdf5", max_files=None):
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {data_dir}")
    return files


# ---------------------------------------------------------------------
#  Preprocessing: band-pass, downsampling, clipping, normalization
# ---------------------------------------------------------------------

def fir_bandpass(data, lowcut, highcut, fs, numtaps=101):
    nyquist = 0.5 * fs
    if highcut >= nyquist:
        highcut = nyquist * 0.99
    taps = firwin(numtaps, [lowcut/nyquist, highcut/nyquist], pass_zero=False)
    return filtfilt(taps, [1.0], data, axis=-1)


def preprocess_snippet(x,
                       fs_time_hz,
                       meters_per_channel,
                       bandpass=None,
                       target_fs_time_hz=None,
                       target_fs_space_m=None,
                       clip_sigma=6.0):
    """
    Apply band-pass filtering and spatial/temporal downsampling.
    Input x: [time, distance] or [distance, time]; function auto-orients.
    """
    if x.shape[0] < x.shape[1]:
        x = x.T  # ensure [distance, time]

    # Band-pass filter (optional)
    if bandpass:
        low, high = bandpass
        x = fir_bandpass(x, low, high, fs_time_hz)

    # Temporal decimation
    if target_fs_time_hz and target_fs_time_hz < fs_time_hz:
        q_t = max(1, int(round(fs_time_hz / target_fs_time_hz)))
        x = decimate(x, q_t, axis=1, zero_phase=True)

    # Spatial decimation
    if target_fs_space_m and target_fs_space_m > meters_per_channel:
        q_s = max(1, int(round(target_fs_space_m / meters_per_channel)))
        x = decimate(x, q_s, axis=0, zero_phase=True)

    # Clip outliers and normalize
    x = np.nan_to_num(x)
    mu, sigma = np.mean(x), np.std(x)
    x = np.clip(x, mu - clip_sigma*sigma, mu + clip_sigma*sigma)
    x = (x - mu) / (sigma + 1e-8)
    return x.astype(np.float32)


def load_hdf5_snippets(data_dir, process_kwargs, max_files=None):
    """
    Load all .hdf5 snippets, applying preprocess_snippet to each.
    """
    files = list_files(data_dir, "*.hdf5", max_files=max_files)
    X_list, meta = [], []
    for fp in tqdm(files, desc="Loading & preprocessing"):
        try:
            start, dt, dx, ch, ns = extract_metadata(fp)
            fs_time = 1.0 / dt
            meters_per_ch = float(dx)
            with h5py.File(fp, "r") as f:
                arr = f["data"][:]
            arr = preprocess_snippet(
                arr,
                fs_time_hz=fs_time,
                meters_per_channel=meters_per_ch,
                **process_kwargs
            )
            X_list.append(arr)
            meta.append(dict(path=fp, start=start, dt=dt, dx=dx, shape=arr.shape))
        except Exception as e:
            print(f"Warning: skipping {fp}: {e}")
    return X_list, meta


# ---------------------------------------------------------------------
#  Feature matrix + IsolationForest
# ---------------------------------------------------------------------

def snippet_features(x):
    f = [
        np.mean(x), np.std(x),
        np.median(np.abs(x)),
        np.max(x), np.min(x),
        np.percentile(x, 95), np.percentile(x, 5),
        np.mean(np.abs(np.diff(x, axis=0))),
        np.mean(np.abs(np.diff(x, axis=1))),
    ]
    return np.array(f, dtype=np.float32)


def compute_feature_matrix(X_list):
    return np.stack([snippet_features(x) for x in X_list])


def self_filter_isolation_forest(F, contamination=0.05, random_state=42):
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(F)
    preds = iso.predict(F)
    inlier_idx = np.where(preds == 1)[0]
    outlier_idx = np.where(preds == -1)[0]
    return inlier_idx, outlier_idx, iso


# ---------------------------------------------------------------------
#  Autoencoder model + training
# ---------------------------------------------------------------------

class ConvAE(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, base, 3, 2, 1),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(base, base*2, 3, 2, 1),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(base*2, base*4, 3, 2, 1),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base*4, base*2, 3, 2, 1, 1),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.ConvTranspose2d(base*2, base, 3, 2, 1, 1),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.ConvTranspose2d(base, 1, 3, 2, 1, 1),
        )

    def forward(self, x): return self.dec(self.enc(x))


class SnippetDataset(Dataset):
    def __init__(self, X_list, indices):
        self.X = [X_list[i] for i in indices]
        H = min(x.shape[0] for x in self.X)
        W = min(x.shape[1] for x in self.X)
        H = max(8, (H // 8) * 8)
        W = max(8, (W // 8) * 8)
        self.H, self.W = H, W

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        h, w = x.shape
        sh, sw = (h - self.H)//2, (w - self.W)//2
        x = x[sh:sh+self.H, sw:sw+self.W]
        return torch.from_numpy(x[None]).float()


def train_autoencoder(dataset, epochs, batch_size, lr, early_stop_patience,
                      val_split, device, seed, base_channels):
    set_seed(seed)
    N = len(dataset)
    n_val = max(1, int(N*val_split))
    n_train = N - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ConvAE(base=base_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, patience = float("inf"), 0
    hist = {"train": [], "val": []}

    for ep in range(epochs):
        model.train()
        tr = 0
        for xb in train_dl:
            xb = xb.to(device)
            yb = model(xb)
            loss = F.mse_loss(yb, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr += loss.item()*xb.size(0)
        tr /= len(train_dl.dataset)

        model.eval()
        va = 0
        with torch.no_grad():
            for xb in val_dl:
                xb = xb.to(device)
                va += F.mse_loss(model(xb), xb).item()*xb.size(0)
        va /= len(val_dl.dataset)

        hist["train"].append(tr); hist["val"].append(va)
        print(f"Epoch {ep+1:03d}: train {tr:.4f}, val {va:.4f}")

        if va < best_val - 1e-5:
            best_val, patience = va, 0
            best_state = model.state_dict()
        else:
            patience += 1
            if patience >= early_stop_patience:
                print("Early stopping."); break

    model.load_state_dict(best_state)
    return model, (dataset.H, dataset.W), hist


# ---------------------------------------------------------------------
#  Scoring + thresholding
# ---------------------------------------------------------------------

def evaluate_scores(model, X_list, in_shape, device):
    model.eval()
    H, W = in_shape
    scores, per_file = [], []
    with torch.no_grad():
        for i, x in enumerate(tqdm(X_list, desc="Scoring")):
            h, w = x.shape
            sh, sw = (h - H)//2, (w - W)//2
            x_crop = x[sh:sh+H, sw:sw+W]
            xb = torch.from_numpy(x_crop[None, None]).float().to(device)
            recon = model(xb).cpu().numpy()[0,0]
            err = (recon - x_crop)**2
            s = err.mean()
            scores.append(s)
            per_file.append(dict(idx=i, path=i, score=s))
    return np.array(scores), per_file


def choose_threshold(scores, percentile=95):
    thr = np.percentile(scores, percentile)
    return scores > thr, thr


def save_scores_csv(per_file, is_anom, path):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "path", "score", "is_anomaly"])
        for rec, flag in zip(per_file, is_anom):
            writer.writerow([rec["idx"], rec["path"], rec["score"], int(flag)])


# ---------------------------------------------------------------------
#  Visualization
# ---------------------------------------------------------------------

def show_topN(model, X_list, scores, in_shape, N=3, device="cpu"):
    idxs = np.argsort(scores)[-N:][::-1]
    H, W = in_shape
    model.eval()
    plt.figure(figsize=(12, 4*N))
    for i, idx in enumerate(idxs):
        x = X_list[idx]
        h, w = x.shape
        sh, sw = (h - H)//2, (w - W)//2
        x_crop = x[sh:sh+H, sw:sw+W]
        xb = torch.from_numpy(x_crop[None, None]).float().to(device)
        with torch.no_grad():
            recon = model(xb).cpu().numpy()[0,0]
        err = np.abs(recon - x_crop)
        vmax = np.percentile(np.abs(x_crop), 99)
        plt.subplot(N,3,3*i+1); plt.imshow(x_crop, aspect='auto', cmap='jet', vmin=-vmax, vmax=vmax); plt.title(f"Input {idx}")
        plt.subplot(N,3,3*i+2); plt.imshow(recon, aspect='auto', cmap='jet', vmin=-vmax, vmax=vmax); plt.title("Reconstruction")
        plt.subplot(N,3,3*i+3); plt.imshow(err, aspect='auto', cmap='inferno'); plt.title("Error")
    plt.tight_layout(); plt.show()




# =========================
# Patch-level scoring
# =========================
import torch.nn.functional as F
import numpy as np
import torch

def _to_kernel_and_stride(window_seconds=None, window_meters=None,
                          stride_seconds=None, stride_meters=None,
                          fs_eff=200.0, mpc_eff=5.0):
    """
    Convert physical window/stride to integer (H,W) kernel/stride in samples.
    H = distance (channels), W = time (samples)
    """
    kH = int(round((window_meters or 0) / mpc_eff)) if window_meters else 1
    kW = int(round((window_seconds or 0) * fs_eff))  if window_seconds else 1
    sH = int(round((stride_meters or 0) / mpc_eff)) if stride_meters else max(1, kH // 2)
    sW = int(round((stride_seconds or 0) * fs_eff))  if stride_seconds else max(1, kW // 2)
    kH = max(1, kH); kW = max(1, kW); sH = max(1, sH); sW = max(1, sW)
    return (kH, kW), (sH, sW)

def patch_score_map(error2d, kernel_hw, stride_hw):
    """
    Average error in sliding windows using 2D average pooling (no padding).
    error2d: np.ndarray [H,W] of per-pixel squared error.
    Returns: np.ndarray [nH, nW] heatmap of patch scores.
    """
    x = torch.from_numpy(error2d[None, None]).float()
    kH,kW = kernel_hw; sH,sW = stride_hw
    heat = F.avg_pool2d(x, kernel_size=(kH,kW), stride=(sH,sW), padding=0)
    return heat.squeeze(0).squeeze(0).numpy()

def evaluate_patch_scores(
    model, X_list, in_shape,
    window_seconds=None, window_meters=None,
    stride_seconds=None, stride_meters=None,
    fs_eff=200.0, mpc_eff=5.0, reduce="max", device=None
):
    """
    Compute a scalar score per instance from a patch-level heatmap.
    reduce: "max" (default), "p95", or "mean".
    Returns: scores [N], heatmaps (list of np.ndarray)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    H,W = in_shape
    k_hw, s_hw = _to_kernel_and_stride(window_seconds, window_meters,
                                       stride_seconds, stride_meters,
                                       fs_eff=fs_eff, mpc_eff=mpc_eff)
    scores, heatmaps = [], []
    model.eval()
    with torch.no_grad():
        for x in X_list:
            h,w = x.shape
            sh, sw = (h - H)//2, (w - W)//2
            x_crop = x[sh:sh+H, sw:sw+W]
            xb = torch.from_numpy(x_crop[None, None]).float().to(device)
            recon = model(xb).cpu().numpy()[0,0]
            err = (recon - x_crop)**2  # per-pixel squared error
            heat = patch_score_map(err, k_hw, s_hw)
            heatmaps.append(heat)

            if reduce == "max":
                s = float(np.max(heat))
            elif reduce == "p95":
                s = float(np.percentile(heat, 95))
            elif reduce == "mean":
                s = float(np.mean(heat))
            else:
                raise ValueError("reduce must be 'max', 'p95', or 'mean'")
            scores.append(s)
    return np.array(scores, dtype=np.float32), heatmaps

def show_patch_heatmap(heatmap, title="Patch score heatmap"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.imshow(heatmap, aspect='auto', cmap='inferno')
    plt.colorbar(label="avg MSE per patch")
    plt.title(title); plt.xlabel("time patches"); plt.ylabel("distance patches")
    plt.tight_layout(); plt.show()
