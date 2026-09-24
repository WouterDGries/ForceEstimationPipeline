"""Dataset preparation for the fingertip force estimator.

Reads the raw dataAcquisition session folders directly (frames.csv,
force.csv, meta.json, video.h5 - see dataAcquisition/dataWriter.py for the
on-disk layout this expects). For each session it low-pass filters
force.csv's Fz and resamples it onto every frame timestamp, hole-fills the
depth crops (they still contain real zero-value holes - checked directly,
see the __main__ block below), then indexes valid 20-frame windows per
config.yaml's model_training block. Produces normalized (4, T, 112, 112)
RGB+relative-depth clips with a normalized centre-frame force label.
Consumed by train.py via build_dataset().
"""

import json
import os

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import DataLoader, Dataset

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
REPO_ROOT = os.path.dirname(os.path.abspath(CONFIG_PATH))

# Kinetics-400 per-channel stats, RGB order. video.h5's "color" dataset is
# already RGB - dataAcquisition/cameraHandler.py converts BGR->RGB before
# writing it (see crop_color_and_depth) - so no channel-order conversion
# happens here, only these stats.
RGB_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
RGB_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def load_config(path=CONFIG_PATH):
    """Input: path to config.yaml. Returns it parsed as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def label_frame_offset(window_length):
    """Input: window length in frames. Returns the single, fixed local
    frame index (0-based, within the window) the force label is read from -
    the centre frame. Both dataset.py and any later inference code must
    call this rather than hardcode the offset, or every force estimate is
    shifted in time relative to its clip.
    """
    return window_length // 2


def _estimate_sample_rate_hz(host_ts_ns):
    """Input: sorted 1D array of monotonic host_ts_ns. Returns the
    median-based sample rate in Hz, robust to the occasional dropped sample.
    """
    dt_ns = np.diff(host_ts_ns.astype(np.float64))
    return 1e9 / np.median(dt_ns)


def _zero_phase_lowpass(values, sample_rate_hz, cutoff_hz, order):
    """Input: 1D force samples and their filter settings
    (config['model_training']['force_filter']). Applies a zero-phase
    Butterworth low-pass (sosfiltfilt - no delay is introduced) and returns
    the filtered array, same shape as `values`.
    """
    sos = butter(order, cutoff_hz, btype="low", fs=sample_rate_hz, output="sos")
    return sosfiltfilt(sos, values)


def _load_filtered_frame_force(session_dir, frame_ts_ns, force_filter_config):
    """Input: session directory, that session's frame host_ts_ns, and
    config['model_training']['force_filter']. Reads force.csv, zero-phase
    low-pass filters the full Fz trace, then resamples it onto every frame
    timestamp (numpy.interp). Returns a (N,) float32 array of filtered Fz
    in newtons, one value per frame - this is what both the window filters
    and the label are read from, never the raw per-sample Fz.
    """
    force_df = pd.read_csv(os.path.join(session_dir, "force.csv"))
    force_ts_ns = force_df["host_ts_ns"].to_numpy()
    sample_rate_hz = _estimate_sample_rate_hz(force_ts_ns)
    filtered_fz = _zero_phase_lowpass(force_df["fz_N"].to_numpy(), sample_rate_hz,
                                       force_filter_config["cutoff_hz"], force_filter_config["order"])
    return np.interp(frame_ts_ns, force_ts_ns, filtered_fz).astype(np.float32)


def _hole_fill_depth_frame(depth_raw_u16, depth_scale_m, depth_config):
    """Input: one raw (H, W) uint16 depth crop, the session's
    depth_scale_m, and config['model_training']['depth']. Fills zero-valued
    holes by inpainting from the nearest valid neighbours, then applies a
    light median filter to knock down flying pixels along the finger
    silhouette - both per-frame only, nothing here ever mixes in another
    frame. Returns (depth_filled_mm, valid_mask): a float32 (H, W) array in
    millimetres, and a bool mask True where the ORIGINAL reading was
    non-zero (used for the per-window depth reference, not fed to the
    network).
    """
    valid_mask = depth_raw_u16 > 0
    depth_mm = depth_raw_u16.astype(np.float32) * depth_scale_m * 1000.0

    if not np.any(valid_mask):
        return depth_mm, valid_mask
    if np.all(valid_mask):
        hole_free_mm = depth_mm
    else:
        hole_mask = (~valid_mask).astype(np.uint8) * 255
        source_mm = np.where(valid_mask, depth_mm, 0.0).astype(np.float32)
        hole_free_mm = cv2.inpaint(source_mm, hole_mask, depth_config["hole_fill_inpaint_radius_px"], cv2.INPAINT_TELEA)

    ksize = depth_config["hole_fill_median_ksize"]
    filled_mm = cv2.medianBlur(hole_free_mm, ksize) if ksize > 1 else hole_free_mm
    return filled_mm, valid_mask


class _SessionData:
    """Holds one session's fully-loaded, cleaned arrays in memory: color
    frames, hole-filled depth (mm) + original-validity mask, per-frame
    filtered force, and frame timestamps. video.h5 is opened once here and
    closed before __init__ returns - only plain numpy arrays are kept
    afterwards, so there's no open HDF5 handle to survive DataLoader worker
    forking (h5py file handles are not fork-safe; numpy arrays are).
    """

    def __init__(self, session_dir, model_config):
        self.name = os.path.basename(session_dir)
        frames_df = pd.read_csv(os.path.join(session_dir, "frames.csv"))
        with open(os.path.join(session_dir, "meta.json")) as f:
            meta = json.load(f)
        depth_scale_m = meta["camera"]["depth"]["depth_scale_m"]

        self.frame_ts_ns = frames_df["host_ts_ns"].to_numpy()
        self.frame_fz_n = _load_filtered_frame_force(session_dir, self.frame_ts_ns, model_config["force_filter"])

        with h5py.File(os.path.join(session_dir, "video.h5"), "r") as h5f:
            self.color = h5f["color"][:]     # (N, 112, 112, 3) uint8, RGB
            depth_raw = h5f["depth"][:]      # (N, 112, 112) uint16, raw sensor counts

        self.num_frames = depth_raw.shape[0]
        self.depth_filled_mm = np.empty(depth_raw.shape, dtype=np.float32)
        self.depth_valid = np.empty(depth_raw.shape, dtype=bool)
        for i in range(self.num_frames):
            self.depth_filled_mm[i], self.depth_valid[i] = _hole_fill_depth_frame(
                depth_raw[i], depth_scale_m, model_config["depth"])
        self.original_invalid_fraction = 1.0 - self.depth_valid.mean()

        self.gap_after = self._find_gap_after(model_config["window"])

    def _find_gap_after(self, window_config):
        """Returns a (num_frames-1,) bool array: True at index i if the
        timestamp gap between frame i and i+1 alone already exceeds
        max_span_s - used to bound the temporal-offset augmentation to a
        gap-free run of frames.
        """
        dt_s = np.diff(self.frame_ts_ns.astype(np.float64)) / 1e9
        return dt_s > window_config["max_span_s"]


def _build_window_index(sessions, window_config, stride, offset, split_name):
    """Input: list of _SessionData for one split, config['model_training']
    ['window'], the window-start stride to use, the label frame offset
    (from label_frame_offset()), and the split name (for the printed
    report). Slides a window of window_config['length'] frames across every
    session at the given stride and keeps only windows passing both hard
    filters:

      1. elapsed time from the first to the last frame <= max_span_s,
         computed from real timestamps - this also rejects any window that
         spans a tracking-failure/dropped-frame gap, since a dropped frame
         inflates the elapsed time for the same frame-index span (frames.csv
         only ever contains frames where tracking succeeded - see
         dataAcquisition/offlineExtract.py's save_only_valid_frames gate - so a
         tracking failure shows up as a timestamp gap, not a flag);
      2. |Fz(last) - Fz(first)| <= max_force_change_n, from the filtered
         per-frame force.

    There is no trial-boundary or synchronization-event marker anywhere in
    this data (checked directly against all 7 sessions), so those two
    exclusions from the spec are not implemented as separate rules - both
    would only ever fire on a timestamp gap or an abrupt force change,
    which the two rules above already catch.

    Prints candidate/discarded/kept counts. Returns a list of
    (session_index, start_frame) tuples.
    """
    window_length = window_config["length"]
    max_span_s = window_config["max_span_s"]
    max_force_change_n = window_config["max_force_change_n"]

    kept, discarded_span, discarded_force, num_candidates = [], 0, 0, 0
    for session_index, session in enumerate(sessions):
        last_start = session.num_frames - window_length
        for start in range(0, max(last_start + 1, 0), stride):
            end = start + window_length - 1
            num_candidates += 1
            span_s = (session.frame_ts_ns[end] - session.frame_ts_ns[start]) / 1e9
            if span_s > max_span_s:
                discarded_span += 1
                continue
            force_change_n = abs(session.frame_fz_n[end] - session.frame_fz_n[start])
            if force_change_n > max_force_change_n:
                discarded_force += 1
                continue
            kept.append((session_index, start))

    print(f"  [dataset] {split_name}: {num_candidates} candidate windows, "
          f"{discarded_span} discarded (span > {max_span_s}s), "
          f"{discarded_force} discarded (|dFz| > {max_force_change_n}N), "
          f"{len(kept)} kept")
    return kept


def _largest_inscribed_crop_size(side, angle_deg):
    """Input: the (square) crop's side length in pixels and a rotation
    angle in degrees. Returns the side length of the largest axis-aligned
    square that fits inside the rotated crop with no empty corners
    (simplifies to side / (cos + sin) for a square input).
    """
    angle_rad = np.deg2rad(abs(angle_deg))
    denom = np.cos(angle_rad) + np.sin(angle_rad)
    return int(max(min(side / denom, side), 1))


def _rotate_and_recrop(color, depth_channel, angle_deg):
    """Input: (T,H,W,3) color in [0,1] and (T,H,W) normalized depth_channel,
    and one rotation angle in degrees shared by every frame in the window
    (the augmentation rule: one random draw per window, not per frame).
    Rotates every frame about the image centre, centre-crops to the largest
    rectangle with no empty border, and resizes back to (H, W). Color uses
    linear interpolation throughout; depth uses nearest only, never cubic -
    cubic overshoots at the sharp finger/skin depth step and invents values
    that don't exist. Returns (color, depth_channel), same shapes as input.
    """
    num_frames, height, width = depth_channel.shape
    assert height == width, "rotation crop-size formula assumes a square crop"
    crop_size = _largest_inscribed_crop_size(height, angle_deg)
    y0, x0 = (height - crop_size) // 2, (width - crop_size) // 2

    rot_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1.0)
    rotated_color = np.empty_like(color)
    rotated_depth = np.empty_like(depth_channel)
    for t in range(num_frames):
        rotated_color[t] = cv2.warpAffine(color[t], rot_matrix, (width, height), flags=cv2.INTER_LINEAR)
        rotated_depth[t] = cv2.warpAffine(depth_channel[t], rot_matrix, (width, height), flags=cv2.INTER_NEAREST)

    cropped_color = rotated_color[:, y0:y0 + crop_size, x0:x0 + crop_size, :]
    cropped_depth = rotated_depth[:, y0:y0 + crop_size, x0:x0 + crop_size]

    resized_color = np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                               for frame in cropped_color])
    resized_depth = np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
                               for frame in cropped_depth])
    return resized_color, resized_depth


class ForceClipDataset(Dataset):
    """torch Dataset over 20-frame RGB+relative-depth clips. `split` is
    "train", "val" or "test" - only "train" uses the training window
    stride and applies augmentation; the others are passed through
    untouched at eval_stride. See the module docstring for the per-window
    processing this builds on.
    """

    def __init__(self, model_config, split):
        window_config = model_config["window"]
        self.split = split
        self.window_length = window_config["length"]
        self.label_offset = label_frame_offset(self.window_length)
        self.max_span_s = window_config["max_span_s"]
        self.max_force_change_n = window_config["max_force_change_n"]
        self.clip_mm = model_config["depth"]["clip_mm"]
        self.force_norm_scale_n = model_config["label"]["force_norm_scale_n"]
        self.augmentation_config = model_config["augmentation"]

        session_names = model_config["splits"][f"{split}_sessions"]
        data_root = os.path.join(REPO_ROOT, model_config["data_root"])
        print(f"[dataset] Loading {split} split: {len(session_names)} session(s)")
        self.sessions = []
        for name in session_names:
            session = _SessionData(os.path.join(data_root, name), model_config)
            print(f"  [dataset] {name}: {session.num_frames} frames, "
                  f"{session.original_invalid_fraction * 100:.1f}% originally-invalid depth pixels (hole-filled)")
            self.sessions.append(session)

        stride = window_config["train_stride"] if split == "train" else window_config["eval_stride"]
        self.windows = _build_window_index(self.sessions, window_config, stride, self.label_offset, split)

    def __len__(self):
        return len(self.windows)

    def mean_normalized_label(self):
        """Returns the mean normalized label across every window in this
        dataset - used to initialize the regression head's output bias
        (see model.py) so training starts out predicting the average force.
        """
        labels = [self.sessions[session_index].frame_fz_n[start + self.label_offset] / self.force_norm_scale_n
                  for session_index, start in self.windows]
        return float(np.mean(labels))

    def _augmented_start(self, session, original_start):
        """Input: session and the window's originally-indexed start frame.
        Temporal-offset augmentation (Section 5): tries shifts in a
        shuffled order, keeping the first that stays in bounds, inside one
        gap-free run, and still passes both hard filters (a shift can push
        a window across a boundary the unshifted one didn't cross). Falls
        back to the original start if none do. Returns the (possibly
        shifted) start frame.
        """
        max_shift = self.augmentation_config["temporal_offset_max_frames"]
        if max_shift <= 0:
            return original_start
        shifts = list(range(-max_shift, max_shift + 1))
        np.random.shuffle(shifts)
        for shift in shifts:
            start = original_start + shift
            end = start + self.window_length - 1
            if start < 0 or end >= session.num_frames:
                continue
            if start < end and session.gap_after[start:end].any():
                continue
            span_s = (session.frame_ts_ns[end] - session.frame_ts_ns[start]) / 1e9
            force_change_n = abs(session.frame_fz_n[end] - session.frame_fz_n[start])
            if span_s <= self.max_span_s and force_change_n <= self.max_force_change_n:
                return start
        return original_start

    def _reference_depth_mm(self, session, label_index):
        """Input: session and the label frame's index. Returns the scalar
        per-window depth reference (Section 5): the median of the
        hole-filled depth at the pixels that were ORIGINALLY valid at the
        label frame, in millimetres. A per-frame reference would erase the
        signal (the finger descending, the skin indenting); this is fixed
        once per window, from the label frame only. Falls back to the
        median of the whole hole-filled label frame if every pixel there
        was originally invalid (rare).
        """
        label_valid = session.depth_valid[label_index]
        label_depth_mm = session.depth_filled_mm[label_index]
        if np.any(label_valid):
            return float(np.median(label_depth_mm[label_valid]))
        return float(np.median(label_depth_mm))

    def _geometric_augment(self, color, depth_channel):
        """Input: (T,H,W,3) float32 color in [0,1] and (T,H,W) float32
        normalized depth_channel. Applies the enabled geometric
        augmentations (horizontal flip, optionally rotation) with one
        random draw per window, to color and depth together. Returns
        (color, depth_channel), same shapes.
        """
        aug = self.augmentation_config
        if np.random.rand() < aug["horizontal_flip_prob"]:
            color = color[:, :, ::-1, :].copy()
            depth_channel = depth_channel[:, :, ::-1].copy()
        if aug["rotation_enabled"]:
            angle_deg = np.random.uniform(-aug["rotation_max_deg"], aug["rotation_max_deg"])
            color, depth_channel = _rotate_and_recrop(color, depth_channel, angle_deg)
        return color, depth_channel

    def __getitem__(self, index):
        session_index, start = self.windows[index]
        session = self.sessions[session_index]
        if self.split == "train":
            start = self._augmented_start(session, start)
        end = start + self.window_length
        label_index = start + self.label_offset

        color = session.color[start:end].astype(np.float32) / 255.0   # (T,H,W,3) in [0,1], RGB
        depth_mm = session.depth_filled_mm[start:end]                  # (T,H,W) millimetres

        reference_mm = self._reference_depth_mm(session, label_index)
        diff_mm = depth_mm - reference_mm

        aug = self.augmentation_config
        if self.split == "train" and aug["depth_noise_enabled"]:
            noise_mm = np.random.normal(0.0, aug["depth_noise_std_mm"], size=diff_mm.shape).astype(np.float32)
            diff_mm = diff_mm + noise_mm

        # Sign convention (fixed, applies everywhere): positive = farther from
        # the camera than the window's reference depth, negative = closer.
        depth_channel = np.clip(diff_mm, -self.clip_mm, self.clip_mm) / self.clip_mm

        if self.split == "train":
            color, depth_channel = self._geometric_augment(color.copy(), depth_channel.copy())

        color = (color - RGB_MEAN) / RGB_STD

        color_chw = np.transpose(color, (3, 0, 1, 2))          # (3,T,H,W)
        depth_chw = depth_channel[np.newaxis, ...]              # (1,T,H,W)
        clip = np.concatenate([color_chw, depth_chw], axis=0).astype(np.float32)   # (4,T,H,W)

        label_n = float(session.frame_fz_n[label_index])
        return {
            "clip": torch.from_numpy(clip),
            "label_normalized": torch.tensor(label_n / self.force_norm_scale_n, dtype=torch.float32),
            "label_n": torch.tensor(label_n, dtype=torch.float32),
            "session": session.name,
            "start_frame": start,
        }


def build_dataset(split, config=None):
    """Input: split name ("train"/"val"/"test") and optionally an
    already-loaded config dict (loads config.yaml if omitted). Returns a
    ForceClipDataset for that split, built from config['model_training'].
    """
    config = config or load_config()
    return ForceClipDataset(config["model_training"], split)


def build_dataloader(split, config=None, shuffle=None):
    """Input: split name, optional config dict, optional shuffle override
    (defaults to True for "train", False otherwise). Returns
    (dataset, DataLoader) built from config['model_training']['loader'].
    """
    config = config or load_config()
    dataset = build_dataset(split, config)
    loader_config = config["model_training"]["loader"]
    shuffle = (split == "train") if shuffle is None else shuffle
    loader = DataLoader(
        dataset, batch_size=loader_config["batch_size"], shuffle=shuffle,
        num_workers=loader_config["num_workers"], pin_memory=True,
    )
    return dataset, loader


if __name__ == "__main__":
    print("[dataset] Building train/val/test datasets from config.yaml's model_training block")
    cfg = load_config()
    train_dataset, train_loader = build_dataloader("train", cfg)
    val_dataset, _ = build_dataloader("val", cfg)
    test_dataset, _ = build_dataloader("test", cfg)

    print(f"[dataset] Window counts: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")
    print(f"[dataset] Mean normalized train label: {train_dataset.mean_normalized_label():.4f} "
          f"({train_dataset.mean_normalized_label() * train_dataset.force_norm_scale_n:.2f} N)")

    batch = next(iter(train_loader))
    print(f"[dataset] One training batch: clip={tuple(batch['clip'].shape)} dtype={batch['clip'].dtype}  "
          f"label_normalized={tuple(batch['label_normalized'].shape)}  "
          f"sessions={sorted(set(batch['session']))}")
    print(f"  clip value range: min={batch['clip'].min():.3f} max={batch['clip'].max():.3f}")
    print(f"  label_n range in this batch: min={batch['label_n'].min():.2f} max={batch['label_n'].max():.2f} N")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preview_dir = os.path.join(os.path.dirname(__file__), "DatasetCheck")
    os.makedirs(preview_dir, exist_ok=True)
    num_samples = min(4, batch["clip"].shape[0])
    shown_frame_offsets = [0, train_dataset.label_offset, train_dataset.window_length - 1]
    fig, axes = plt.subplots(num_samples, len(shown_frame_offsets) * 2, figsize=(3 * len(shown_frame_offsets) * 2, 3 * num_samples))
    for row in range(num_samples):
        clip = batch["clip"][row]                         # (4,T,H,W) normalized
        label_n = batch["label_n"][row].item()
        for col, frame_offset in enumerate(shown_frame_offsets):
            color_frame = clip[:3, frame_offset].permute(1, 2, 0).numpy() * RGB_STD + RGB_MEAN
            depth_frame = clip[3, frame_offset].numpy()
            ax_color = axes[row, col * 2]
            ax_color.imshow(np.clip(color_frame, 0, 1))
            title = f"frame {frame_offset}" + (" (label)" if frame_offset == train_dataset.label_offset else "")
            if col == 0:
                title = f"Fz={label_n:.1f}N\n{title}"
            ax_color.set_title(title)
            ax_color.axis("off")
            ax_depth = axes[row, col * 2 + 1]
            ax_depth.imshow(depth_frame, cmap="coolwarm", vmin=-1, vmax=1)
            ax_depth.set_title("relative depth")
            ax_depth.axis("off")
    fig.suptitle("Training batch preview: color + relative-depth channel at window start/label/end frames")
    fig.tight_layout()
    preview_path = os.path.join(preview_dir, "batch_preview.png")
    fig.savefig(preview_path, dpi=130)
    plt.close(fig)
    print(f"[dataset] Batch preview saved to {preview_path}")
