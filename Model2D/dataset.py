"""Dataset preparation for the single-frame (2D CNN) fingertip force
estimator - the 2D counterpart of Model/dataset.py, used to test whether
the 3D CNN's temporal context is actually needed.

Reads the session folders preprocessing/dataProcessor.py writes (frames.csv
with the per-frame label force_n = max(-Fz_filtered, 0) and artifact flag,
crops.h5 with the color and hole-filled depth crops, meta.json - see that
file's docstring). Every tracked frame is in there, also the ones without
contact. This file indexes valid 20-frame windows per config.yaml's
model2d_training block (see window_reject_reason) - exactly like Model/dataset.py, so
with matching window/split settings both models see the same labelled
samples. The difference: each sample is only the window's label (centre)
frame, a normalized (4, 112, 112) RGB+relative-depth image with that
frame's force label - the other 19 frames only decide which label frames
are kept (the window filters), never reach the network. Consumed by
train.py via build_dataset().
"""

import json
import os

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
REPO_ROOT = os.path.dirname(os.path.abspath(CONFIG_PATH))

CONFIG_SECTION = "model2d_training"
SPLIT_NAMES = ("train", "val", "test")
PROCESSED_MARKER = "PROCESSED_OK"   # written by preprocessing/dataProcessor.py once a session is complete

# ImageNet per-channel stats, RGB order - the 2D backbone is ImageNet
# pretrained (the 3D model uses Kinetics-400 stats for its Kinetics
# backbone). crops.h5's "color" dataset is already RGB (dataProcessor.py
# decodes the video as rgb24), so no channel-order conversion happens here,
# only these stats.
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_config(path=CONFIG_PATH):
    """Input: path to config.yaml. Returns it parsed as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def discover_sessions(model_config):
    """Input: config['model2d_training']. Returns the sorted names of every
    session_* folder in data_root that dataProcessor.py finished.
    """
    data_root = os.path.join(REPO_ROOT, model_config["data_root"])
    return sorted(name for name in os.listdir(data_root)
                  if name.startswith("session_") and os.path.isfile(os.path.join(data_root, name, PROCESSED_MARKER)))


def resolve_splits(model_config):
    """Input: config['model2d_training']. Returns {"train": [...], "val":
    [...], "test": [...]} session names. Explicit <split>_sessions lists in
    the splits block are used as-is if any is present; otherwise all
    discovered sessions are shuffled with split_seed and divided by the
    fractions (val/test get at least one session each if their fraction is
    non-zero). Raises ValueError on bad fractions or too few sessions.
    """
    splits_config = model_config["splits"]
    if any(f"{split}_sessions" in splits_config for split in SPLIT_NAMES):
        return {split: list(splits_config.get(f"{split}_sessions") or []) for split in SPLIT_NAMES}

    fractions = {split: float(splits_config[f"{split}_fraction"]) for split in SPLIT_NAMES}
    if any(fraction < 0 for fraction in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must be >= 0 and sum to 1, got {fractions}")

    names = discover_sessions(model_config)
    rng = np.random.default_rng(splits_config.get("split_seed", 0))
    shuffled = [names[i] for i in rng.permutation(len(names))]

    def count(split):
        return 0 if fractions[split] == 0 else max(1, int(round(fractions[split] * len(shuffled))))

    num_val, num_test = count("val"), count("test")
    num_train = len(shuffled) - num_val - num_test
    if num_train < 1:
        raise ValueError(f"only {len(shuffled)} processed session(s) in {model_config['data_root']} - "
                         f"not enough for a train/val/test split with fractions {fractions}")
    return {
        "train": sorted(shuffled[:num_train]),
        "val": sorted(shuffled[num_train:num_train + num_val]),
        "test": sorted(shuffled[num_train + num_val:]),
    }


def label_frame_offset(window_length):
    """Input: window length in frames. Returns the single, fixed local
    frame index (0-based, within the window) the force label is read from -
    the centre frame. Both dataset.py and any later inference code must
    call this rather than hardcode the offset, or every force estimate is
    shifted in time relative to its clip.
    """
    return window_length // 2


class _SessionData:
    """Holds one session's arrays in memory: color crops, hole-filled depth
    (uint16 sensor counts - converted to mm per item, which keeps the whole
    dataset at 5 bytes per pixel), the per-frame depth reference, force
    label, artifact flag and timestamps. crops.h5 is opened once here and
    closed before __init__ returns - only plain numpy arrays are kept
    afterwards, so there's no open HDF5 handle to survive DataLoader worker
    forking (h5py file handles are not fork-safe; numpy arrays are).
    """

    def __init__(self, session_dir, model_config):
        self.name = os.path.basename(session_dir)
        frames_df = pd.read_csv(os.path.join(session_dir, "frames.csv"))
        with open(os.path.join(session_dir, "meta.json")) as f:
            meta = json.load(f)
        self.depth_scale_mm = float(meta["camera"]["depth"]["depth_scale_m"]) * 1000.0

        self.frame_ts_ns = frames_df["host_ts_ns"].to_numpy()
        self.raw_frame_index = frames_df["raw_frame_index"].to_numpy()   # frame id shared with PressureVision's data
        self.frame_force_n = frames_df["force_n"].to_numpy().astype(np.float32)
        self.frame_artifact = frames_df["artifact"].to_numpy().astype(bool)
        self.depth_ref_mm = frames_df["depth_ref_mm"].to_numpy().astype(np.float32)
        self.original_invalid_fraction = float(frames_df["depth_invalid_fraction"].mean())

        with h5py.File(os.path.join(session_dir, "crops.h5"), "r") as h5f:
            self.color = h5f["color"][:]                # (N, 112, 112, 3) uint8, RGB
            self.depth_filled = h5f["depth_filled"][:]  # (N, 112, 112) uint16, hole-filled sensor counts
        self.num_frames = len(frames_df)
        if self.color.shape[0] != self.num_frames:
            raise ValueError(f"{self.name}: crops.h5 has {self.color.shape[0]} frames, frames.csv {self.num_frames}")

        self.gap_after = self._find_gap_after(model_config["window"])
        # Cumulative artifact count, so "any artifact in [start, end]" is O(1) per window.
        self.artifact_cumsum = np.concatenate([[0], np.cumsum(self.frame_artifact)])

    def depth_mm(self, index):
        """Returns the hole-filled depth of frame(s) `index` in millimetres, float32."""
        return self.depth_filled[index].astype(np.float32) * self.depth_scale_mm

    def _find_gap_after(self, window_config):
        """Returns a (num_frames-1,) bool array: True at index i if the
        timestamp gap between frame i and i+1 alone already exceeds
        max_span_s - used to bound the temporal-offset augmentation to a
        gap-free run of frames.
        """
        dt_s = np.diff(self.frame_ts_ns.astype(np.float64)) / 1e9
        return dt_s > window_config["max_span_s"]


def window_reject_reason(session, start, window_config, label_offset):
    """Input: a _SessionData, window start frame, config['model2d_training']['window'] and
    the label frame offset. Returns None if the window may be used,
    otherwise the name of the first rule it breaks:

      "span"     first-to-last-frame elapsed time > max_span_s, from real
                 timestamps. frames.csv only holds tracked frames, so a
                 tracking loss or dropped camera frames show up as a
                 timestamp gap - this keeps every clip temporally contiguous.
      "artifact" any frame of the window is a force artifact (forearm
                 lift-off, or outside the force recording) - its label is
                 not a press, and the clip shows a different motion.
      "label"    the label exceeds max_label_force_n (PressureVision's
                 max_force_n - no presses this hard were intended).
      "change"   |force(last) - force(first)| > max_force_change_n; only if
                 that is set (null = off). Off by default: it removes the
                 fastest presses and releases, exactly where the temporal
                 context should help, and PressureVision has no such rule.
    """
    end = start + window_config["length"] - 1
    if (session.frame_ts_ns[end] - session.frame_ts_ns[start]) / 1e9 > window_config["max_span_s"]:
        return "span"
    if session.artifact_cumsum[end + 1] - session.artifact_cumsum[start] > 0:
        return "artifact"
    if session.frame_force_n[start + label_offset] > window_config["max_label_force_n"]:
        return "label"
    max_change = window_config.get("max_force_change_n")
    if max_change is not None and abs(session.frame_force_n[end] - session.frame_force_n[start]) > max_change:
        return "change"
    return None


def _build_window_index(sessions, window_config, stride, offset, split_name):
    """Input: list of _SessionData for one split, config['model2d_training']
    ['window'], the window-start stride to use, the label frame offset
    (from label_frame_offset()), and the split name (for the printed
    report). Slides a window of window_config['length'] frames across every
    session at the given stride and keeps only windows passing every rule in
    window_reject_reason(). Prints candidate/discarded/kept counts. Returns
    a list of (session_index, start_frame) tuples.
    """
    window_length = window_config["length"]
    kept, num_candidates = [], 0
    discarded = {"span": 0, "artifact": 0, "label": 0, "change": 0}
    for session_index, session in enumerate(sessions):
        last_start = session.num_frames - window_length
        for start in range(0, max(last_start + 1, 0), stride):
            num_candidates += 1
            reason = window_reject_reason(session, start, window_config, offset)
            if reason is None:
                kept.append((session_index, start))
            else:
                discarded[reason] += 1

    print(f"  [dataset] {split_name}: {num_candidates} candidate windows, discarded: "
          f"{discarded['span']} span > {window_config['max_span_s']}s, "
          f"{discarded['artifact']} artifact frames, "
          f"{discarded['label']} label > {window_config['max_label_force_n']}N, "
          f"{discarded['change']} |dF| > {window_config.get('max_force_change_n')}N - "
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
    """Input: (H,W,3) color in [0,1] and (H,W) normalized depth_channel of
    one frame, and one rotation angle in degrees. Rotates about the image
    centre, centre-crops to the largest square with no empty border, and
    resizes back to (H, W). Color uses linear interpolation; depth uses
    nearest only, never cubic - cubic overshoots at the sharp finger/skin
    depth step and invents values that don't exist. Returns
    (color, depth_channel), same shapes as input.
    """
    height, width = depth_channel.shape
    assert height == width, "rotation crop-size formula assumes a square crop"
    crop_size = _largest_inscribed_crop_size(height, angle_deg)
    y0, x0 = (height - crop_size) // 2, (width - crop_size) // 2

    rot_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1.0)
    rotated_color = cv2.warpAffine(color, rot_matrix, (width, height), flags=cv2.INTER_LINEAR)
    rotated_depth = cv2.warpAffine(depth_channel, rot_matrix, (width, height), flags=cv2.INTER_NEAREST)

    cropped_color = rotated_color[y0:y0 + crop_size, x0:x0 + crop_size, :]
    cropped_depth = rotated_depth[y0:y0 + crop_size, x0:x0 + crop_size]

    resized_color = cv2.resize(cropped_color, (width, height), interpolation=cv2.INTER_LINEAR)
    resized_depth = cv2.resize(cropped_depth, (width, height), interpolation=cv2.INTER_NEAREST)
    return resized_color, resized_depth


class ForceFrameDataset(Dataset):
    """torch Dataset over single RGB+relative-depth frames. Samples are
    indexed by the same (session, window start) pairs as Model/dataset.py's
    ForceClipDataset - so the same filters and strides apply - but each
    item is only the window's label frame. `split` is "train", "val" or
    "test" - only "train" uses the training window stride and applies
    augmentation; the others are passed through untouched at eval_stride.
    """

    def __init__(self, model_config, split):
        window_config = model_config["window"]
        self.split = split
        self.window_config = window_config
        self.window_length = window_config["length"]
        self.label_offset = label_frame_offset(self.window_length)
        self.clip_mm = model_config["depth"]["clip_mm"]
        self.force_norm_scale_n = model_config["label"]["force_norm_scale_n"]
        self.augmentation_config = model_config["augmentation"]

        session_names = resolve_splits(model_config)[split]
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
        labels = [self.sessions[session_index].frame_force_n[start + self.label_offset] / self.force_norm_scale_n
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
            if window_reject_reason(session, start, self.window_config, self.label_offset) is None:
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
        was originally invalid (rare). Precomputed per frame by
        preprocessing/dataProcessor.py (depth_ref_mm in frames.csv).
        """
        return float(session.depth_ref_mm[label_index])   # precomputed by dataProcessor.py

    def _geometric_augment(self, color, depth_channel):
        """Input: (H,W,3) float32 color in [0,1] and (H,W) float32
        normalized depth_channel. Applies the enabled geometric
        augmentations (horizontal flip, optionally rotation) with one
        random draw per sample, to color and depth together. Returns
        (color, depth_channel), same shapes.
        """
        aug = self.augmentation_config
        if np.random.rand() < aug["horizontal_flip_prob"]:
            color = color[:, ::-1, :].copy()
            depth_channel = depth_channel[:, ::-1].copy()
        if aug["rotation_enabled"]:
            angle_deg = np.random.uniform(-aug["rotation_max_deg"], aug["rotation_max_deg"])
            color, depth_channel = _rotate_and_recrop(color, depth_channel, angle_deg)
        return color, depth_channel

    def __getitem__(self, index):
        session_index, start = self.windows[index]
        session = self.sessions[session_index]
        if self.split == "train":
            start = self._augmented_start(session, start)
        label_index = start + self.label_offset

        color = session.color[label_index].astype(np.float32) / 255.0   # (H,W,3) in [0,1], RGB
        depth_mm = session.depth_mm(label_index)                         # (H,W) millimetres

        # Same reference as the 3D model; for a single frame it is simply
        # that frame's own median valid depth.
        reference_mm = self._reference_depth_mm(session, label_index)
        diff_mm = depth_mm - reference_mm

        aug = self.augmentation_config
        if self.split == "train" and aug["depth_noise_enabled"]:
            noise_mm = np.random.normal(0.0, aug["depth_noise_std_mm"], size=diff_mm.shape).astype(np.float32)
            diff_mm = diff_mm + noise_mm

        # Sign convention (fixed, applies everywhere): positive = farther from
        # the camera than the frame's reference depth, negative = closer.
        depth_channel = np.clip(diff_mm, -self.clip_mm, self.clip_mm) / self.clip_mm

        if self.split == "train":
            color, depth_channel = self._geometric_augment(color.copy(), depth_channel.copy())

        color = (color - RGB_MEAN) / RGB_STD

        color_chw = np.transpose(color, (2, 0, 1))             # (3,H,W)
        depth_chw = depth_channel[np.newaxis, ...]              # (1,H,W)
        image = np.concatenate([color_chw, depth_chw], axis=0).astype(np.float32)   # (4,H,W)

        label_n = float(session.frame_force_n[label_index])
        return {
            "image": torch.from_numpy(image),
            "label_normalized": torch.tensor(label_n / self.force_norm_scale_n, dtype=torch.float32),
            "label_n": torch.tensor(label_n, dtype=torch.float32),
            "session": session.name,
            "start_frame": start,
            "label_frame": label_index,
        }


def build_dataset(split, config=None):
    """Input: split name ("train"/"val"/"test") and optionally an
    already-loaded config dict (loads config.yaml if omitted). Returns a
    ForceFrameDataset for that split, built from config['model2d_training'].
    """
    config = config or load_config()
    return ForceFrameDataset(config[CONFIG_SECTION], split)


def build_dataloader(split, config=None, shuffle=None):
    """Input: split name, optional config dict, optional shuffle override
    (defaults to True for "train", False otherwise). Returns
    (dataset, DataLoader) built from config['model2d_training']['loader'].
    The loader gets its own generator seeded from training.seed, so the
    shuffle order and the per-worker numpy seeds (augmentation) don't
    depend on how much of torch's global RNG was used before it.
    """
    config = config or load_config()
    dataset = build_dataset(split, config)
    loader_config = config[CONFIG_SECTION]["loader"]
    shuffle = (split == "train") if shuffle is None else shuffle
    seed = config[CONFIG_SECTION]["training"].get("seed")
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    loader = DataLoader(
        dataset, batch_size=loader_config["batch_size"], shuffle=shuffle,
        num_workers=loader_config["num_workers"], pin_memory=True, generator=generator,
    )
    return dataset, loader


if __name__ == "__main__":
    print("[dataset] Building train/val/test datasets from config.yaml's model2d_training block")
    cfg = load_config()
    for split_name, names in resolve_splits(cfg[CONFIG_SECTION]).items():
        print(f"[dataset] {split_name} ({len(names)}): {names}")
    train_dataset, train_loader = build_dataloader("train", cfg)
    val_dataset, _ = build_dataloader("val", cfg)
    test_dataset, _ = build_dataloader("test", cfg)

    print(f"[dataset] Window counts: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")
    print(f"[dataset] Mean normalized train label: {train_dataset.mean_normalized_label():.4f} "
          f"({train_dataset.mean_normalized_label() * train_dataset.force_norm_scale_n:.2f} N)")

    batch = next(iter(train_loader))
    print(f"[dataset] One training batch: image={tuple(batch['image'].shape)} dtype={batch['image'].dtype}  "
          f"label_normalized={tuple(batch['label_normalized'].shape)}  "
          f"sessions={sorted(set(batch['session']))}")
    print(f"  image value range: min={batch['image'].min():.3f} max={batch['image'].max():.3f}")
    print(f"  label_n range in this batch: min={batch['label_n'].min():.2f} max={batch['label_n'].max():.2f} N")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preview_dir = os.path.join(os.path.dirname(__file__), "DatasetCheck")
    os.makedirs(preview_dir, exist_ok=True)
    num_samples = min(4, batch["image"].shape[0])
    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples))
    for row in range(num_samples):
        image = batch["image"][row]                       # (4,H,W) normalized
        label_n = batch["label_n"][row].item()
        color_frame = image[:3].permute(1, 2, 0).numpy() * RGB_STD + RGB_MEAN
        axes[row, 0].imshow(np.clip(color_frame, 0, 1))
        axes[row, 0].set_title(f"Fz={label_n:.1f}N  frame {batch['label_frame'][row].item()}")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(image[3].numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        axes[row, 1].set_title("relative depth")
        axes[row, 1].axis("off")
    fig.suptitle("Training batch preview: label frame color + relative depth")
    fig.tight_layout()
    preview_path = os.path.join(preview_dir, "batch_preview.png")
    fig.savefig(preview_path, dpi=130)
    plt.close(fig)
    print(f"[dataset] Batch preview saved to {preview_path}")
