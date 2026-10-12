"""Turns raw recordings into the per-frame dataset Model/ (3D) and Model2D/
(2D) train on - the RGB-D counterpart of ForceEstimationPressureVision's
preprocessing/dataProcessor.py, with the same label definition and frame
selection so the three models can be compared on the same frames.

Reads dataAcquisition/DataCollection/Raw/session_<ts>/ (color.mkv,
depth.mkv, frames_raw.csv, force_raw.csv, meta_raw.json, align_check.npz -
as written by dataAcquisition/dataWriter.RawSessionWriter) and writes
data/processed/session_<ts>/:

    crops.h5     color (N, px, px, 3) uint8 RGB, depth_raw (N, px, px) uint16
                 sensor counts (0 = no reading), depth_filled (N, px, px)
                 uint16 counts, hole-filled - row i belongs to frames.csv row i
    frames.csv   one row per saved crop: timestamps, force label, flags, crop geometry
    meta.json    settings used, camera calibration
    qc/          qc.json, force_plot.png, crop_strip.png, full_frame_check.jpg, frames_all.csv
    PROCESSED_OK written last; Model*/dataset.py only read sessions that have it

Per session:
  1. Force: zero-phase Butterworth low-pass of Fz, resampled onto every frame
     timestamp. Label force_n = max(-Fz, 0) (pressing reads negative). Frames
     where Fz goes strongly positive (forearm lifted off the plate, e.g. at
     the end of every session) are flagged as artifacts - saved, but the
     datasets skip them.
  2. Pass 1 (tracking): MediaPipe HandLandmarker in VIDEO mode on every full
     frame; fingertip/DIP smoothed zero-phase (Savitzky-Golay), short
     detection gaps interpolated, fingertip depth read every frame.
  3. Pass 2 (cropping): every tracked frame is cropped (crop_side_mm at the
     smoothed fingertip depth, as in dataAcquisition/offlineExtract.py) and
     saved if the crop lies mostly inside the image - no matter whether the
     finger touches. The fingertip depth check (8-point ring) is only stored
     as a flag (fingertip_depth_valid), it no longer drops frames: it failed
     on almost every hovering frame, which left the models with ~8% of the
     no-contact frames.
  4. Depth cleaning: zero-value holes inpainted + light median filter, per
     frame. The per-frame depth reference (median of the originally-valid
     pixels, mm) is stored as depth_ref_mm, so the datasets don't need the
     validity mask in memory.

Usage (trajPipeline env, from the repo root):
    python preprocessing/dataProcessor.py dataAcquisition/DataCollection/Raw/session_x [...]
    python preprocessing/dataProcessor.py --all               # every raw session not processed yet
    python preprocessing/dataProcessor.py --all --force       # re-process everything (e.g. new crop settings)
    python preprocessing/dataProcessor.py --all --workers 4   # sessions in parallel
"""

import argparse
import glob
import json
import math
import multiprocessing
import os
import shutil
import sys
import time
import traceback

import cv2
import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, sosfiltfilt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "config.yaml")
PROCESSED_MARKER = "PROCESSED_OK"

# Tracking, alignment and cropping are shared with the live recorder's
# offline extractor, so the crops look exactly like the ones the models were
# developed on.
sys.path.insert(0, os.path.join(REPO_ROOT, "dataAcquisition"))
import cameraHandler  # noqa: E402
import offlineExtract  # noqa: E402


def load_config(path=CONFIG_PATH):
    """Input: path to config.yaml. Returns it parsed as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def repo_path(path):
    """Returns path resolved against the repo root (absolute paths unchanged)."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def frame_force_labels(force_raw, frame_ts_ns, force_config):
    """Input: force_raw.csv as a DataFrame, (N,) frame host timestamps,
    config['processing']['force']. Zero-phase low-pass filters Fz at its
    own sample rate and resamples it onto the frame timestamps (shifted by
    frame_time_offset_ms). Returns (fz_filtered_n, force_n, artifact), all
    (N,): filtered Fz in newtons, the label max(-Fz, 0), and a bool flag for
    frames within artifact_margin_s of Fz > artifact_positive_n.
    """
    force_ts_ns = force_raw["host_ts_ns"].to_numpy().astype(np.int64)
    sample_rate_hz = 1e9 / np.median(np.diff(force_ts_ns.astype(np.float64)))
    sos = butter(force_config["order"], force_config["cutoff_hz"], btype="low", fs=sample_rate_hz, output="sos")
    fz_filtered = sosfiltfilt(sos, force_raw["fz_N"].to_numpy())

    lookup_ts = frame_ts_ns.astype(np.float64) + force_config["frame_time_offset_ms"] * 1e6
    fz_frames = np.interp(lookup_ts, force_ts_ns.astype(np.float64), fz_filtered)

    # Artifact mask on the force timeline, dilated by the margin, then sampled per frame.
    margin_ns = force_config["artifact_margin_s"] * 1e9
    artifact_samples = fz_filtered > force_config["artifact_positive_n"]
    artifact = np.zeros(len(frame_ts_ns), dtype=bool)
    for start, end in offlineExtract._run_slices(artifact_samples):
        t0 = force_ts_ns[start] - margin_ns
        t1 = force_ts_ns[end - 1] + margin_ns
        artifact |= (lookup_ts >= t0) & (lookup_ts <= t1)
    # Frames outside the force recording cannot be labelled.
    artifact |= (lookup_ts < force_ts_ns[0]) | (lookup_ts > force_ts_ns[-1])

    force_n = np.clip(-fz_frames, 0.0, None)
    return fz_frames.astype(np.float32), force_n.astype(np.float32), artifact


def outside_fraction(center, side_px, width, height):
    """Share of the (axis-aligned) crop square that lies outside the image."""
    half = side_px / 2
    inside_w = max(0.0, min(center[0] + half, width) - max(center[0] - half, 0))
    inside_h = max(0.0, min(center[1] + half, height) - max(center[1] - half, 0))
    return 1.0 - (inside_w * inside_h) / (side_px * side_px)


def clean_depth_crop(depth_raw_u16, depth_scale_m, depth_config):
    """Input: one raw (H, W) uint16 depth crop, the session's depth_scale_m,
    config['processing']['depth_cleaning']. Fills zero-valued holes by
    inpainting from the nearest valid neighbours, then applies a light
    median filter to knock down flying pixels along the finger silhouette -
    per frame only, nothing mixes in another frame. Returns (depth_filled
    uint16 counts, reference_mm, invalid_fraction): the reference is the
    median of the filled depth at the ORIGINALLY valid pixels, in
    millimetres (median of the whole crop if none were valid).
    """
    valid_mask = depth_raw_u16 > 0
    depth_mm = depth_raw_u16.astype(np.float32) * depth_scale_m * 1000.0
    if not np.any(valid_mask):
        return depth_raw_u16.copy(), 0.0, 1.0
    if np.all(valid_mask):
        hole_free_mm = depth_mm
    else:
        hole_mask = (~valid_mask).astype(np.uint8) * 255
        hole_free_mm = cv2.inpaint(depth_mm, hole_mask, depth_config["inpaint_radius_px"], cv2.INPAINT_TELEA)
    ksize = depth_config["median_ksize"]
    filled_mm = cv2.medianBlur(hole_free_mm, ksize) if ksize > 1 else hole_free_mm
    reference_mm = float(np.median(filled_mm[valid_mask]))
    filled_counts = np.clip(np.round(filled_mm / (depth_scale_m * 1000.0)), 0, 65535).astype(np.uint16)
    return filled_counts, reference_mm, float(1.0 - valid_mask.mean())


def _write_qc(qc_dir, frames_df, all_frames, strip, full_frame_preview, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(qc_dir, exist_ok=True)
    with open(os.path.join(qc_dir, "qc.json"), "w") as f:
        json.dump(stats, f, indent=2)
    all_frames.to_csv(os.path.join(qc_dir, "frames_all.csv"), index=False)

    t0 = all_frames["host_ts_ns"].iloc[0]
    t_all = (all_frames["host_ts_ns"] - t0) / 1e9
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t_all, all_frames["fz_filtered_n"], lw=0.6, color="0.6", label="Fz filtered (all frames)")
    usable = frames_df[~frames_df["artifact"].astype(bool)]
    ax.scatter((usable["host_ts_ns"] - t0) / 1e9, usable["force_n"], s=0.5, color="C0", label="label force_n (saved, usable)")
    artifact = all_frames["artifact"].to_numpy().astype(bool)
    for start, end in offlineExtract._run_slices(artifact):
        ax.axvspan(t_all.iloc[start], t_all.iloc[end - 1], color="C3", alpha=0.2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("N")
    ax.set_title(f"{stats['session']}: red = artifact (excluded)")
    ax.legend(loc="upper right", markerscale=10)
    fig.tight_layout()
    fig.savefig(os.path.join(qc_dir, "force_plot.png"), dpi=90)
    plt.close(fig)

    if strip:
        tiles = []
        for color, depth, label in strip:
            tile = np.vstack([cv2.cvtColor(color, cv2.COLOR_RGB2BGR), offlineExtract._depth_to_bgr(depth)])
            cv2.putText(tile, f"{label:.1f}N", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(tile, f"{label:.1f}N", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        cols = 8
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        cv2.imwrite(os.path.join(qc_dir, "crop_strip.png"), np.vstack(rows))
    if full_frame_preview is not None:
        cv2.imwrite(os.path.join(qc_dir, "full_frame_check.jpg"), full_frame_preview, [cv2.IMWRITE_JPEG_QUALITY, 90])


def process_session(raw_dir, config):
    """Input: raw session directory, full config dict. Writes the processed
    session to processing.output_root. Returns the processed session
    directory. Raises on failure (no PROCESSED_OK then).
    """
    started = time.time()
    proc_config = config["processing"]
    tracking_config = proc_config["hand_tracking"]
    crop_config = proc_config["crop"]
    output_px = crop_config["output_px"]

    raw_dir = os.path.abspath(raw_dir)
    session_name = os.path.basename(raw_dir.rstrip("/"))
    out_dir = os.path.join(repo_path(proc_config["output_root"]), session_name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    with open(os.path.join(raw_dir, "meta_raw.json")) as f:
        meta_raw = json.load(f)
    calibration = meta_raw["camera"]["stream_calibration"]
    color_intrinsics = calibration["color_intrinsics"]
    depth_intrinsics = calibration["depth_intrinsics"]
    depth_scale_m = meta_raw["camera"]["depth"]["depth_scale_m"]
    width, height = color_intrinsics["width"], color_intrinsics["height"]
    hand_label = tracking_config["tracked_hand_label"] or meta_raw["hand_tracking"]["tracked_hand_label"]
    # cameraHandler's helpers read their settings from config['hand_tracking'].
    handler_config = {"hand_tracking": {**tracking_config, "tracked_hand_label": hand_label,
                                        "model_path": repo_path(tracking_config["model_path"])}}

    frames_raw = pd.read_csv(os.path.join(raw_dir, "frames_raw.csv"))
    num_frames = len(frames_raw)
    frame_ts_ns = frames_raw["host_ts_ns"].to_numpy().astype(np.int64)
    print(f"[process] {session_name}: {num_frames} raw frames at {width}x{height}, tracking the '{hand_label}' hand")

    force_raw = pd.read_csv(os.path.join(raw_dir, "force_raw.csv"))
    fz_filtered_n, force_n, artifact = frame_force_labels(force_raw, frame_ts_ns, proc_config["force"])

    aligner = cameraHandler.build_depth_aligner(calibration, depth_scale_m)
    align_stats = offlineExtract.check_alignment(aligner, dict(np.load(os.path.join(raw_dir, "align_check.npz"))),
                                                 proc_config["align_check_tolerance_counts"])
    if not align_stats["passed"]:
        print(f"[process]   {session_name}: WARNING numpy depth alignment disagrees with rs.align - see qc/qc.json")

    def color_frames():
        return offlineExtract.read_video_frames(os.path.join(raw_dir, "color.mkv"), width, height, "rgb24", 3, np.uint8)

    def depth_frames():
        return offlineExtract.read_video_frames(os.path.join(raw_dir, "depth.mkv"), depth_intrinsics["width"],
                                                depth_intrinsics["height"], "gray16le", 1, np.dtype("<u2"))

    # Pass 1: tracking + fingertip depth.
    import mediapipe as mp
    tip_index = tracking_config["fingertip_landmark_index"]
    dip_index = tracking_config["dip_landmark_index"]
    tip = np.full((num_frames, 2), np.nan)
    dip = np.full((num_frames, 2), np.nan)
    tip_depth_m = np.full(num_frames, np.nan)
    landmarker = cameraHandler.build_hand_landmarker(handler_config, running_mode="VIDEO")
    last_timestamp_ms = None
    decoded = 0
    try:
        for i, (color_rgb, depth_raw) in enumerate(zip(color_frames(), depth_frames())):
            if i >= num_frames:
                break
            decoded += 1
            last_timestamp_ms = cameraHandler.next_mediapipe_timestamp_ms(int(frame_ts_ns[i]), last_timestamp_ms)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(color_rgb)), last_timestamp_ms)
            landmarks = cameraHandler.select_tracked_hand(result, handler_config)
            if landmarks is not None:
                tip[i] = landmarks[tip_index].x * width, landmarks[tip_index].y * height
                dip[i] = landmarks[dip_index].x * width, landmarks[dip_index].y * height
                tip_depth_m[i] = offlineExtract._center_depth_m(aligner, depth_raw, tip[i])
            if (i + 1) % 1000 == 0:
                print(f"[process]   {session_name} pass 1 (tracking): {i + 1}/{num_frames}")
    finally:
        landmarker.close()
    if decoded != num_frames:
        raise RuntimeError(f"decoded {decoded} frames but frames_raw.csv has {num_frames}")

    detected = ~np.isnan(tip[:, 0])
    smoothed, interpolated = offlineExtract.fill_and_smooth(np.hstack([tip, dip]), tracking_config)
    tip_s, dip_s = smoothed[:, :2], smoothed[:, 2:]
    tracked = ~np.isnan(tip_s[:, 0])
    depth_s, _ = offlineExtract.fill_and_smooth(offlineExtract._fill_within_runs(tip_depth_m, tracked)[:, None],
                                                tracking_config)
    depth_s = depth_s[:, 0]

    axis = tip_s - dip_s
    axis_len = np.hypot(axis[:, 0], axis[:, 1])
    centers = tip_s - crop_config["anchor_alpha"] * axis
    angles = np.arctan2(axis[:, 0], -axis[:, 1]) if crop_config["rotate_to_finger_axis"] else np.zeros(num_frames)
    sides = crop_config["side_mm"] / 1000.0 * color_intrinsics["fx"] / depth_s

    # QC strip: evenly spread over the tracked, non-artifact frames.
    croppable = tracked & ~np.isnan(sides) & (axis_len > 0)
    strip_pool = np.flatnonzero(croppable & ~artifact)
    strip_pool = strip_pool if strip_pool.size else np.flatnonzero(croppable)
    strip_n = min(proc_config["qc_strip_frames"], strip_pool.size)
    strip_indices = set(strip_pool[np.linspace(0, strip_pool.size - 1, strip_n).astype(int)].tolist()) if strip_n else set()

    # Pass 2: crop, clean depth, collect.
    rows, all_rows, strip = [], [], []
    colors, depths_raw, depths_filled = [], [], []
    croppable_indices = np.flatnonzero(croppable)
    preview_index = int(croppable_indices[len(croppable_indices) // 2]) if croppable_indices.size else None
    full_frame_preview = None
    for i, (color_rgb, depth_raw) in enumerate(zip(color_frames(), depth_frames())):
        if i >= num_frames:
            break
        saved = False
        fraction_outside = float("nan")
        depth_info = cameraHandler.empty_depth_info()
        if croppable[i]:
            fraction_outside = outside_fraction(centers[i], sides[i], width, height)
            if fraction_outside <= crop_config["max_outside_fraction"]:
                depth_info = cameraHandler.validate_fingertip_depth_unaligned(aligner, depth_raw, tip_s[i], handler_config)
                color_crop, depth_crop = offlineExtract.crop_frame(color_rgb, depth_raw, aligner, centers[i], sides[i],
                                                                   angles[i], output_px)
                depth_filled, depth_ref_mm, invalid_fraction = clean_depth_crop(
                    depth_crop, depth_scale_m, proc_config["depth_cleaning"])
                colors.append(color_crop)
                depths_raw.append(depth_crop)
                depths_filled.append(depth_filled)
                rows.append({
                    "raw_frame_index": i,
                    "host_ts_ns": int(frame_ts_ns[i]),
                    "fz_filtered_n": round(float(fz_filtered_n[i]), 4),
                    "force_n": round(float(force_n[i]), 4),
                    "artifact": int(artifact[i]),
                    "detected": int(detected[i]),
                    "interpolated": int(interpolated[i]),
                    "fingertip_depth_valid": int(depth_info["valid"]),
                    "tip_x": round(float(tip_s[i][0]), 2), "tip_y": round(float(tip_s[i][1]), 2),
                    "tip_depth_m": round(float(depth_s[i]), 5),
                    "crop_cx": round(float(centers[i][0]), 2), "crop_cy": round(float(centers[i][1]), 2),
                    "crop_side_px": round(float(sides[i]), 2),
                    "crop_angle_deg": round(math.degrees(angles[i]), 3),
                    "outside_fraction": round(fraction_outside, 4),
                    "depth_invalid_fraction": round(invalid_fraction, 4),
                    "depth_ref_mm": round(depth_ref_mm, 3),
                })
                if i in strip_indices:
                    strip.append((color_crop, depth_crop, float(force_n[i])))
                saved = True
        if i == preview_index:
            preview = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            half = sides[i] / 2
            cv2.rectangle(preview, (int(centers[i][0] - half), int(centers[i][1] - half)),
                          (int(centers[i][0] + half), int(centers[i][1] + half)), (0, 255, 0), 2)
            cv2.circle(preview, tuple(np.round(tip_s[i]).astype(int)), 4, (0, 0, 255), -1)
            full_frame_preview = preview
        all_rows.append({
            "raw_frame_index": i, "host_ts_ns": int(frame_ts_ns[i]),
            "fz_filtered_n": float(fz_filtered_n[i]), "force_n": float(force_n[i]), "artifact": int(artifact[i]),
            "detected": int(detected[i]), "tracked": int(tracked[i]), "saved": int(saved),
            "fingertip_depth_valid": int(depth_info["valid"]),
            "tip_depth_m": depth_s[i], "outside_fraction": fraction_outside,
        })
        if (i + 1) % 1000 == 0:
            print(f"[process]   {session_name} pass 2 (cropping): {i + 1}/{num_frames}")

    if not rows:
        raise RuntimeError("no frames saved - hand never tracked?")
    frames_df = pd.DataFrame(rows)
    frames_df.to_csv(os.path.join(out_dir, "frames.csv"), index=False)
    with h5py.File(os.path.join(out_dir, "crops.h5"), "w") as h5f:
        h5f.create_dataset("color", data=np.stack(colors), compression="gzip")
        h5f.create_dataset("depth_raw", data=np.stack(depths_raw), compression="gzip")
        h5f.create_dataset("depth_filled", data=np.stack(depths_filled), compression="gzip")
    all_frames = pd.DataFrame(all_rows)

    usable = (frames_df["artifact"] == 0).to_numpy()
    stats = {
        "session": session_name,
        "raw_frames": num_frames,
        "saved_frames": len(frames_df),
        "usable_frames": int(usable.sum()),
        "detected_pct": float(detected.mean() * 100),
        "tracked_pct": float(tracked.mean() * 100),
        "artifact_pct": float(artifact.mean() * 100),
        "fingertip_depth_valid_pct_saved": float(frames_df["fingertip_depth_valid"].mean() * 100),
        "hand_label": hand_label,
        "median_crop_side_px": float(frames_df["crop_side_px"].median()),
        "median_tip_depth_m": float(frames_df["tip_depth_m"].median()),
        "mean_depth_invalid_fraction": float(frames_df["depth_invalid_fraction"].mean()),
        "force_n_percentiles_usable": dict(zip(["p5", "p50", "p95", "max"], np.percentile(
            frames_df["force_n"].to_numpy()[usable], [5, 50, 95, 100]).round(2).tolist())) if usable.any() else None,
        "contact_pct_usable": float((frames_df["force_n"].to_numpy()[usable] > 0.5).mean() * 100) if usable.any() else None,
        "alignment_check": align_stats,
        "processing_time_s": round(time.time() - started, 1),
    }
    _write_qc(os.path.join(out_dir, "qc"), frames_df, all_frames, strip, full_frame_preview, stats)

    meta = {
        "session_name": session_name,
        "raw_session_dir": raw_dir,
        "processing": proc_config,
        "hand_label": hand_label,
        "output_px": output_px,
        "force_label": "force_n = max(-Fz_filtered, 0), newtons; Fz reads negative while pressing",
        "depth": "crops.h5 depth_raw / depth_filled in sensor counts, multiply by camera.depth.depth_scale_m for metres",
        "camera": meta_raw["camera"],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out_dir, PROCESSED_MARKER), "w") as f:
        json.dump({"saved_frames": len(frames_df), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    print(f"[process] {session_name}: saved {len(frames_df)}/{num_frames} crops ({stats['usable_frames']} usable, "
          f"{stats['contact_pct_usable']:.0f}% in contact), {stats['processing_time_s']}s")
    return out_dir


def _process_one(raw_dir):
    """Worker entry point: returns (raw_dir, error string or None)."""
    try:
        process_session(raw_dir, load_config())
        return raw_dir, None
    except Exception as e:  # keep going with the other sessions
        traceback.print_exc()
        return raw_dir, repr(e)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_session_dirs", nargs="*", help="raw session_* folders to process")
    parser.add_argument("--all", action="store_true", help="process every raw session under processing.raw_root")
    parser.add_argument("--force", action="store_true", help="with --all: also re-process finished sessions")
    parser.add_argument("--workers", type=int, default=1, help="number of sessions processed in parallel")
    args = parser.parse_args()

    config = load_config()
    proc_config = config["processing"]
    raw_dirs = list(args.raw_session_dirs)
    if args.all:
        for raw_dir in sorted(glob.glob(os.path.join(repo_path(proc_config["raw_root"]), "session_*"))):
            done = os.path.exists(os.path.join(repo_path(proc_config["output_root"]),
                                               os.path.basename(raw_dir), PROCESSED_MARKER))
            if args.force or not done:
                raw_dirs.append(raw_dir)
    if not raw_dirs:
        parser.error("no raw sessions to process (pass folders or --all; finished sessions need --force)")

    print(f"[process] {len(raw_dirs)} session(s), {args.workers} worker(s)")
    if args.workers > 1:
        with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
            results = pool.map(_process_one, raw_dirs, chunksize=1)
    else:
        results = [_process_one(raw_dir) for raw_dir in raw_dirs]

    failures = [(raw_dir, error) for raw_dir, error in results if error]
    for raw_dir, error in failures:
        print(f"[process] FAILED {raw_dir}: {error}")
    print(f"[process] done: {len(results) - len(failures)} ok, {len(failures)} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
