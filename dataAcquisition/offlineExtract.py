"""Offline extraction of the fingertip dataset from a raw recording.

Turns DataCollection/Raw/session_<ts>/ (written live by
dataWriter.RawSessionWriter) into DataCollection/Data/session_<ts>/ in the
format Model/dataset.py reads (video.h5, frames.csv, force.csv, meta.json,
README.md), plus a qc/ folder and a PROCESSED_OK marker written last.

Doing this offline instead of in the live loop is what allows recording at
1920x1080 without losing frames, and it fixes the unstable crops of the
first dataset:
  * MediaPipe runs in VIDEO mode - synchronous, one result per frame - so
    every crop is placed with the landmarks of its own frame (the live
    LIVE_STREAM mode returned the latest finished result, i.e. a stale one,
    which made the crop lag a sliding finger and then jump).
  * The crop center is placed on the finger itself (TIP moved back towards
    DIP by anchor_alpha) instead of on the extreme tip landmark.
  * The trajectory is smoothed zero-phase (Savitzky-Golay over whole runs),
    which is impossible live, and short detection gaps are interpolated.
  * The crop covers a fixed physical size (crop_side_mm at the fingertip
    depth), is placed with sub-pixel accuracy (one warpAffine), optionally
    rotated to the finger axis, and downsampled with anti-aliasing - at
    1080p that averages ~3x3 sensor pixels per output pixel.

Usage (trajPipeline env, from anywhere):
    python dataAcquisition/offlineExtract.py DataCollection/Raw/session_x [...]
    python dataAcquisition/offlineExtract.py --all      # every raw session not yet processed
    python dataAcquisition/offlineExtract.py --all --force   # re-process everything (e.g. new crop settings)
mainAcquisition.py runs it automatically when a recording stops.
"""

import argparse
import csv
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import cameraHandler
import dataWriter
from mainAcquisition import load_config

PROCESSED_MARKER = "PROCESSED_OK"


def read_video_frames(path, width, height, pix_fmt, channels, dtype):
    """Input: video path, frame size, ffmpeg output pixel format ("rgb24" or
    "gray16le"), channels per pixel and numpy dtype. Yields decoded frames
    one at a time, exactly as stored (no frame-rate conversion).
    """
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
               "-f", "rawvideo", "-pix_fmt", pix_fmt, "-fps_mode", "passthrough", "-"]
    shape = (height, width, channels) if channels > 1 else (height, width)
    frame_bytes = width * height * channels * np.dtype(dtype).itemsize
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            buffer = process.stdout.read(frame_bytes)
            if len(buffer) < frame_bytes:
                break
            yield np.frombuffer(buffer, dtype=dtype).reshape(shape)
    finally:
        process.stdout.close()
        process.kill()
        process.wait()


def check_alignment(aligner, align_check, tolerance_counts):
    """Input: DepthAligner, the dict loaded from align_check.npz, tolerance in
    depth counts. Aligns the stored raw depth frames over the full color
    image with numpy and compares against librealsense's rs.align output.
    Returns a dict of agreement statistics.
    """
    width = aligner.color_intrinsics["width"]
    height = aligner.color_intrinsics["height"]
    agree, coverage, median_abs = [], [], []
    for depth_raw, depth_reference in zip(align_check["depth_raw"], align_check["depth_aligned"]):
        depth_numpy = aligner.align_roi(depth_raw, (0, 0, width, height))
        both = (depth_numpy > 0) & (depth_reference > 0)
        difference = np.abs(depth_numpy[both].astype(np.int64) - depth_reference[both].astype(np.int64))
        agree.append(float(np.mean(difference <= tolerance_counts)) if difference.size else 0.0)
        coverage.append(float(both.sum() / max((depth_reference > 0).sum(), 1)))
        median_abs.append(float(np.median(difference)) if difference.size else None)
    return {
        "frames_checked": len(agree),
        "fraction_within_tolerance": agree,
        "coverage_of_rs_align_pixels": coverage,
        "median_abs_difference_counts": median_abs,
        "tolerance_counts": tolerance_counts,
        "passed": bool(agree) and min(agree) > 0.95 and min(coverage) > 0.95,
    }


def _run_slices(mask):
    """Returns a list of (start, end_exclusive) for the True runs in mask."""
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[0::2], changes[1::2]))


def fill_and_smooth(values, offline_config):
    """Input: (N, k) float array with NaN rows where there is no detection,
    config['offline']. Linearly interpolates gaps of at most
    max_gap_interp_frames that have detections on both sides, then applies a
    zero-phase Savitzky-Golay filter within every continuous run. Returns
    (smoothed (N, k) with NaN rows left where untracked, interpolated (N,) bool).
    """
    values = values.astype(np.float64).copy()
    present = ~np.isnan(values).any(axis=1)
    interpolated = np.zeros(len(values), dtype=bool)

    for start, end in _run_slices(~present):
        if start == 0 or end == len(values) or end - start > offline_config["max_gap_interp_frames"]:
            continue
        weights = (np.arange(start, end) - (start - 1)) / (end - (start - 1))
        values[start:end] = values[start - 1] + weights[:, None] * (values[end] - values[start - 1])
        interpolated[start:end] = True

    tracked = ~np.isnan(values).any(axis=1)
    window = offline_config["smoothing_window_frames"]
    polyorder = offline_config["smoothing_polyorder"]
    for start, end in _run_slices(tracked):
        run_window = min(window, (end - start) if (end - start) % 2 == 1 else (end - start) - 1)
        if run_window > polyorder:
            values[start:end] = savgol_filter(values[start:end], run_window, polyorder, axis=0)
    return values, interpolated


def _fill_within_runs(series, tracked):
    """Input: (N,) float series with NaNs, (N,) tracked mask. Fills NaNs
    inside each tracked run by linear interpolation from the run's valid
    samples (or the session median if a run has none). Returns a new array.
    """
    series = series.copy()
    session_median = np.nanmedian(series) if np.any(~np.isnan(series)) else np.nan
    for start, end in _run_slices(tracked):
        run = series[start:end]
        valid = ~np.isnan(run)
        if not valid.any():
            run[:] = session_median
        elif not valid.all():
            run[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), run[valid])
    return series


def _extract_roi(image, box):
    """Returns image[y0:y1, x0:x1] for a box that may extend past the image,
    zero-padded outside it.
    """
    x0, y0, x1, y1 = box
    out = np.zeros((y1 - y0, x1 - x0) + image.shape[2:], dtype=image.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, image.shape[1]), min(y1, image.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return out


def crop_frame(color_rgb, depth_raw, aligner, center, side_px, angle_rad, output_px):
    """Input: full color frame (RGB), raw unaligned depth frame, DepthAligner,
    float crop center, side length in color pixels, rotation (radians; 0 =
    axis-aligned), output size. Returns (color_crop_rgb, depth_crop_u16),
    both output_px x output_px. Color is anti-aliased before the
    (downsampling) warp; depth uses nearest-neighbour so holes stay holes.
    """
    scale = side_px / output_px  # source pixels per output pixel
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    half_extent = side_px / 2 * (abs(cos_a) + abs(sin_a)) + 3
    box = (int(math.floor(center[0] - half_extent)), int(math.floor(center[1] - half_extent)),
           int(math.ceil(center[0] + half_extent)) + 1, int(math.ceil(center[1] + half_extent)) + 1)

    # Output pixel p maps to source c + scale * R (p - h), h = the output center.
    h = (output_px - 1) / 2
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]]) * scale
    translation = np.array([center[0] - box[0], center[1] - box[1]]) - rotation @ np.array([h, h])
    matrix = np.hstack([rotation, translation[:, None]])

    color_roi = _extract_roi(color_rgb, box)
    if scale > 1:
        color_roi = cv2.GaussianBlur(color_roi, (0, 0), sigmaX=0.45 * scale)
    color_crop = cv2.warpAffine(color_roi, matrix, (output_px, output_px),
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

    depth_roi = aligner.align_roi(depth_raw, box)
    depth_crop = cv2.warpAffine(depth_roi, matrix, (output_px, output_px),
                                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return color_crop, depth_crop


def _center_depth_m(aligner, depth_raw, point, half=4):
    """Median valid aligned depth (m) in a (2*half+1)^2 box around point, or NaN."""
    cx, cy = int(round(point[0])), int(round(point[1]))
    roi = aligner.align_roi(depth_raw, (cx - half, cy - half, cx + half + 1, cy + half + 1))
    valid = roi[roi > 0]
    return float(np.median(valid)) * aligner.depth_scale_m if valid.size else np.nan


def _depth_to_bgr(depth_crop):
    valid = depth_crop > 0
    out = np.zeros(depth_crop.shape + (3,), dtype=np.uint8)
    if valid.any():
        low, high = np.percentile(depth_crop[valid], [2, 98])
        norm = np.clip((depth_crop.astype(np.float32) - low) / max(high - low, 1), 0, 1)
        out = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        out[~valid] = 0
    return out


def _write_qc(qc_dir, all_rows, saved_crops, full_frame_preview, stats):
    os.makedirs(qc_dir, exist_ok=True)
    with open(os.path.join(qc_dir, "frames_all.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with open(os.path.join(qc_dir, "qc.json"), "w") as f:
        json.dump(stats, f, indent=2)
    if saved_crops:
        color_row = np.hstack([cv2.cvtColor(color, cv2.COLOR_RGB2BGR) for color, _ in saved_crops])
        depth_row = np.hstack([_depth_to_bgr(depth) for _, depth in saved_crops])
        cv2.imwrite(os.path.join(qc_dir, "crop_strip.png"), np.vstack([color_row, depth_row]))
    if full_frame_preview is not None:
        cv2.imwrite(os.path.join(qc_dir, "full_frame_check.jpg"), full_frame_preview, [cv2.IMWRITE_JPEG_QUALITY, 90])


def _stability_stats(saved_rows, output_px):
    """Frame-to-frame crop-center statistics over consecutive saved frames,
    in output-crop pixels (so they are comparable across resolutions).
    """
    if len(saved_rows) < 3:
        return {}
    df = pd.DataFrame(saved_rows)
    consecutive = np.diff(df["raw_frame_index"].to_numpy()) == 1
    scale = df["crop_side_px"].to_numpy()[1:] / output_px
    step = np.hypot(np.diff(df["crop_cx"].to_numpy()), np.diff(df["crop_cy"].to_numpy())) / scale
    step = step[consecutive]
    raw_offset = (np.hypot(df["raw_anchor_x"] - df["crop_cx"], df["raw_anchor_y"] - df["crop_cy"])
                  / (df["crop_side_px"] / output_px)).dropna()
    return {
        "consecutive_pairs": int(consecutive.sum()),
        "center_step_output_px_median": float(np.median(step)) if step.size else None,
        "center_step_output_px_p95": float(np.percentile(step, 95)) if step.size else None,
        "center_step_output_px_max": float(np.max(step)) if step.size else None,
        "repeated_center_pct": float(np.mean(step == 0) * 100) if step.size else None,
        "raw_mediapipe_anchor_vs_smoothed_output_px_rms": float(np.sqrt(np.mean(raw_offset ** 2)))
        if len(raw_offset) else None,
    }


def process_session(raw_dir, config):
    """Input: raw session directory, full config dict (paths already resolved
    by load_config). Writes the processed session to output.root. Returns the
    processed session directory. Raises on failure (no PROCESSED_OK then).
    """
    started = time.time()
    raw_dir = os.path.abspath(raw_dir)
    session_name = os.path.basename(raw_dir.rstrip("/"))
    out_dir = os.path.join(config["output"]["root"], session_name)
    marker_path = os.path.join(out_dir, PROCESSED_MARKER)
    if os.path.exists(marker_path):
        os.remove(marker_path)

    with open(os.path.join(raw_dir, "meta_raw.json")) as f:
        meta_raw = json.load(f)
    camera_settings = meta_raw["camera"]
    calibration = camera_settings["stream_calibration"]
    depth_scale_m = camera_settings["depth"]["depth_scale_m"]
    color_intrinsics = calibration["color_intrinsics"]
    depth_intrinsics = calibration["depth_intrinsics"]
    width, height = color_intrinsics["width"], color_intrinsics["height"]
    frames_raw = pd.read_csv(os.path.join(raw_dir, "frames_raw.csv"))
    num_frames = len(frames_raw)
    print(f"[offline] {session_name}: {num_frames} raw frames at {width}x{height}")

    hand_config = config["hand_tracking"]
    offline_config = config["offline"]
    output_px = hand_config["crop_output_px"]
    tip_index = hand_config["fingertip_landmark_index"]
    dip_index = hand_config["anchor_dip_landmark_index"]
    aligner = cameraHandler.build_depth_aligner(calibration, depth_scale_m)

    align_check = dict(np.load(os.path.join(raw_dir, "align_check.npz")))
    align_stats = check_alignment(aligner, align_check, offline_config["align_check_tolerance_counts"])
    print(f"[offline] alignment check vs rs.align: within tolerance {align_stats['fraction_within_tolerance']}, "
          f"coverage {align_stats['coverage_of_rs_align_pixels']}")
    if not align_stats["passed"]:
        print("[offline] WARNING: numpy alignment disagrees with rs.align - see qc/qc.json")

    def color_frames():
        return read_video_frames(os.path.join(raw_dir, "color.mkv"), width, height, "rgb24", 3, np.uint8)

    def depth_frames():
        return read_video_frames(os.path.join(raw_dir, "depth.mkv"), depth_intrinsics["width"],
                                 depth_intrinsics["height"], "gray16le", 1, np.dtype("<u2"))

    # Pass 1: MediaPipe on every frame (VIDEO mode) + fingertip depth.
    tip = np.full((num_frames, 2), np.nan)
    dip = np.full((num_frames, 2), np.nan)
    tip_depth_m = np.full(num_frames, np.nan)
    landmarker = cameraHandler.build_hand_landmarker(config, running_mode="VIDEO")
    last_timestamp_ms = None
    decoded = 0
    try:
        for i, (color_rgb, depth_raw) in enumerate(zip(color_frames(), depth_frames())):
            if i >= num_frames:
                break
            decoded += 1
            last_timestamp_ms = cameraHandler.next_mediapipe_timestamp_ms(
                int(frames_raw["host_ts_ns"].iat[i]), last_timestamp_ms)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(color_rgb)), last_timestamp_ms)
            hand_landmarks = cameraHandler.select_tracked_hand(result, config)
            if hand_landmarks is not None:
                tip[i] = hand_landmarks[tip_index].x * width, hand_landmarks[tip_index].y * height
                dip[i] = hand_landmarks[dip_index].x * width, hand_landmarks[dip_index].y * height
                tip_depth_m[i] = _center_depth_m(aligner, depth_raw, tip[i])
            if (i + 1) % 300 == 0:
                print(f"[offline]   pass 1 (tracking): {i + 1}/{num_frames}")
    finally:
        landmarker.close()
    if decoded != num_frames:
        raise RuntimeError(f"decoded {decoded} frames but frames_raw.csv has {num_frames}")

    detected = ~np.isnan(tip[:, 0])
    smoothed, interpolated = fill_and_smooth(np.hstack([tip, dip]), offline_config)
    tip_s, dip_s = smoothed[:, :2], smoothed[:, 2:]
    tracked = ~np.isnan(tip_s[:, 0])
    depth_filled = _fill_within_runs(tip_depth_m, tracked)
    depth_s, _ = fill_and_smooth(depth_filled[:, None], offline_config)
    depth_s = depth_s[:, 0]

    axis = tip_s - dip_s
    axis_len = np.hypot(axis[:, 0], axis[:, 1])
    centers = tip_s - hand_config["anchor_alpha"] * axis
    raw_centers = tip - hand_config["anchor_alpha"] * (tip - dip)
    angles = np.arctan2(axis[:, 0], -axis[:, 1]) if hand_config["rotate_to_finger_axis"] \
        else np.zeros(num_frames)
    sides = hand_config["crop_side_mm"] / 1000.0 * color_intrinsics["fx"] / depth_s

    # Pass 2: crop, validate depth, save.
    session_writer = dataWriter.SessionWriter(
        out_dir, camera_settings, output_px, hand_config["tracked_hand_label"], tip_index,
        config["force_sensor"], {
            "tolerance_m": hand_config["depth_validation_tolerance_m"],
            "radius_m": hand_config["depth_validation_radius_m"],
            "minimum_valid_samples": hand_config["minimum_valid_depth_samples"],
        }, {
            "raw_session_dir": raw_dir,
            "crop_side_mm": hand_config["crop_side_mm"],
            "anchor_alpha": hand_config["anchor_alpha"],
            "anchor_dip_landmark_index": dip_index,
            "rotate_to_finger_axis": hand_config["rotate_to_finger_axis"],
            "smoothing": {k: offline_config[k] for k in
                          ("smoothing_window_frames", "smoothing_polyorder", "max_gap_interp_frames")},
            "mediapipe_mode": "VIDEO",
        })
    tracked_indices = np.flatnonzero(tracked & ~np.isnan(sides))
    preview_index = int(tracked_indices[len(tracked_indices) // 2]) if tracked_indices.size else None
    full_frame_preview = None
    all_rows, saved_rows, saved_crops_by_index = [], [], {}
    for i, (color_rgb, depth_raw) in enumerate(zip(color_frames(), depth_frames())):
        if i >= num_frames:
            break
        row = frames_raw.iloc[i]
        usable = bool(tracked[i]) and not np.isnan(sides[i]) and axis_len[i] > 0
        depth_info = cameraHandler.empty_depth_info()
        saved = False
        if usable:
            depth_info = cameraHandler.validate_fingertip_depth_unaligned(aligner, depth_raw, tip_s[i], config)
            if depth_info["valid"] or not offline_config["save_only_valid_frames"]:
                color_crop, depth_crop = crop_frame(color_rgb, depth_raw, aligner, centers[i], sides[i],
                                                    angles[i], output_px)
                box = (int(round(centers[i][0] - sides[i] / 2)), int(round(centers[i][1] - sides[i] / 2)))
                geometry = {
                    "raw_frame_index": i,
                    "crop_cx": round(float(centers[i][0]), 3),
                    "crop_cy": round(float(centers[i][1]), 3),
                    "crop_side_px": round(float(sides[i]), 3),
                    "crop_angle_deg": round(math.degrees(angles[i]), 3),
                    "fingertip_interpolated": int(interpolated[i]),
                }
                session_writer.add_frame(session_writer.num_frames(), int(row["host_ts_ns"]),
                                         row["color_device_ts_ms"], row["depth_device_ts_ms"],
                                         bool(detected[i]), depth_info, box, color_crop, depth_crop, geometry)
                saved_rows.append({**geometry, "raw_anchor_x": raw_centers[i][0], "raw_anchor_y": raw_centers[i][1]})
                saved_crops_by_index[i] = (color_crop, depth_crop)  # same arrays the writer holds, no copy
                saved = True
        if i == preview_index:
            preview = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            half = sides[i] / 2
            corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
            rot = np.array([[math.cos(angles[i]), -math.sin(angles[i])], [math.sin(angles[i]), math.cos(angles[i])]])
            polygon = (corners @ rot.T + centers[i]).astype(np.int32)
            cv2.polylines(preview, [polygon], True, (0, 255, 0), 2)
            cv2.circle(preview, tuple(np.round(tip_s[i]).astype(int)), 4, (0, 0, 255), -1)
            cv2.circle(preview, tuple(np.round(dip_s[i]).astype(int)), 4, (255, 0, 0), -1)
            full_frame_preview = cv2.resize(preview, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
        all_rows.append({
            "raw_frame_index": i,
            "host_ts_ns": int(row["host_ts_ns"]),
            "detected": int(detected[i]),
            "interpolated": int(interpolated[i]),
            "tracked": int(usable),
            "depth_valid": int(depth_info["valid"]),
            "saved": int(saved),
            "tip_x": tip_s[i][0], "tip_y": tip_s[i][1],
            "crop_cx": centers[i][0], "crop_cy": centers[i][1],
            "crop_side_px": sides[i], "tip_depth_m": depth_s[i],
        })
        if (i + 1) % 300 == 0:
            print(f"[offline]   pass 2 (cropping): {i + 1}/{num_frames}")

    force_raw = pd.read_csv(os.path.join(raw_dir, "force_raw.csv"))
    session_writer.add_force_samples(zip(force_raw["host_ts_ns"].astype(np.int64).tolist(),
                                         force_raw["fz_N"].tolist()))
    session_writer.flush()

    # QC strip: the middle of the longest run of consecutive saved frames.
    strip_length = offline_config["qc_strip_frames"]
    saved_mask = np.zeros(num_frames, dtype=bool)
    saved_mask[[r["raw_frame_index"] for r in saved_rows]] = True
    runs = sorted(_run_slices(saved_mask), key=lambda run: run[1] - run[0], reverse=True)
    strip = []
    if runs:
        start, end = runs[0]
        first = max(start, (start + end) // 2 - strip_length // 2)
        strip = [saved_crops_by_index[i] for i in range(first, min(first + strip_length, end))
                 if i in saved_crops_by_index]

    frame_dt_ms = np.diff(frames_raw["host_ts_ns"].to_numpy()) / 1e6
    stats = {
        "session": session_name,
        "raw_frames": num_frames,
        "saved_frames": len(saved_rows),
        "detected_pct": float(detected.mean() * 100) if num_frames else 0.0,
        "interpolated_frames": int(interpolated.sum()),
        "tracked_pct": float(tracked.mean() * 100) if num_frames else 0.0,
        "camera_dropped_frames": meta_raw.get("camera_dropped_frames"),
        "queue_dropped_frames": meta_raw.get("queue_dropped_frames"),
        "raw_frame_interval_ms_median": float(np.median(frame_dt_ms)) if frame_dt_ms.size else None,
        "raw_frame_gaps_over_50ms": int(np.sum(frame_dt_ms > 50)),
        "raw_video_mb_per_s": meta_raw.get("video_mb_per_s"),
        "median_crop_side_px": float(np.nanmedian(sides)) if tracked.any() else None,
        "median_tip_depth_m": float(np.nanmedian(depth_s)) if tracked.any() else None,
        "stability": _stability_stats(saved_rows, output_px),
        "alignment_check": align_stats,
        "processing_time_s": round(time.time() - started, 1),
    }
    _write_qc(os.path.join(out_dir, "qc"), all_rows, strip, full_frame_preview, stats)

    with open(marker_path, "w") as f:
        json.dump({"raw_session_dir": raw_dir, "saved_frames": len(saved_rows),
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"[offline] {session_name}: saved {len(saved_rows)}/{num_frames} frames to {out_dir} "
          f"in {stats['processing_time_s']}s; stability {stats['stability']}")

    if config["output"]["delete_raw_after_processing"]:
        shutil.rmtree(raw_dir)
        print(f"[offline] deleted raw session {raw_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_session_dirs", nargs="*", help="Raw/session_* folders to process")
    parser.add_argument("--all", action="store_true", help="process every raw session under output.raw_root")
    parser.add_argument("--force", action="store_true", help="with --all: also re-process finished sessions")
    args = parser.parse_args()

    config = load_config()
    raw_dirs = list(args.raw_session_dirs)
    if args.all:
        for raw_dir in sorted(glob.glob(os.path.join(config["output"]["raw_root"], "session_*"))):
            done = os.path.exists(os.path.join(config["output"]["root"], os.path.basename(raw_dir), PROCESSED_MARKER))
            if args.force or not done:
                raw_dirs.append(raw_dir)
    if not raw_dirs:
        parser.error("no raw sessions to process (pass folders or --all)")

    failures = 0
    for raw_dir in raw_dirs:
        try:
            process_session(raw_dir, config)
        except Exception as e:  # keep going with the other sessions
            failures += 1
            print(f"[offline] FAILED {raw_dir}: {e!r}")
            traceback.print_exc()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
