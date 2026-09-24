"""RealSense D435 camera pipeline, live fingertip tracking and on-screen
preview/overlay for the data-collection recorder. All tunable values come
from the config dict loaded from config.yaml - nothing here is hardcoded.

The live side only captures: color arrives as raw YUYV (what the camera
sends over USB) and depth stays unaligned, both handed to
dataWriter.RawSessionWriter untouched. Hand tracking here runs on a
downscaled preview frame and only drives the operator overlay and the force
auto-zero; the dataset crops are made offline by offlineExtract.py.
"""

import threading
import time
import math

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

import depthAlign

_hand_result_lock = threading.Lock()
_latest_hand_result = None

# Nearest fingertip depth assumed when sizing the depth-validation ROI; the
# D435 cannot measure much closer than this anyway.
_VALIDATION_ROI_MIN_DEPTH_M = 0.15


def _on_hand_result(result, output_image, timestamp_ms):
    global _latest_hand_result
    with _hand_result_lock:
        _latest_hand_result = result


def _apply_color_sensor_options(color_sensor, color_config):
    if color_sensor.supports(rs.option.enable_auto_exposure):
        color_sensor.set_option(rs.option.enable_auto_exposure, 1 if color_config["auto_exposure"] else 0)
    if not color_config["auto_exposure"]:
        if color_sensor.supports(rs.option.exposure):
            color_sensor.set_option(rs.option.exposure, color_config["exposure"])
        if color_sensor.supports(rs.option.gain):
            color_sensor.set_option(rs.option.gain, color_config["gain"])

    if color_sensor.supports(rs.option.enable_auto_white_balance):
        color_sensor.set_option(rs.option.enable_auto_white_balance, 1 if color_config["auto_white_balance"] else 0)
    if not color_config["auto_white_balance"] and color_sensor.supports(rs.option.white_balance):
        color_sensor.set_option(rs.option.white_balance, color_config["white_balance"])

    if color_sensor.supports(rs.option.brightness):
        color_sensor.set_option(rs.option.brightness, color_config["brightness"])
    if color_sensor.supports(rs.option.contrast):
        color_sensor.set_option(rs.option.contrast, color_config["contrast"])


def _apply_depth_sensor_options(depth_sensor, depth_config):
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1 if depth_config["emitter_enabled"] else 0)
    if depth_config["emitter_enabled"] and depth_sensor.supports(rs.option.laser_power):
        depth_sensor.set_option(rs.option.laser_power, depth_config["laser_power"])

    if depth_sensor.supports(rs.option.depth_units):
        depth_sensor.set_option(rs.option.depth_units, depth_config["depth_units_m"])

    if depth_sensor.supports(rs.option.visual_preset):
        preset = getattr(rs.rs400_visual_preset, depth_config["visual_preset"])
        depth_sensor.set_option(rs.option.visual_preset, int(preset))


def start_camera_pipeline(config):
    """Input: full config dict (uses config['camera']).
    Starts the RealSense pipeline with a raw YUYV color stream and an
    unaligned depth stream, and applies the sensor options. Returns
    (pipeline, stream_calibration, depth_scale_m), where stream_calibration
    is a JSON-serializable dict with both intrinsics and the depth->color
    extrinsics (everything offlineExtract.py needs to align depth later).
    """
    color_config = config["camera"]["color"]
    depth_config = config["camera"]["depth"]

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, color_config["width"], color_config["height"],
                             rs.format.yuyv, color_config["framerate"])
    rs_config.enable_stream(rs.stream.depth, depth_config["width"], depth_config["height"],
                             rs.format.z16, depth_config["framerate"])

    profile = pipeline.start(rs_config)
    device = profile.get_device()

    color_sensor = device.first_color_sensor()
    _apply_color_sensor_options(color_sensor, color_config)

    depth_sensor = device.first_depth_sensor()
    _apply_depth_sensor_options(depth_sensor, depth_config)

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    stream_calibration = {
        "color_intrinsics": depthAlign.intrinsics_to_dict(color_profile.get_intrinsics()),
        "depth_intrinsics": depthAlign.intrinsics_to_dict(depth_profile.get_intrinsics()),
        "depth_to_color_extrinsics": depthAlign.extrinsics_to_dict(depth_profile.get_extrinsics_to(color_profile)),
    }
    depth_scale_m = depth_sensor.get_depth_scale()
    return pipeline, stream_calibration, depth_scale_m


def build_depth_aligner(stream_calibration, depth_scale_m):
    """Input: stream_calibration and depth_scale_m from start_camera_pipeline()
    (or from a session's meta_raw.json). Returns a depthAlign.DepthAligner.
    """
    return depthAlign.DepthAligner(stream_calibration["depth_intrinsics"],
                                   stream_calibration["color_intrinsics"],
                                   stream_calibration["depth_to_color_extrinsics"], depth_scale_m)


def capture_align_check(pipeline, num_framesets=3, warmup_framesets=15):
    """Input: a started pipeline. Grabs a few framesets and aligns them with
    librealsense's own rs.align, as ground truth for offlineExtract.py's
    numpy alignment. Done once at startup (a full-frame rs.align at 1080p is
    too slow to run while recording). Returns a dict of stacked arrays:
    'depth_raw' (N,h,w) and 'depth_aligned' (N,H,W), both uint16.
    """
    align = rs.align(rs.stream.color)
    for _ in range(warmup_framesets):
        pipeline.wait_for_frames()
    depth_raw, depth_aligned = [], []
    for _ in range(num_framesets):
        frames = pipeline.wait_for_frames()
        depth_raw.append(np.asanyarray(frames.get_depth_frame().get_data()).copy())
        aligned = align.process(frames)
        depth_aligned.append(np.asanyarray(aligned.get_depth_frame().get_data()).copy())
    return {"depth_raw": np.stack(depth_raw), "depth_aligned": np.stack(depth_aligned)}


def get_frame(pipeline):
    """Input: pipeline from start_camera_pipeline().
    Waits for the next frameset. Returns a dict with 'color_yuyv' (H,W,2
    uint8) and 'depth_u16' (h,w, unaligned) - both copied out of the driver
    buffer so they can be queued for writing - plus 'color_frame_number',
    'color_device_ts_ms', 'depth_device_ts_ms' and 'host_ts_ns'; or None if
    either frame is missing.
    """
    frames = pipeline.wait_for_frames()
    host_ts_ns = time.perf_counter_ns()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not color_frame or not depth_frame:
        return None

    color_packed = np.asanyarray(color_frame.get_data())  # (H, W) uint16, one Y+U/V pair per pixel
    return {
        "color_yuyv": color_packed.view(np.uint8).reshape(color_packed.shape[0], color_packed.shape[1], 2).copy(),
        "depth_u16": np.asanyarray(depth_frame.get_data()).copy(),
        "color_frame_number": color_frame.get_frame_number(),
        "color_device_ts_ms": color_frame.get_timestamp(),
        "depth_device_ts_ms": depth_frame.get_timestamp(),
        "host_ts_ns": host_ts_ns,
    }


def yuyv_to_preview_bgr(color_yuyv, config):
    """Input: full-resolution YUYV frame, full config dict (uses
    config['hand_tracking']['live_preview_width']). Returns (full_bgr,
    preview_bgr, scale) where scale = preview width / full width.
    """
    full_bgr = cv2.cvtColor(color_yuyv, cv2.COLOR_YUV2BGR_YUYV)
    preview_width = config["hand_tracking"]["live_preview_width"]
    scale = preview_width / full_bgr.shape[1]
    preview_height = int(round(full_bgr.shape[0] * scale))
    preview_bgr = cv2.resize(full_bgr, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
    return full_bgr, preview_bgr, scale


def build_hand_landmarker(config, running_mode="LIVE_STREAM"):
    """Input: full config dict (uses config['hand_tracking']['model_path']),
    and the MediaPipe running mode: "LIVE_STREAM" (live preview, async with a
    result callback) or "VIDEO" (offline, synchronous, one result per frame).
    Returns a MediaPipe HandLandmarker.
    """
    model_path = config["hand_tracking"]["model_path"]
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    extra = {"result_callback": _on_hand_result} if running_mode == "LIVE_STREAM" else {}
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=getattr(VisionRunningMode, running_mode),
        num_hands=2,
        **extra,
    )
    return HandLandmarker.create_from_options(options)


def next_mediapipe_timestamp_ms(host_ts_ns, last_timestamp_ms):
    """Input: current host timestamp in ns, previous timestamp passed to
    MediaPipe (or None). Returns the next timestamp in ms, bumped by 1ms if
    needed since MediaPipe LIVE_STREAM/VIDEO require strictly increasing values.
    """
    timestamp_ms = host_ts_ns // 1_000_000
    if last_timestamp_ms is not None and timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    return timestamp_ms


def select_tracked_hand(result, config):
    """Input: a MediaPipe HandLandmarkerResult, full config dict (uses
    config['hand_tracking']['tracked_hand_label']). Returns the tracked
    hand's list of 21 normalized landmarks, or None if it isn't in the result.
    """
    if not result or not result.hand_landmarks or not result.handedness:
        return None
    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        if handedness[0].category_name == config["hand_tracking"]["tracked_hand_label"]:
            return hand_landmarks
    return None


def track_fingertip_live(preview_bgr, landmarker, timestamp_ms, submit, full_size, config):
    """Input: downscaled preview frame (BGR), LIVE_STREAM landmarker from
    build_hand_landmarker(), timestamp from next_mediapipe_timestamp_ms(),
    whether to submit this frame to MediaPipe (the preview only runs it every
    live_mediapipe_every_n frames), the full-resolution (width, height), and
    the full config dict (uses config['hand_tracking']). Returns the full-
    resolution float pixel (x, y) of the tracked fingertip landmark from the
    most recently completed detection, or None. Only used for the preview and
    force auto-zero; the dataset crops are re-tracked offline per frame.
    """
    if submit:
        frame_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
        landmarker.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb), timestamp_ms)

    with _hand_result_lock:
        result = _latest_hand_result

    hand_landmarks = select_tracked_hand(result, config)
    if hand_landmarks is None:
        return None
    landmark = hand_landmarks[config["hand_tracking"]["fingertip_landmark_index"]]
    return landmark.x * full_size[0], landmark.y * full_size[1]


def crop_side_px(depth_m, fx, config):
    """Input: fingertip depth in meters, color focal length fx in pixels,
    full config dict (uses config['hand_tracking']['crop_side_mm']). Returns
    the crop side length in color pixels that covers crop_side_mm at that depth.
    """
    return config["hand_tracking"]["crop_side_mm"] / 1000.0 * fx / depth_m


def compute_crop_box(center_x, center_y, size, frame_width, frame_height):
    """Input: crop center pixel, square crop side in pixels, frame size.
    Returns integer (x0, y0, x1, y1) for a size x size box centered on
    (center_x, center_y), clamped so it stays fully inside the frame.
    """
    size = int(round(min(size, frame_width, frame_height)))
    half = size // 2
    x0 = int(round(center_x)) - half
    y0 = int(round(center_y)) - half
    x0 = max(min(x0, frame_width - size), 0)
    y0 = max(min(y0, frame_height - size), 0)
    return x0, y0, x0 + size, y0 + size


def validate_fingertip_depth(depth_u16, fingertip_px, depth_scale_m, color_intrinsics, config):
    """Check whether the fingertip depth agrees with an 8-point physical ring.
    depth_u16 must be aligned to the color image (full frame, or a region of
    it with fingertip_px given relative to that region), and
    color_intrinsics is a dict as stored by depthAlign.intrinsics_to_dict.
    Returns validation status and depth measurements for frame metadata and
    the overlay. A zero depth value is treated as an invalid measurement.
    """
    hand_config = config["hand_tracking"]
    tolerance_m = hand_config["depth_validation_tolerance_m"]
    radius_m = hand_config["depth_validation_radius_m"]
    minimum_valid_samples = hand_config["minimum_valid_depth_samples"]
    center_x, center_y = int(round(fingertip_px[0])), int(round(fingertip_px[1]))
    height, width = depth_u16.shape[:2]

    result = empty_depth_info()

    if not (0 <= center_x < width and 0 <= center_y < height):
        return result

    center_depth_raw = int(depth_u16[center_y, center_x])
    if center_depth_raw <= 0:
        return result

    center_depth_m = center_depth_raw * depth_scale_m
    sample_points = []
    reference_depths_raw = []
    for point_index in range(8):
        angle = point_index * math.pi / 4
        offset_x = int(round(radius_m * color_intrinsics["fx"] / center_depth_m * math.cos(angle)))
        offset_y = int(round(radius_m * color_intrinsics["fy"] / center_depth_m * math.sin(angle)))
        reference_x = center_x + offset_x
        reference_y = center_y + offset_y
        if not (0 <= reference_x < width and 0 <= reference_y < height):
            sample_points.append({"x": reference_x, "y": reference_y, "valid": False})
            continue

        reference_depth_raw = int(depth_u16[reference_y, reference_x])
        sample_valid = reference_depth_raw > 0 and abs(
            reference_depth_raw * depth_scale_m - center_depth_m
        ) <= tolerance_m
        sample_points.append({"x": reference_x, "y": reference_y, "valid": sample_valid})
        if reference_depth_raw > 0:
            reference_depths_raw.append(reference_depth_raw)

    if not reference_depths_raw:
        result["sample_points"] = sample_points
        return result

    reference_depths_m = [depth * depth_scale_m for depth in reference_depths_raw]
    neighborhood_min_m = min(reference_depths_m)
    neighborhood_max_m = max(reference_depths_m)
    valid_sample_count = sum(sample_point["valid"] for sample_point in sample_points)
    result.update({
        "valid": valid_sample_count >= minimum_valid_samples,
        "center_depth_m": center_depth_m,
        "neighborhood_min_depth_m": neighborhood_min_m,
        "neighborhood_max_depth_m": neighborhood_max_m,
        "reference_count": len(reference_depths_m),
        "valid_sample_count": valid_sample_count,
        "sample_points": sample_points,
    })
    return result


def empty_depth_info():
    """Returns the depth-validation result for a frame without a fingertip."""
    return {
        "valid": False,
        "center_depth_m": None,
        "neighborhood_min_depth_m": None,
        "neighborhood_max_depth_m": None,
        "reference_count": 0,
        "valid_sample_count": 0,
        "sample_points": [],
    }


def validation_roi_box(fingertip_px, color_intrinsics, config):
    """Input: fingertip pixel, color intrinsics dict, full config dict.
    Returns the color-image box (x0, y0, x1, y1) that contains the whole
    depth-validation ring for any fingertip depth the camera can measure.
    """
    radius_m = config["hand_tracking"]["depth_validation_radius_m"]
    half = int(math.ceil(radius_m * max(color_intrinsics["fx"], color_intrinsics["fy"])
                         / _VALIDATION_ROI_MIN_DEPTH_M)) + 2
    cx, cy = int(round(fingertip_px[0])), int(round(fingertip_px[1]))
    return cx - half, cy - half, cx + half + 1, cy + half + 1


def validate_fingertip_depth_unaligned(aligner, depth_u16, fingertip_px, config):
    """Input: DepthAligner from build_depth_aligner(), raw unaligned depth
    frame, fingertip pixel in the color image, full config dict. Aligns only
    the validation ROI and runs validate_fingertip_depth() on it. Returns the
    same dict, with sample_points in full color-image coordinates.
    """
    box = validation_roi_box(fingertip_px, aligner.color_intrinsics, config)
    roi_depth = aligner.align_roi(depth_u16, box)
    info = validate_fingertip_depth(roi_depth, (fingertip_px[0] - box[0], fingertip_px[1] - box[1]),
                                    aligner.depth_scale_m, aligner.color_intrinsics, config)
    for sample_point in info["sample_points"]:
        sample_point["x"] += box[0]
        sample_point["y"] += box[1]
    return info


def build_camera_settings_dict(config, stream_calibration, depth_scale_m):
    """Input: full config dict, stream_calibration and depth_scale_m from
    start_camera_pipeline(). Returns a dict combining the camera config with
    values read back from the device, for session metadata (meta.json /
    README.md).
    """
    color_config = config["camera"]["color"]
    depth_config = config["camera"]["depth"]
    color_intrinsics = stream_calibration["color_intrinsics"]
    return {
        "color": {
            **color_config,
            "intrinsics_fx_fy_ppx_ppy": [color_intrinsics["fx"], color_intrinsics["fy"],
                                          color_intrinsics["ppx"], color_intrinsics["ppy"]],
        },
        "depth": {
            **depth_config,
            "depth_scale_m": depth_scale_m,
        },
        "stream_calibration": stream_calibration,
    }


def draw_overlay(frame, scale, box, fingertip_detected, fingertip_depth_info,
                  is_recording, recording_stats, latest_force_fz, force_idle_s, processing_status):
    """Input: preview frame to draw on (mutated in place), preview scale
    (preview px / full-resolution px; box and sample points are given in
    full-resolution pixels), current live crop box, tracking/recording/force
    status, recording_stats dict from RawSessionWriter.stats() (or None), and
    a one-line offline-processing status string. Returns nothing.
    """
    def to_preview(x, y):
        return int(round(x * scale)), int(round(y * scale))

    if box is not None:
        cv2.rectangle(frame, to_preview(box[0], box[1]), to_preview(box[2], box[3]),
                      (0, 255, 0) if fingertip_detected else (0, 0, 255), 2)

    for sample_point in fingertip_depth_info.get("sample_points", []):
        point = to_preview(sample_point["x"], sample_point["y"])
        color = (0, 255, 0) if sample_point["valid"] else (0, 0, 255)
        if 0 <= point[0] < frame.shape[1] and 0 <= point[1] < frame.shape[0]:
            cv2.circle(frame, point, 3, color, -1)

    y = 20
    finger_color = (0, 255, 0) if fingertip_detected else (0, 0, 255)
    cv2.putText(frame, f"Fingertip: {'tracked' if fingertip_detected else 'lost'}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, finger_color, 1, cv2.LINE_AA)
    y += 20

    depth_valid = fingertip_depth_info["valid"]
    depth_text = "valid" if depth_valid else "invalid"
    center_depth_m = fingertip_depth_info["center_depth_m"]
    if center_depth_m is None:
        depth_line = f"Fingertip depth: {depth_text}"
    else:
        depth_line = f"Fingertip depth: {depth_text} ({center_depth_m * 100:.1f} cm)"
    cv2.putText(frame, depth_line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0) if depth_valid else (0, 0, 255), 1, cv2.LINE_AA)
    y += 20

    cv2.putText(frame, f"Force: Fz={latest_force_fz:+.2f} N (zeroed)",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    y += 20

    if force_idle_s is None:
        idle_line, idle_color = "Force sensor: no packet received yet", (0, 0, 255)
    elif force_idle_s > 2.0:
        idle_line, idle_color = f"Force sensor: IDLE for {force_idle_s:.1f}s (no packets)", (0, 0, 255)
    else:
        idle_line, idle_color = f"Force sensor: alive ({force_idle_s:.2f}s since last packet)", (0, 255, 0)
    cv2.putText(frame, idle_line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, idle_color, 1, cv2.LINE_AA)
    y += 20

    if processing_status:
        cv2.putText(frame, processing_status, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1, cv2.LINE_AA)

    if is_recording and recording_stats is not None:
        cv2.circle(frame, (frame.shape[1] - 20, 20), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (frame.shape[1] - 65, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        lines = [
            f"frames={recording_stats['num_frames']} force={recording_stats['num_force_samples']}",
            f"fps={recording_stats['fps']:.1f}",
            f"camera drops={recording_stats['camera_dropped']} queue drops={recording_stats['queue_dropped']}",
        ]
        for line_index, line in enumerate(lines):
            drops = line_index == 2 and (recording_stats["camera_dropped"] or recording_stats["queue_dropped"])
            cv2.putText(frame, line, (frame.shape[1] - 300, 46 + 20 * line_index), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255) if drops or line_index < 2 else (0, 170, 0), 1, cv2.LINE_AA)

    cv2.putText(frame, "r: record  q: quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_force_histogram(bin_percentages, bin_width_n, is_recording):
    """Input: list of per-bin percentages (0-100) from ForceHistogram.percentages(),
    the bin width in N, and whether a session is currently recording. Returns a BGR
    canvas with one bar per bin (labelled with its |Fz| range and live percentage),
    plus a dashed line at the balanced-dataset target (100 / num_bins %), so the
    operator can see at a glance which force ranges still need more presses.
    """
    num_bins = len(bin_percentages)
    canvas_width, canvas_height = 640, 300
    margin_left, margin_right, margin_top, margin_bottom = 10, 10, 30, 50
    plot_width = canvas_width - margin_left - margin_right
    plot_height = canvas_height - margin_top - margin_bottom

    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    title = "Force distribution (recording)" if is_recording else "Force distribution (last take)"
    cv2.putText(canvas, title, (margin_left, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    target_pct = 100.0 / num_bins
    max_scale = max(target_pct * 2, max(bin_percentages, default=0.0) * 1.1, 1e-6)

    target_y = margin_top + int(plot_height * (1 - target_pct / max_scale))
    for x in range(margin_left, margin_left + plot_width, 10):
        cv2.line(canvas, (x, target_y), (x + 5, target_y), (160, 160, 160), 1)

    bar_area_width = plot_width / num_bins
    bar_width = int(bar_area_width * 0.7)
    for bin_index, pct in enumerate(bin_percentages):
        bar_height = int(plot_height * min(pct / max_scale, 1.0))
        x0 = int(margin_left + bin_index * bar_area_width + (bar_area_width - bar_width) / 2)
        y1 = margin_top + plot_height
        y0 = y1 - bar_height
        color = (0, 170, 0) if pct >= target_pct else (0, 120, 255)
        cv2.rectangle(canvas, (x0, y0), (x0 + bar_width, y1), color, -1)

        pct_label = f"{pct:.0f}%"
        label_size = cv2.getTextSize(pct_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(canvas, pct_label, (x0 + (bar_width - label_size[0]) // 2, max(y0 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        low_n, high_n = bin_index * bin_width_n, (bin_index + 1) * bin_width_n
        range_label = f"{low_n:.0f}-{high_n:.0f}"
        range_size = cv2.getTextSize(range_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        cv2.putText(canvas, range_label, (x0 + (bar_width - range_size[0]) // 2, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(canvas, "|Fz| (N)", (canvas_width - 70, canvas_height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas



def show_previews(main_frame, color_crop_bgr, depth_crop, force_histogram_canvas, config):
    """Input: main (downscaled) display frame, optional live color/depth
    crops (None if no fingertip), the force-distribution canvas from
    draw_force_histogram(), full config dict (uses config['hand_tracking']).
    Shows the main RealSense window, the force distribution chart, plus
    whichever crop preview windows are enabled. Returns nothing.
    """
    hand_config = config["hand_tracking"]
    cv2.imshow("RealSense", main_frame)
    cv2.imshow("Force Distribution", force_histogram_canvas)
    if hand_config["show_tracked_crop"] and color_crop_bgr is not None:
        cv2.imshow("Fingertip Crop (color)", color_crop_bgr)
    if hand_config["show_depth_preview"] and depth_crop is not None:
        depth_norm = cv2.normalize(depth_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        cv2.imshow("Fingertip Crop (depth)", depth_color)


def stop(pipeline, landmarker):
    """Input: pipeline and landmarker to release. Stops the camera pipeline,
    closes the hand landmarker, and destroys all OpenCV preview windows.
    Returns nothing.
    """
    landmarker.close()
    pipeline.stop()
    cv2.destroyAllWindows()
