"""RealSense D435 camera pipeline, MediaPipe fingertip tracking, cropping,
and on-screen preview/overlay for the data-collection recorder. All tunable
values come from the config dict loaded from config.yaml - nothing here is
hardcoded.
"""

import threading
import time
import math

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

_hand_result_lock = threading.Lock()
_latest_hand_result = None


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
    Starts the RealSense pipeline with color+depth streams and applies the
    sensor options. Returns (pipeline, align, color_intrinsics, depth_scale_m).
    """
    color_config = config["camera"]["color"]
    depth_config = config["camera"]["depth"]

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, color_config["width"], color_config["height"],
                             rs.format.bgr8, color_config["framerate"])
    rs_config.enable_stream(rs.stream.depth, depth_config["width"], depth_config["height"],
                             rs.format.z16, depth_config["framerate"])

    profile = pipeline.start(rs_config)
    device = profile.get_device()

    color_sensor = device.first_color_sensor()
    _apply_color_sensor_options(color_sensor, color_config)

    depth_sensor = device.first_depth_sensor()
    _apply_depth_sensor_options(depth_sensor, depth_config)

    align = rs.align(rs.stream.color)
    color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_scale_m = depth_sensor.get_depth_scale()

    return pipeline, align, color_intrinsics, depth_scale_m


def get_frame(pipeline, align):
    """Input: pipeline and align from start_camera_pipeline().
    Waits for the next frameset and aligns depth to color. Returns
    (color_bgr, depth_u16, color_ts_ms, depth_ts_ms, host_ts_ns); the first
    four are None if either the color or depth frame is missing.
    """
    frames = pipeline.wait_for_frames()
    host_ts_ns = time.perf_counter_ns()

    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not color_frame or not depth_frame:
        return None, None, None, None, host_ts_ns

    color_bgr = np.asanyarray(color_frame.get_data())
    depth_u16 = np.asanyarray(depth_frame.get_data())
    return color_bgr, depth_u16, color_frame.get_timestamp(), depth_frame.get_timestamp(), host_ts_ns


def build_hand_landmarker(config):
    """Input: full config dict (uses config['hand_tracking']['model_path']).
    Returns a MediaPipe HandLandmarker running in LIVE_STREAM mode.
    """
    model_path = config["hand_tracking"]["model_path"]
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=_on_hand_result,
        num_hands=2,
    )
    return HandLandmarker.create_from_options(options)


def next_mediapipe_timestamp_ms(host_ts_ns, last_timestamp_ms):
    """Input: current host timestamp in ns, previous timestamp passed to
    MediaPipe (or None). Returns the next timestamp in ms, bumped by 1ms if
    needed since MediaPipe LIVE_STREAM requires strictly increasing values.
    """
    timestamp_ms = host_ts_ns // 1_000_000
    if last_timestamp_ms is not None and timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    return timestamp_ms


def track_fingertip(color_bgr, landmarker, timestamp_ms, config):
    """Input: current color frame (BGR), landmarker from build_hand_landmarker(),
    timestamp from next_mediapipe_timestamp_ms(), full config dict (uses
    config['hand_tracking']). Runs (async) hand detection and returns the
    pixel (x, y) of the tracked hand's fingertip landmark from the most
    recently completed detection, or None if it isn't currently visible.
    """
    hand_config = config["hand_tracking"]
    frame_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    landmarker.detect_async(mp_image, timestamp_ms)

    with _hand_result_lock:
        result = _latest_hand_result

    if not result or not result.hand_landmarks or not result.handedness:
        return None

    h, w, _ = color_bgr.shape
    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        if handedness[0].category_name == hand_config["tracked_hand_label"]:
            landmark = hand_landmarks[hand_config["fingertip_landmark_index"]]
            return int(landmark.x * w), int(landmark.y * h)

    return None


def compute_crop_box(center_x, center_y, frame_width, frame_height, config):
    """Input: crop center pixel, frame size, full config dict (uses
    config['hand_tracking']['crop_size']). Returns (x0, y0, x1, y1) for a
    crop_size x crop_size box centered on (center_x, center_y), clamped so
    it stays fully inside the frame.
    """
    size = config["hand_tracking"]["crop_size"]
    half = size // 2
    x0 = min(max(center_x - half, 0), frame_width - size)
    y0 = min(max(center_y - half, 0), frame_height - size)
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    return x0, y0, x0 + size, y0 + size


def crop_color_and_depth(color_bgr, depth_u16, box):
    """Input: full color/depth frames and a crop box from compute_crop_box().
    Returns (color_crop_bgr, color_crop_rgb, depth_crop).
    """
    x0, y0, x1, y1 = box
    color_crop_bgr = color_bgr[y0:y1, x0:x1].copy()
    color_crop_rgb = cv2.cvtColor(color_crop_bgr, cv2.COLOR_BGR2RGB)
    depth_crop = depth_u16[y0:y1, x0:x1].copy()
    return color_crop_bgr, color_crop_rgb, depth_crop


def validate_fingertip_depth(depth_u16, fingertip_px, depth_scale_m, color_intrinsics, config):
    """Check whether the fingertip depth agrees with an 8-point physical ring.
    Returns validation status and depth measurements for frame metadata and the
    live overlay. A zero depth value is treated as an invalid measurement.
    """
    hand_config = config["hand_tracking"]
    tolerance_m = hand_config["depth_validation_tolerance_m"]
    radius_m = hand_config["depth_validation_radius_m"]
    minimum_valid_samples = hand_config["minimum_valid_depth_samples"]
    center_x, center_y = fingertip_px
    height, width = depth_u16.shape[:2]

    result = {
        "valid": False,
        "center_depth_m": None,
        "neighborhood_min_depth_m": None,
        "neighborhood_max_depth_m": None,
        "reference_count": 0,
        "valid_sample_count": 0,
        "sample_points": [],
    }

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
        offset_x = int(round(radius_m * color_intrinsics.fx / center_depth_m * math.cos(angle)))
        offset_y = int(round(radius_m * color_intrinsics.fy / center_depth_m * math.sin(angle)))
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


def build_camera_settings_dict(config, color_intrinsics, depth_scale_m):
    """Input: full config dict, color_intrinsics and depth_scale_m from
    start_camera_pipeline(). Returns a dict combining the camera config with
    values read back from the device, for session metadata (meta.json /
    README.md).
    """
    color_config = config["camera"]["color"]
    depth_config = config["camera"]["depth"]
    return {
        "color": {
            **color_config,
            "intrinsics_fx_fy_ppx_ppy": [color_intrinsics.fx, color_intrinsics.fy,
                                          color_intrinsics.ppx, color_intrinsics.ppy],
        },
        "depth": {
            **depth_config,
            "depth_scale_m": depth_scale_m,
        },
    }


def draw_overlay(frame, box, fingertip_detected, fingertip_depth_info,
                  is_recording, num_frames, num_force_samples, latest_force_fz, force_idle_s,
                  recording_fps):
    """Input: frame to draw on (mutated in place), current crop box, current
    tracking/recording/force status, and the live recording FPS (frames
    actually saved per second, over a short rolling window - can run below
    the camera's nominal frame rate if fingertip tracking/depth validation
    drops frames). Returns nothing.
    """
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                  (0, 255, 0) if fingertip_detected else (0, 0, 255), 2)

    for sample_point in fingertip_depth_info.get("sample_points", []):
        point = (sample_point["x"], sample_point["y"])
        color = (0, 255, 0) if sample_point["valid"] else (0, 0, 255)
        if 0 <= point[0] < frame.shape[1] and 0 <= point[1] < frame.shape[0]:
            cv2.circle(frame, point, 5, color, -1)

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

    if is_recording:
        cv2.circle(frame, (frame.shape[1] - 20, 20), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (frame.shape[1] - 65, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"frames={num_frames} force={num_force_samples}",
                    (frame.shape[1] - 220, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"recording fps={recording_fps:.1f}",
                    (frame.shape[1] - 220, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

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
    """Input: main display frame, color/depth crops from crop_color_and_depth(),
    the force-distribution canvas from draw_force_histogram(), full config dict
    (uses config['hand_tracking']). Shows the main RealSense window, the force
    distribution chart, plus whichever crop preview windows are enabled.
    Returns nothing.
    """
    hand_config = config["hand_tracking"]
    cv2.imshow("RealSense", main_frame)
    cv2.imshow("Force Distribution", force_histogram_canvas)
    if hand_config["show_tracked_crop"]:
        cv2.imshow("Fingertip Crop (color)", color_crop_bgr)
    if hand_config["show_depth_preview"]:
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
