"""Main acquisition loop for the RGB-D + force-torque data-collection
recorder (see force_estimation_model_guide.md, Stage 3: "Recording
software"). Grabs an aligned color+depth frame, tracks the fingertip, crops
around it, drains force samples, and (while recording) writes everything
via dataWriter. All tunable parameters live in config.yaml.

Controls:
    r - start / stop recording (writes a new session folder on stop)
    q / ESC - quit (flushes an in-progress recording first)
"""

import os
from collections import deque

import cv2
import yaml

import cameraHandler
import dataWriter
import forceHandler

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

#getting config from file
def load_config(path=CONFIG_PATH):
    """Input: path to a YAML config file. Returns it parsed as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()                                      #Loading configuration
    os.makedirs(config["output"]["root"], exist_ok=True)        #Creating output folder

    pipeline, align, color_intrinsics, depth_scale_m = cameraHandler.start_camera_pipeline(config)  #Starting camera
    camera_settings = cameraHandler.build_camera_settings_dict(config, color_intrinsics, depth_scale_m)  #Camera metadata

    landmarker = cameraHandler.build_hand_landmarker(config)  #Preparing hand tracking
    last_mp_timestamp_ms = None

    force_reader = forceHandler.build_force_reader(config)   #Preparing force sensor
    force_reader.start()                                     #Starting force sensor
    force_calibrator = forceHandler.build_force_calibrator(config)  #Preparing calibration
    force_histogram = forceHandler.ForceHistogram()          #Tracking force-range coverage for this take

    frame_width = config["camera"]["color"]["width"]
    frame_height = config["camera"]["color"]["height"]
    last_crop_center = (frame_width // 2, frame_height // 2)

    session_writer = None
    is_recording = False
    frame_index = 0
    latest_force_fz = 0.0
    recorded_frame_times_ns = deque(maxlen=30)  # rolling window for the live recording-FPS readout

    try:
        while True:
            color_bgr, depth_u16, color_ts_ms, depth_ts_ms, host_ts_ns = cameraHandler.get_frame(pipeline, align)  #Getting camera frame
            if color_bgr is None:
                continue

            last_mp_timestamp_ms = cameraHandler.next_mediapipe_timestamp_ms(host_ts_ns, last_mp_timestamp_ms)  #Updating timestamp
            fingertip_px = cameraHandler.track_fingertip(color_bgr, landmarker, last_mp_timestamp_ms, config)  #Finding fingertip

            fingertip_detected = fingertip_px is not None
            last_crop_center = fingertip_px if fingertip_detected else last_crop_center
            fingertip_depth_info = cameraHandler.validate_fingertip_depth(
                depth_u16, fingertip_px, depth_scale_m, color_intrinsics, config
            ) if fingertip_detected else {
                "valid": False,
                "center_depth_m": None,
                "neighborhood_min_depth_m": None,
                "neighborhood_max_depth_m": None,
                "reference_count": 0,
                "valid_sample_count": 0,
                "sample_points": [],
            }

            box = cameraHandler.compute_crop_box(last_crop_center[0], last_crop_center[1],
                                                  frame_width, frame_height, config)  #Calculating crop area
            color_crop_bgr, color_crop_rgb, depth_crop = cameraHandler.crop_color_and_depth(
                color_bgr, depth_u16, box)  #Cropping color and depth images

            raw_force_samples = force_reader.drain()  #Reading force samples
            force_calibrator.update([fz for _, fz in raw_force_samples], fingertip_depth_info["valid"])  #Updating calibration
            force_samples = [(ts, force_calibrator.apply(fz)) for ts, fz in raw_force_samples]  #Applying calibration
            latest_force_fz = force_samples[-1][1] if force_samples else latest_force_fz  #Updating displayed force

            frame_is_valid = is_recording and fingertip_detected and fingertip_depth_info["valid"]
            if is_recording:
                if frame_is_valid:
                    session_writer.add_frame(frame_index, host_ts_ns, color_ts_ms, depth_ts_ms,  #Saving frame
                                              fingertip_detected, fingertip_depth_info, box,
                                              color_crop_rgb, depth_crop)
                    frame_index += 1
                    recorded_frame_times_ns.append(host_ts_ns)  #Tracking live recording FPS
                    force_histogram.add_samples(fz for _, fz in force_samples)  #Tracking force-range coverage
                session_writer.add_force_samples(force_samples)  #Saving force samples

            if len(recorded_frame_times_ns) >= 2:
                span_s = (recorded_frame_times_ns[-1] - recorded_frame_times_ns[0]) / 1e9
                recording_fps = (len(recorded_frame_times_ns) - 1) / span_s if span_s > 0 else 0.0
            else:
                recording_fps = 0.0

            num_frames = session_writer.num_frames() if session_writer else 0
            num_force_samples = session_writer.num_force_samples() if session_writer else 0
            # Draw on a copy, not color_bgr itself: color_bgr is a zero-copy view over the
            # RealSense driver's own frame buffer, which can be reused for a later frame -
            # mutating it in place risks bleeding these overlay pixels into a future capture.
            display_frame = color_bgr.copy()
            cameraHandler.draw_overlay(display_frame, box, fingertip_detected,
                                        fingertip_depth_info, is_recording,  #Drawing status overlay
                                        num_frames, num_force_samples, latest_force_fz,
                                        force_reader.seconds_since_last_packet(), recording_fps)
            histogram_canvas = cameraHandler.draw_force_histogram(
                force_histogram.percentages(), forceHandler.ForceHistogram.BIN_WIDTH_N, is_recording)
            cameraHandler.show_previews(display_frame, color_crop_bgr, depth_crop,
                                         histogram_canvas, config)  #Showing previews

            key = cv2.waitKey(1) & 0xFF  #Checking keyboard input
            if key in (ord('q'), 27):
                break
            elif key == ord('r') and not is_recording:
                session_writer = dataWriter.start_session(config, camera_settings)  #Starting recording
                is_recording, frame_index = True, 0
                force_histogram.reset()                    #Starting a fresh force-coverage chart for this take
                recorded_frame_times_ns.clear()             #Starting a fresh recording-FPS readout for this take
                print(f"Recording started: {session_writer.session_dir}")
            elif key == ord('r') and is_recording:
                session_dir = dataWriter.stop_session(session_writer)  #Stopping recording
                is_recording, session_writer = False, None
                print(f"Recording stopped, saved to: {session_dir}")

    finally:
        if is_recording and session_writer is not None:
            session_dir = dataWriter.stop_session(session_writer)  #Saving unfinished recording
            print(f"Recording flushed on exit: {session_dir}")
        force_reader.stop()                            #Stopping force sensor
        cameraHandler.stop(pipeline, landmarker)        #Stopping camera and hand tracking


if __name__ == "__main__":
    main()
