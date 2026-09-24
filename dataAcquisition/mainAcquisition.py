"""Main acquisition loop for the RGB-D + force-torque data-collection
recorder (see force_estimation_model_guide.md, Stage 3: "Recording
software"). While recording, every camera frameset (full-resolution raw
YUYV color + unaligned depth) and every force sample is streamed to
DataCollection/Raw/session_<ts>/ by dataWriter.RawSessionWriter - nothing
heavy runs here, so the camera's full frame rate is kept at 1920x1080.
When a recording stops, offlineExtract.py is started in the background to
track the fingertip, crop, and write DataCollection/Data/session_<ts>/.
All tunable parameters live in config.yaml.

The live preview (downscaled frame, MediaPipe every n-th frame, depth
validation on a small aligned ROI) only serves the operator and the force
auto-zero; it does not decide what ends up in the dataset.

Controls:
    r - start / stop recording (stop also queues offline processing)
    q / ESC - quit (finishes an in-progress recording first, then waits for
              queued offline processing - Ctrl+C skips that; run
              offlineExtract.py --all later)
"""

import os
import queue
import subprocess
import sys
import threading

import cv2
import yaml

import cameraHandler
import dataWriter
import forceHandler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config.yaml")


#getting config from file
def load_config(path=CONFIG_PATH):
    """Input: path to a YAML config file. Returns it parsed as a dict, with
    the recorder's relative paths (model, output folders) resolved against
    dataAcquisition/ so the scripts work from any working directory.
    """
    with open(path) as f:
        config = yaml.safe_load(f)

    def resolve(path_value):
        return path_value if os.path.isabs(path_value) else os.path.normpath(os.path.join(SCRIPT_DIR, path_value))

    config["hand_tracking"]["model_path"] = resolve(config["hand_tracking"]["model_path"])
    config["output"]["root"] = resolve(config["output"]["root"])
    config["output"]["raw_root"] = resolve(config["output"]["raw_root"])
    return config


def _low_priority_kwargs():
    """subprocess kwargs that start the child at low CPU priority (Linux/WSL or Windows)."""
    if os.name == "nt":
        return {"creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS}
    return {"preexec_fn": lambda: os.nice(10)}


class OfflineProcessingQueue:
    """Runs offlineExtract.py on finished recordings one at a time in a
    background thread, as a low-priority subprocess so the live loop keeps
    the CPU. Each run's output goes to offline_extract.log in the raw folder.
    """

    def __init__(self):
        self._queue = queue.Queue()
        self._pending = 0
        self._lock = threading.Lock()
        self.status = ""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, raw_session_dir):
        with self._lock:
            self._pending += 1
        self._queue.put(raw_session_dir)

    def pending(self):
        with self._lock:
            return self._pending

    def _run(self):
        while True:
            raw_session_dir = self._queue.get()
            name = os.path.basename(raw_session_dir)
            self.status = f"Offline: processing {name} ({self.pending() - 1} queued)"
            log_path = os.path.join(raw_session_dir, "offline_extract.log")
            with open(log_path, "w") as log:
                result = subprocess.run(
                    [sys.executable, os.path.join(SCRIPT_DIR, "offlineExtract.py"), raw_session_dir],
                    stdout=log, stderr=subprocess.STDOUT, **_low_priority_kwargs())
            if result.returncode == 0:
                self.status = f"Offline: {name} done"
                print(f"Offline processing finished: {name}")
            else:
                self.status = f"Offline: {name} FAILED (see offline_extract.log)"
                print(f"Offline processing FAILED for {name}, see {log_path}")
            with self._lock:
                self._pending -= 1

    def wait(self):
        """Blocks until every submitted session has been processed."""
        while self.pending() > 0:
            threading.Event().wait(0.5)


def main():
    config = load_config()                                      #Loading configuration
    os.makedirs(config["output"]["root"], exist_ok=True)        #Creating output folders
    os.makedirs(config["output"]["raw_root"], exist_ok=True)

    pipeline, stream_calibration, depth_scale_m = cameraHandler.start_camera_pipeline(config)  #Starting camera
    camera_settings = cameraHandler.build_camera_settings_dict(config, stream_calibration, depth_scale_m)  #Camera metadata
    aligner = cameraHandler.build_depth_aligner(stream_calibration, depth_scale_m)  #ROI depth->color alignment
    align_check = cameraHandler.capture_align_check(pipeline)  #rs.align reference for the offline alignment check

    landmarker = cameraHandler.build_hand_landmarker(config)  #Preparing live (preview-only) hand tracking
    last_mp_timestamp_ms = None
    offline_queue = OfflineProcessingQueue()

    force_reader = forceHandler.build_force_reader(config)   #Preparing force sensor
    force_reader.start()                                     #Starting force sensor
    force_calibrator = forceHandler.build_force_calibrator(config)  #Preparing calibration
    force_histogram = forceHandler.ForceHistogram()          #Tracking force-range coverage for this take

    full_size = (config["camera"]["color"]["width"], config["camera"]["color"]["height"])
    color_intrinsics = stream_calibration["color_intrinsics"]
    hand_config = config["hand_tracking"]

    raw_writer = None
    loop_index = 0
    latest_force_fz = 0.0

    try:
        while True:
            frame = cameraHandler.get_frame(pipeline)  #Getting camera frameset (raw YUYV + unaligned depth)
            if frame is None:
                continue
            loop_index += 1

            if raw_writer is not None:
                raw_writer.add_frameset(frame)  #Queuing frameset for encoding (never blocks)

            full_bgr, preview_bgr, scale = cameraHandler.yuyv_to_preview_bgr(frame["color_yuyv"], config)
            last_mp_timestamp_ms = cameraHandler.next_mediapipe_timestamp_ms(frame["host_ts_ns"], last_mp_timestamp_ms)
            submit = loop_index % hand_config["live_mediapipe_every_n"] == 0
            fingertip_px = cameraHandler.track_fingertip_live(preview_bgr, landmarker, last_mp_timestamp_ms,
                                                              submit, full_size, config)  #Finding fingertip (preview)

            fingertip_detected = fingertip_px is not None
            fingertip_depth_info = cameraHandler.validate_fingertip_depth_unaligned(
                aligner, frame["depth_u16"], fingertip_px, config
            ) if fingertip_detected else cameraHandler.empty_depth_info()

            box, color_crop_bgr, depth_crop = None, None, None
            if fingertip_detected and fingertip_depth_info["center_depth_m"]:
                side = cameraHandler.crop_side_px(fingertip_depth_info["center_depth_m"], color_intrinsics["fx"], config)
                box = cameraHandler.compute_crop_box(fingertip_px[0], fingertip_px[1], side, *full_size)
                if hand_config["show_tracked_crop"] or hand_config["show_depth_preview"]:
                    output_px = hand_config["crop_output_px"]
                    color_crop_bgr = cv2.resize(full_bgr[box[1]:box[3], box[0]:box[2]], (output_px, output_px),
                                                interpolation=cv2.INTER_AREA)
                    depth_crop = cv2.resize(aligner.align_roi(frame["depth_u16"], box), (output_px, output_px),
                                            interpolation=cv2.INTER_NEAREST)

            raw_force_samples = force_reader.drain()  #Reading force samples
            force_calibrator.update([fz for _, fz in raw_force_samples], fingertip_depth_info["valid"])  #Updating calibration
            force_samples = [(ts, force_calibrator.apply(fz)) for ts, fz in raw_force_samples]  #Applying calibration
            latest_force_fz = force_samples[-1][1] if force_samples else latest_force_fz  #Updating displayed force

            if raw_writer is not None:
                raw_writer.add_force_samples(force_samples)  #Saving force samples
                if fingertip_depth_info["valid"]:  #Tracking force-range coverage while touching
                    force_histogram.add_samples(fz for _, fz in force_samples)

            cameraHandler.draw_overlay(preview_bgr, scale, box, fingertip_detected,
                                        fingertip_depth_info, raw_writer is not None,  #Drawing status overlay
                                        raw_writer.stats() if raw_writer else None, latest_force_fz,
                                        force_reader.seconds_since_last_packet(), offline_queue.status)
            histogram_canvas = cameraHandler.draw_force_histogram(
                force_histogram.percentages(), forceHandler.ForceHistogram.BIN_WIDTH_N, raw_writer is not None)
            cameraHandler.show_previews(preview_bgr, color_crop_bgr, depth_crop,
                                         histogram_canvas, config)  #Showing previews

            key = cv2.waitKey(1) & 0xFF  #Checking keyboard input
            if key in (ord('q'), 27):
                break
            elif key == ord('r') and raw_writer is None:
                raw_writer = dataWriter.RawSessionWriter(config["output"]["raw_root"], dataWriter.new_session_name(),
                                                         camera_settings, config, align_check)  #Starting recording
                force_histogram.reset()                    #Starting a fresh force-coverage chart for this take
                print(f"Recording started: {raw_writer.session_dir}")
            elif key == ord('r') and raw_writer is not None:
                _finish_recording(raw_writer, config, offline_queue)  #Stopping recording
                raw_writer = None

    finally:
        if raw_writer is not None:
            _finish_recording(raw_writer, config, offline_queue)  #Saving unfinished recording
        force_reader.stop()                            #Stopping force sensor
        cameraHandler.stop(pipeline, landmarker)        #Stopping camera and hand tracking
        if offline_queue.pending():
            print(f"Waiting for offline processing of {offline_queue.pending()} session(s) "
                  "(Ctrl+C to skip; run offlineExtract.py --all later)")
            try:
                offline_queue.wait()
            except KeyboardInterrupt:
                print("Skipped - unprocessed sessions stay in Raw/.")


def _finish_recording(raw_writer, config, offline_queue):
    """Closes the raw writer, reports drops, and queues offline processing."""
    stats = raw_writer.stats()
    session_dir, error = raw_writer.close()
    print(f"Recording stopped, raw data saved to: {session_dir} ({stats['num_frames']} frames, "
          f"camera drops={stats['camera_dropped']}, queue drops={stats['queue_dropped']})")
    if error:
        print(f"WARNING: encoder error, not processing this take: {error}")
    elif config["output"]["process_on_stop"]:
        offline_queue.submit(session_dir)


if __name__ == "__main__":
    main()
