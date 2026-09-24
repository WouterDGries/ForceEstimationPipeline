"""Session writers for the data-collection recorder.

RawSessionWriter (live, while recording) streams every camera frameset and
force sample to DataCollection/Raw/session_<timestamp>/ without buffering
it in memory:

    color.mkv        - full-resolution color, the camera's raw YUYV encoded
                        with ffmpeg (libx264 at config output.color_crf, 4:2:2)
    depth.mkv        - unaligned uint16 depth, lossless FFV1 (gray16le)
    frames_raw.csv   - one row per recorded frameset; row i is frame i of both
                        videos: frame number, device timestamps, host_ts_ns
    force_raw.csv    - one row per force sample: timestamp, Fz (N), zeroed live
                        against the gravity baseline (forceHandler.ForceCalibrator)
    meta_raw.json    - camera settings + stream calibration (intrinsics,
                        depth->color extrinsics) needed to align offline
    align_check.npz  - a few framesets aligned by librealsense itself, used
                        offline to verify the numpy alignment

SessionWriter (offline, see offlineExtract.py) buffers the fingertip crops
of one session and flushes them to DataCollection/Data/session_<timestamp>/
in the format Model/dataset.py reads:

    video.h5     - datasets 'color' (N,crop,crop,3 uint8, RGB) and
                    'depth' (N,crop,crop uint16, raw counts, see meta.json
                    for the depth scale in meters/count)
    frames.csv   - one row per saved video frame: timestamps, crop location,
                    whether the fingertip was detected and depth-validated
    force.csv    - one row per force sample: timestamp, Fz (N), zeroed against
                    the gravity baseline
    meta.json    - machine-readable camera/crop/force settings for this session
    README.md    - human-readable file list + camera settings for this session
"""

import csv
import json
import os
import queue
import subprocess
import threading
from collections import deque
from datetime import datetime

import h5py
import numpy as np

FRAME_FIELDS = [
    "frame_index", "host_ts_ns", "color_device_ts_ms", "depth_device_ts_ms",
    "fingertip_detected", "fingertip_depth_valid", "fingertip_depth_m",
    "neighborhood_min_depth_m", "neighborhood_max_depth_m", "depth_reference_count",
    "crop_x0", "crop_y0",
    "raw_frame_index", "crop_cx", "crop_cy", "crop_side_px", "crop_angle_deg", "fingertip_interpolated",
]
FORCE_FIELDS = ["sample_index", "host_ts_ns", "fz_N"]

RAW_FRAME_FIELDS = ["raw_frame_index", "color_frame_number", "color_device_ts_ms",
                    "depth_device_ts_ms", "host_ts_ns"]


def new_session_name():
    """Returns a fresh session folder name, session_<YYYYmmdd_HHMMSS>."""
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _color_encoder_args(output_config):
    encoder = output_config["color_encoder"]
    quality = str(output_config["color_crf"])
    if encoder == "libx264":
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", quality, "-pix_fmt", "yuv422p"]
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "constqp", "-qp", quality, "-pix_fmt", "yuv420p"]
    raise ValueError(f"Unknown output.color_encoder: {encoder}")


class _FfmpegPipe:
    """One ffmpeg subprocess fed raw frames over stdin by a writer thread
    from a bounded queue, so encoding never blocks the camera loop.
    """

    def __init__(self, name, input_args, output_args, output_path, queue_size):
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", *input_args, "-i", "-", *output_args, output_path]
        self.name = name
        self.queue = queue.Queue(maxsize=queue_size)
        self.error = None
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            frame = self.queue.get()
            if frame is None:
                break
            if self.error is not None:
                continue
            try:
                self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
            except (BrokenPipeError, OSError) as e:
                self.error = repr(e)
                print(f"[RawSessionWriter] {self.name} encoder stopped: {e!r}")

    def close(self):
        """Flushes the queue, closes ffmpeg's stdin and waits for it to finish.
        Returns an error string, or None if encoding succeeded.
        """
        self.queue.put(None)
        self._thread.join()
        try:
            self._process.stdin.close()
        except OSError:
            pass
        stderr = self._process.stderr.read().decode(errors="replace")
        return_code = self._process.wait()
        if return_code != 0 and self.error is None:
            self.error = f"ffmpeg exited with {return_code}: {stderr.strip()}"
        return self.error


class RawSessionWriter:
    """Streams one recording to raw_root/session_<timestamp>/ (see the module
    docstring). Every frameset goes to disk; if an encoder falls behind so
    far that its queue fills up, the frameset is dropped from both videos
    (keeping them index-aligned) and counted in queue_dropped.
    """

    def __init__(self, raw_root, session_name, camera_settings, config, align_check):
        self.session_name = session_name
        self.session_dir = os.path.join(raw_root, session_name)
        os.makedirs(self.session_dir)
        self._camera_settings = camera_settings
        self._config = config
        self._align_check = align_check

        color_config = config["camera"]["color"]
        depth_config = config["camera"]["depth"]
        output_config = config["output"]
        queue_size = output_config["writer_queue_frames"]
        self._color_pipe = _FfmpegPipe(
            "color",
            ["-pix_fmt", "yuyv422", "-s", f"{color_config['width']}x{color_config['height']}",
             "-framerate", str(color_config["framerate"])],
            [*_color_encoder_args(output_config), "-color_range", "tv", "-colorspace", "smpte170m"],
            os.path.join(self.session_dir, "color.mkv"), queue_size)
        self._depth_pipe = _FfmpegPipe(
            "depth",
            ["-pix_fmt", "gray16le", "-s", f"{depth_config['width']}x{depth_config['height']}",
             "-framerate", str(depth_config["framerate"])],
            ["-c:v", "ffv1", "-level", "3", "-slices", "4", "-threads", "4"],
            os.path.join(self.session_dir, "depth.mkv"), queue_size)

        self._frames_file = open(os.path.join(self.session_dir, "frames_raw.csv"), "w", newline="")
        self._frames_csv = csv.DictWriter(self._frames_file, fieldnames=RAW_FRAME_FIELDS)
        self._frames_csv.writeheader()
        self._force_file = open(os.path.join(self.session_dir, "force_raw.csv"), "w", newline="")
        self._force_csv = csv.DictWriter(self._force_file, fieldnames=FORCE_FIELDS)
        self._force_csv.writeheader()

        self._num_frames = 0
        self._num_force_samples = 0
        self._camera_dropped = 0
        self._queue_dropped = 0
        self._last_frame_number = None
        self._first_host_ts_ns = None
        self._last_host_ts_ns = None
        self._recent_host_ts_ns = deque(maxlen=30)  # rolling window for the live FPS readout

    def add_frameset(self, frame):
        """Input: frame dict from cameraHandler.get_frame(). Queues it for
        encoding and logs its row in frames_raw.csv. Returns True if queued,
        False if dropped because an encoder queue was full.
        """
        frame_number = frame["color_frame_number"]
        if self._last_frame_number is not None and frame_number > self._last_frame_number + 1:
            self._camera_dropped += frame_number - self._last_frame_number - 1
        self._last_frame_number = frame_number

        if self._color_pipe.queue.full() or self._depth_pipe.queue.full():
            self._queue_dropped += 1
            return False
        self._color_pipe.queue.put(frame["color_yuyv"])
        self._depth_pipe.queue.put(frame["depth_u16"])

        self._frames_csv.writerow({
            "raw_frame_index": self._num_frames,
            "color_frame_number": frame_number,
            "color_device_ts_ms": frame["color_device_ts_ms"],
            "depth_device_ts_ms": frame["depth_device_ts_ms"],
            "host_ts_ns": frame["host_ts_ns"],
        })
        self._num_frames += 1
        if self._first_host_ts_ns is None:
            self._first_host_ts_ns = frame["host_ts_ns"]
        self._last_host_ts_ns = frame["host_ts_ns"]
        self._recent_host_ts_ns.append(frame["host_ts_ns"])
        return True

    def add_force_samples(self, samples):
        """Input: list of (host_ts_ns, fz_N) tuples, already zeroed by the
        live ForceCalibrator. Writes them to force_raw.csv. Returns nothing.
        """
        for host_ts_ns, fz in samples:
            self._force_csv.writerow({
                "sample_index": self._num_force_samples,
                "host_ts_ns": host_ts_ns,
                "fz_N": fz,
            })
            self._num_force_samples += 1

    def stats(self):
        """Returns a dict with the live counters shown in the overlay."""
        span_ns = self._recent_host_ts_ns[-1] - self._recent_host_ts_ns[0] if len(self._recent_host_ts_ns) >= 2 else 0
        return {
            "num_frames": self._num_frames,
            "num_force_samples": self._num_force_samples,
            "fps": (len(self._recent_host_ts_ns) - 1) / (span_ns / 1e9) if span_ns > 0 else 0.0,
            "camera_dropped": self._camera_dropped,
            "queue_dropped": self._queue_dropped,
        }

    def close(self):
        """Finishes both encoders and writes meta_raw.json and align_check.npz.
        Returns (session_dir, error), error being None on success.
        """
        errors = [e for e in (self._color_pipe.close(), self._depth_pipe.close()) if e]
        self._frames_file.close()
        self._force_file.close()

        np.savez_compressed(os.path.join(self.session_dir, "align_check.npz"), **self._align_check)

        duration_s = ((self._last_host_ts_ns - self._first_host_ts_ns) / 1e9
                      if self._num_frames >= 2 else 0.0)
        raw_bytes = sum(os.path.getsize(os.path.join(self.session_dir, name))
                        for name in ("color.mkv", "depth.mkv"))
        meta = {
            "session_name": self.session_name,
            "num_frames": self._num_frames,
            "num_force_samples": self._num_force_samples,
            "camera_dropped_frames": self._camera_dropped,
            "queue_dropped_frames": self._queue_dropped,
            "duration_s": duration_s,
            "video_bytes": raw_bytes,
            "video_mb_per_s": raw_bytes / 1e6 / duration_s if duration_s > 0 else None,
            "encoder_errors": errors,
            "timestamp_clock": "time.perf_counter_ns() - monotonic, shared by frames_raw.csv and force_raw.csv",
            "color_encoding": {"encoder": self._config["output"]["color_encoder"],
                               "quality": self._config["output"]["color_crf"],
                               "source_format": "yuyv422 (camera native)"},
            "depth_encoding": "ffv1 gray16le (lossless), unaligned raw counts",
            "camera": self._camera_settings,
            "hand_tracking": self._config["hand_tracking"],
            "force_sensor": self._config["force_sensor"],
        }
        with open(os.path.join(self.session_dir, "meta_raw.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return self.session_dir, ("; ".join(errors) if errors else None)


class SessionWriter:
    def __init__(self, session_dir, camera_settings, crop_size, tracked_hand_label,
                 fingertip_landmark_index, force_config, depth_validation_config, crop_config):
        self._camera_settings = camera_settings
        self._crop_size = crop_size
        self._tracked_hand_label = tracked_hand_label
        self._fingertip_landmark_index = fingertip_landmark_index
        self._force_config = force_config
        self._depth_validation_config = depth_validation_config
        self._crop_config = crop_config

        self.session_dir = session_dir
        self._color_frames = []
        self._depth_frames = []
        self._frame_rows = []
        self._force_rows = []

    def add_frame(self, frame_index, host_ts_ns, color_device_ts_ms, depth_device_ts_ms,
                  fingertip_detected, fingertip_depth_info, box, color_crop_rgb, depth_crop,
                  crop_geometry):
        """Input: frame index/timestamps, whether the fingertip was found,
        depth validation info, axis-aligned bounding box of the crop,
        color/depth crop images, and crop_geometry (dict with raw_frame_index,
        crop_cx, crop_cy, crop_side_px, crop_angle_deg, fingertip_interpolated).
        Buffers this frame in memory. Returns nothing.
        """
        self._color_frames.append(color_crop_rgb)
        self._depth_frames.append(depth_crop)
        self._frame_rows.append({
            "frame_index": frame_index,
            "host_ts_ns": host_ts_ns,
            "color_device_ts_ms": color_device_ts_ms,
            "depth_device_ts_ms": depth_device_ts_ms,
            "fingertip_detected": int(fingertip_detected),
            "fingertip_depth_valid": int(fingertip_depth_info["valid"]),
            "fingertip_depth_m": fingertip_depth_info["center_depth_m"],
            "neighborhood_min_depth_m": fingertip_depth_info["neighborhood_min_depth_m"],
            "neighborhood_max_depth_m": fingertip_depth_info["neighborhood_max_depth_m"],
            "depth_reference_count": fingertip_depth_info["reference_count"],
            "crop_x0": box[0],
            "crop_y0": box[1],
            **crop_geometry,
        })

    def add_force_samples(self, samples):
        """Input: list of (host_ts_ns, fz_N) tuples. Buffers them in memory.
        Returns nothing.
        """
        for host_ts_ns, fz in samples:
            self._force_rows.append({
                "sample_index": len(self._force_rows),
                "host_ts_ns": host_ts_ns,
                "fz_N": fz,
            })

    def num_frames(self):
        return len(self._frame_rows)

    def num_force_samples(self):
        return len(self._force_rows)

    def flush(self):
        """Writes video.h5, frames.csv, force.csv, meta.json and README.md
        to self.session_dir. Returns self.session_dir.
        """
        os.makedirs(self.session_dir, exist_ok=True)

        color_array = np.stack(self._color_frames, axis=0) if self._color_frames else \
            np.zeros((0, self._crop_size, self._crop_size, 3), dtype=np.uint8)
        depth_array = np.stack(self._depth_frames, axis=0) if self._depth_frames else \
            np.zeros((0, self._crop_size, self._crop_size), dtype=np.uint16)

        video_path = os.path.join(self.session_dir, "video.h5")
        with h5py.File(video_path, "w") as h5f:
            h5f.create_dataset("color", data=color_array, compression="gzip")
            h5f.create_dataset("depth", data=depth_array, compression="gzip")

        frames_path = os.path.join(self.session_dir, "frames.csv")
        with open(frames_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FRAME_FIELDS)
            writer.writeheader()
            writer.writerows(self._frame_rows)

        force_path = os.path.join(self.session_dir, "force.csv")
        with open(force_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FORCE_FIELDS)
            writer.writeheader()
            writer.writerows(self._force_rows)

        meta = self._build_meta()
        meta_path = os.path.join(self.session_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self._write_readme(meta)
        return self.session_dir

    def _build_meta(self):
        return {
            "session_dir": self.session_dir,
            "num_frames": len(self._frame_rows),
            "num_force_samples": len(self._force_rows),
            "timestamp_clock": "time.perf_counter_ns() - monotonic, shared by frames.csv and force.csv",
            "camera": self._camera_settings,
            "crop": {
                "size_px": self._crop_size,
                "color_channel_order": "RGB",
                "depth_units": "raw uint16 counts; multiply by camera.depth_scale_m for meters",
                "tracked_hand": self._tracked_hand_label,
                "fingertip_landmark_index": self._fingertip_landmark_index,
                "depth_validation": self._depth_validation_config,
                **self._crop_config,
            },
            "force_sensor": {
                "ip": self._force_config["ip"],
                "port": self._force_config["port"],
                "counts_per_newton": self._force_config["counts_per_newton"],
                "calibration_verified": False,
                "auto_zero_interval_s": self._force_config["calibration_interval_s"],
            },
        }

    def _write_readme(self, meta):
        cam = meta["camera"]
        crop = meta["crop"]
        lines = [
            f"# Recording session: {os.path.basename(self.session_dir)}",
            "",
            "## Files in this session",
            "",
            "| File | Contents |",
            "|---|---|",
            f"| `video.h5` | datasets `color` (N,{self._crop_size},{self._crop_size},3 uint8, RGB) and "
            f"`depth` (N,{self._crop_size},{self._crop_size} uint16, raw counts) |",
            "| `frames.csv` | one row per saved video frame, including fingertip detection/depth validation "
            "measurements and crop location |",
            "| `force.csv` | one row per force sample: `sample_index, host_ts_ns, fz_N` (zeroed against "
            "the gravity baseline) |",
            "| `meta.json` | machine-readable copy of the settings below |",
            "| `qc/` | offline-extraction quality check: per-frame log of all raw frames, stability "
            "metrics and a strip of crops |",
            "",
            f"Frame count: {meta['num_frames']}  |  Force sample count: {meta['num_force_samples']}",
            "",
            "`frames.csv` and `force.csv` share the same clock (`time.perf_counter_ns`, monotonic, "
            "not wall-clock), so rows from the two files can be time-aligned by `host_ts_ns` during "
            "offline post-processing.",
            "",
            f"Extracted offline by `dataAcquisition/offlineExtract.py` from raw session "
            f"`{crop.get('raw_session_dir', '?')}`.",
            "",
            "## Camera settings used",
            "",
            f"- Color: {cam['color']['width']}x{cam['color']['height']} @ {cam['color']['framerate']} fps, "
            f"auto_exposure={cam['color']['auto_exposure']}, exposure={cam['color']['exposure']}, "
            f"gain={cam['color']['gain']}, auto_white_balance={cam['color']['auto_white_balance']}, "
            f"white_balance={cam['color']['white_balance']}K, brightness={cam['color']['brightness']}, "
            f"contrast={cam['color']['contrast']}",
            f"- Depth: {cam['depth']['width']}x{cam['depth']['height']} @ {cam['depth']['framerate']} fps, "
            f"depth_units={cam['depth']['depth_units_m']} m/count, actual depth_scale={cam['depth']['depth_scale_m']} "
            f"m/count, emitter_enabled={cam['depth']['emitter_enabled']}, laser_power={cam['depth']['laser_power']}, "
            f"visual_preset={cam['depth']['visual_preset']}",
            f"- Crop: {self._crop_size}x{self._crop_size} px covering {crop['crop_side_mm']} mm at the fingertip "
            f"depth, centered {crop['anchor_alpha']} x |TIP-DIP| behind the {self._tracked_hand_label} hand's "
            f"MediaPipe landmark {self._fingertip_landmark_index} (index fingertip), zero-phase smoothed, "
            f"rotate_to_finger_axis={crop['rotate_to_finger_axis']}",
            f"- Force sensor: ATI NetFT over UDP at {self._force_config['ip']}:{self._force_config['port']}, Fz only, "
            f"counts_per_newton={self._force_config['counts_per_newton']} "
            "(**placeholder value — verify against the sensor's calibration file**), "
            f"auto-zeroed to the mean Fz every {self._force_config['calibration_interval_s']:.0f}s "
            "whenever the fingertip isn't validated as touching (depth-validation 5-of-8 rule)",
            "",
        ]
        with open(os.path.join(self.session_dir, "README.md"), "w") as f:
            f.write("\n".join(lines))
