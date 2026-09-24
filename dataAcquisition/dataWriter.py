"""Session writer for the data-collection recorder: buffers frames + force
samples in memory while recording, and flushes everything to
DataCollection/Data/session_<timestamp>/ on stop:

    video.h5     - datasets 'color' (N,crop,crop,3 uint8, RGB) and
                    'depth' (N,crop,crop uint16, raw counts, see meta.json
                    for the depth scale in meters/count)
    frames.csv   - one row per video frame: timestamps, crop location,
                    whether the fingertip was detected and depth-validated
    force.csv    - one row per force sample: timestamp, Fz (N), zeroed against
                    the gravity baseline
    meta.json    - machine-readable camera/crop/force settings for this session
    README.md    - human-readable file list + camera settings for this session
"""

import csv
import json
import os
from datetime import datetime
###
import h5py
import numpy as np

FRAME_FIELDS = [
    "frame_index", "host_ts_ns", "color_device_ts_ms", "depth_device_ts_ms",
    "fingertip_detected", "fingertip_depth_valid", "fingertip_depth_m",
    "neighborhood_min_depth_m", "neighborhood_max_depth_m", "depth_reference_count",
    "crop_x0", "crop_y0",
]
FORCE_FIELDS = ["sample_index", "host_ts_ns", "fz_N"]


class SessionWriter:
    def __init__(self, output_root, camera_settings, crop_size, tracked_hand_label,
                 fingertip_landmark_index, force_config, depth_validation_config):
        self._camera_settings = camera_settings
        self._crop_size = crop_size
        self._tracked_hand_label = tracked_hand_label
        self._fingertip_landmark_index = fingertip_landmark_index
        self._force_config = force_config
        self._depth_validation_config = depth_validation_config

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_root, f"session_{timestamp}")
        self._color_frames = []
        self._depth_frames = []
        self._frame_rows = []
        self._force_rows = []

    def add_frame(self, frame_index, host_ts_ns, color_device_ts_ms, depth_device_ts_ms,
                  fingertip_detected, fingertip_depth_info, box, color_crop_rgb, depth_crop):
        """Input: frame index/timestamps, whether the fingertip was found,
        depth validation info, crop box, and color/depth crop images. Buffers
        this frame in memory. Returns nothing.
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
            "",
            f"Frame count: {meta['num_frames']}  |  Force sample count: {meta['num_force_samples']}",
            "",
            "`frames.csv` and `force.csv` share the same clock (`time.perf_counter_ns`, monotonic, "
            "not wall-clock), so rows from the two files can be time-aligned by `host_ts_ns` during "
            "offline post-processing.",
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
            f"- Crop: {self._crop_size}x{self._crop_size} px, centered on the {self._tracked_hand_label} hand's "
            f"MediaPipe landmark {self._fingertip_landmark_index} (index fingertip)",
            f"- Force sensor: ATI NetFT over UDP at {self._force_config['ip']}:{self._force_config['port']}, Fz only, "
            f"counts_per_newton={self._force_config['counts_per_newton']} "
            "(**placeholder value — verify against the sensor's calibration file**), "
            f"auto-zeroed to the mean Fz every {self._force_config['calibration_interval_s']:.0f}s "
            "whenever the fingertip isn't validated as touching (depth-validation 5-of-8 rule)",
            "",
        ]
        with open(os.path.join(self.session_dir, "README.md"), "w") as f:
            f.write("\n".join(lines))


def start_session(config, camera_settings):
    """Input: full config dict, camera_settings from
    cameraHandler.build_camera_settings_dict(). Creates a new SessionWriter
    (and its session_dir) using config['output']/['hand_tracking']/['force_sensor'].
    Returns the SessionWriter.
    """
    hand_config = config["hand_tracking"]
    return SessionWriter(
        config["output"]["root"], camera_settings, hand_config["crop_size"],
        hand_config["tracked_hand_label"], hand_config["fingertip_landmark_index"],
        config["force_sensor"], {
            "tolerance_m": hand_config["depth_validation_tolerance_m"],
            "radius_m": hand_config["depth_validation_radius_m"],
            "minimum_valid_samples": hand_config["minimum_valid_depth_samples"],
        },
    )


def stop_session(session_writer):
    """Input: an active SessionWriter. Flushes it to disk. Returns the
    session directory path.
    """
    return session_writer.flush()
