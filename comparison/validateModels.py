"""Live validation of the 3D, 2D and PressureVision force models on new,
unseen recordings, with TrajectoryPipeline's capture and processing and the
weights trained in ForceEstimationPipeline (Model/, Model2D/) and
ForceEstimationPressureVision.

Per test:
  1. Capture as in TrajectoryPipeline's main.py: color + depth aligned by
     librealsense (rs.align), MediaPipe LIVE_STREAM on a downscaled frame
     every live_mediapipe_every_n-th frame, and the fingertip only counts as
     touching when its depth agrees with the 8-point ring
     (functions.getFingerVector). Every recorded frame is kept in memory at
     full resolution, with the live MediaPipe result of its own frame. The
     force sensor and its auto-zero are dataAcquisition's (forceHandler);
     its samples are kept in memory as well. Press 'a': after countdown_s
     the recording starts and the touching fingertip is drawn as a trail
     (colour = measured force). Once the finger has touched and is lifted
     (TrajectoryPipeline's lift_stop_frames frames without a valid
     fingertip), the recording stops and is saved, losslessly, to
     comparison/readyForProcessing/<session>/ (color.npy, depth.npy,
     recording.pkl); the next test can start right away.
  2. Later, with --process, every saved recording is processed (models
     loaded once): the models run one after another (3D, 2D, PressureVision), each from
     scratch with its own timer: offline tracking + crops
     (functions.extractSession: MediaPipe VIDEO on every full frame, the live
     fingertip where it finds none), saveSession, estimateForce, the Potts
     filter (pottsFilterL1) - then that model's session data is deleted, the
     timer restarts and the next model creates its data again. The 3D timer
     starts once the recording has been loaded from disk (in a replay: once
     the window has been replayed).
  3. Labels from the recorded force (preprocessing/dataProcessor.py's
     frame_force_labels: filtered Fz, force_n = max(-Fz, 0), artifacts
     flagged). comparison/validation/<session>/ gets report.md, one plot per
     model, one with all models, one with the Potts-filtered estimates,
     predictions.csv and (live) screen_frames/: the displayed preview with the
     trail once per second of the recording, written when it is saved).
     Afterwards every model's MAE is printed. A recording's folder in
     readyForProcessing is deleted once its report was written without
     problems; recordings with problems are kept for another --process run.

Nothing in the three projects is modified; their functions are imported.
Models are loaded and warmed up once at startup (not timed); capturing
does not load them.

Settings: comparison/validateModels.yaml (checkpoints, countdown, output),
TrajectoryPipeline's pipeline/config.yaml (camera, tracking, crops,
lift_stop_frames, Potts filter) and config.yaml (force_sensor, processing.force
for the labels, the comparison block's frame and metric settings).

Controls: a - start a test (countdown, then record until the finger is lifted)
          x - cancel the running countdown / recording
          q / ESC - quit

Usage (trajPipeline env, from the repo root):
    python comparison/validateModels.py              record tests (saved to readyForProcessing)
    python comparison/validateModels.py --process    process every saved recording
    python comparison/validateModels.py --raw_session dataAcquisition/DataCollection/Raw/session_x --start_s 50 --end_s 75
        replays a window of a raw recording through the same live capture code (at the recorded pace, depth
        aligned with the acquisition's numpy aligner instead of rs.align) instead of using the camera
"""

import argparse
import contextlib
import copy
import importlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pyrealsense2 as rs
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "config.yaml")
SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validateModels.yaml")

# Force sensor, raw-recording decoding and depth alignment are the recorder's own code
sys.path.insert(0, os.path.join(REPO_ROOT, "dataAcquisition"))
import cameraHandler  # noqa: E402
import forceHandler  # noqa: E402
import offlineExtract  # noqa: E402
from depthAlign import intrinsics_to_dict  # noqa: E402
from mainAcquisition import load_config as load_acquisition_config  # noqa: E402

# Top-level module names used by more than one of the projects
GENERIC_MODULE_NAMES = ("dataset", "model", "train", "evaluate", "dataProcessor", "depthAlign", "depth")


def repo_path(path):
    """Returns path resolved against the repo root (absolute paths unchanged)."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(REPO_ROOT, path))


def import_from(code_dir, names):
    """Input: a project folder and module names in it. Imports them from that
    folder while the generic names (GENERIC_MODULE_NAMES) of every other
    project are hidden, then hides them again, so modules with the same name
    from different projects can live in one interpreter: each keeps its own
    dependencies in its globals. Returns a SimpleNamespace of the modules.
    """
    saved = {name: sys.modules.pop(name) for name in GENERIC_MODULE_NAMES if name in sys.modules}
    sys.path.insert(0, code_dir)
    try:
        modules = {name: importlib.import_module(name) for name in names}
    finally:
        sys.path.remove(code_dir)
        for name in GENERIC_MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    return SimpleNamespace(**modules)


def autocast_context(device):
    """bf16 on Ampere or newer, fp16 on older GPUs, nothing on CPU - as TrajectoryPipeline's predictForce."""
    if device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


class Timer:
    """Wall-clock stages relative to t0."""

    def __init__(self, t0):
        self.t0 = t0
        self.stages = []

    @contextlib.contextmanager
    def stage(self, key, label):
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self.stages.append({"key": key, "label": label, "start_s": start - self.t0, "end_s": end - self.t0,
                                "duration_s": end - start})

    def total_s(self):
        return self.stages[-1]["end_s"] if self.stages else None


def _read_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- TrajectoryPipeline

class TrajectoryPipeline:
    """TrajectoryPipeline's pipeline/functions.py with its own config.yaml,
    its own model loaders, and the validation checkpoints.
    """

    def __init__(self, root, models_settings, device):
        root = repo_path(root)
        if root not in sys.path:
            sys.path.insert(0, root)
        self.tp = importlib.import_module("pipeline.functions")
        self.config = self.tp.loadConfig()
        self.device = device
        self.models = {}
        for name, model_settings in models_settings.items():
            checkpoint_path = repo_path(model_settings["checkpoint"])
            net, metadata = self.tp.loadForceModel(model_settings["kind"], checkpoint_path, device)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.models[name] = {"type": model_settings["kind"], "checkpoint": checkpoint_path, "net": net,
                                 "metadata": metadata, "epoch": checkpoint["epoch"], "splits": checkpoint["splits"]}

    def warm_up(self):
        for entry in self.models.values():
            with torch.inference_mode(), autocast_context(self.device):
                if entry["type"] == "pressurevision":
                    px = entry["metadata"]["crop_output_px"]
                    entry["net"](torch.zeros((1, 3, px, px), device=self.device), torch.full((1,), 1e-6, device=self.device))
                else:
                    shape = (1, 4, int(entry["metadata"]["window_length"]), 112, 112) if entry["type"] == "3d" \
                        else (1, 4, 112, 112)
                    entry["net"](torch.zeros(shape, device=self.device))

    def live_landmarker(self):
        """MediaPipe LIVE_STREAM landmarker reporting to functions.print_result, as in main.py."""
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.config["hand_tracking"]["model_path"]),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self.tp.print_result,
            num_hands=2)
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    def live_step(self, clean_frame, depth_array, depth_scale, color_intrinsics, timestamp_ms, submit, landmarker):
        """One live frame as in main.py: fingertip + depth ring
        (getFingerVector) and the live record kept with every recorded frame
        for extractSession's fallback. Returns (vect_top, tip_px, depth_info, live).
        """
        tp = self.tp
        vect_top, tip_px, depth_info = tp.getFingerVector(clean_frame, None, depth_array, depth_scale, timestamp_ms,
                                                          landmarker, color_intrinsics, self.config, submit)
        live = {
            "point": int(vect_top != (0, 0, 0)),
            "mediapipe_submitted": int(submit),
            "mediapipe_timestamp_ms": timestamp_ms if submit else None,
            "result": tp.liveLandmarks(clean_frame.shape, self.config),
            "result_age_ms": timestamp_ms - tp.liveResultTimestampMs() if tp.liveResultTimestampMs() is not None else None,
            "tip_px": tip_px,
            "valid_samples": depth_info["valid_sample_count"],
        }
        return vect_top, tip_px, depth_info, live

    def run_model(self, name, recorded, color_intrinsics, depth_scale, session_meta, timer, temp_root):
        """One model from scratch: offline tracking + crops, saveSession,
        estimateForce, Potts filter; the session data is deleted afterwards.
        Returns (DataFrame raw_frame_index, pred_n, pred_potts_n - or None if
        nothing was cropped, extraction info).
        """
        tp, entry = self.tp, self.models[name]
        config = copy.deepcopy(self.config)
        config["force_estimation"]["model"] = entry["type"]
        config["force_estimation"]["checkpoint_path"] = entry["checkpoint"]
        session_dir = tempfile.mkdtemp(prefix=f"validate_{name}_", dir=temp_root)
        try:
            with timer.stage("extract", "offline tracking + crops (extractSession)"):
                color_crops, depth_crops, pv_crops, pv_crop_meta, frame_rows, debug_rows = tp.extractSession(
                    recorded, config, color_intrinsics, depth_scale)
            info = {"frame_rows": frame_rows, "debug_rows": debug_rows, "pv_crop": pv_crop_meta}
            if not frame_rows:
                return None, info
            with timer.stage("save", "write the session (saveSession)"):
                tp.saveSession(session_dir, color_crops, depth_crops, pv_crops, pv_crop_meta, frame_rows, debug_rows,
                               session_meta)
            del color_crops, depth_crops, pv_crops
            with timer.stage("estimate", "force estimation (estimateForce)"):
                fz = tp.estimateForce(entry["net"], entry["metadata"], session_dir, frame_rows, config, self.device)
            fe_config = config["force_estimation"]
            potts = None
            if fe_config["potts_enabled"]:
                with timer.stage("potts", "Potts filter (pottsFilterL1)"):
                    potts = tp.pottsFilterL1(fz, fe_config["potts_gamma"])
            predictions = pd.DataFrame({
                "raw_frame_index": [row["raw_frame_index"] for row in frame_rows],
                "pred_n": -np.asarray(fz, dtype=float),                     # Fz (pressing negative) -> force
                "pred_potts_n": -np.asarray(potts, dtype=float) if potts is not None else np.nan})
            return predictions, info
        finally:
            shutil.rmtree(session_dir, ignore_errors=True)


def rs_intrinsics(intrinsics_dict):
    """pyrealsense2 intrinsics from a depthAlign.intrinsics_to_dict dict (for a replayed recording)."""
    intrinsics = rs.intrinsics()
    for key in ("width", "height", "ppx", "ppy", "fx", "fy"):
        setattr(intrinsics, key, intrinsics_dict[key])
    intrinsics.model = getattr(rs.distortion, intrinsics_dict["model"])
    intrinsics.coeffs = list(intrinsics_dict["coeffs"])
    return intrinsics


# ---------------------------------------------------------------- validation

class Validator:
    """Holds the models (loaded and warmed up once) and turns one recording into a report."""

    def __init__(self, settings):
        self.settings = settings
        with open(CONFIG_PATH) as f:
            self.config = yaml.safe_load(f)
        self.metric_settings = self.config["comparison"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        started = time.perf_counter()
        self.fe_processor = import_from(repo_path("preprocessing"), ["dataProcessor"]).dataProcessor
        self.regression_metrics = import_from(repo_path("Model"), ["evaluate"]).evaluate.regression_metrics
        self.trajectory = TrajectoryPipeline(settings["trajectory_pipeline_root"], settings["models"], self.device)
        self.trajectory.warm_up()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.load_s = time.perf_counter() - started
        for name, entry in self.trajectory.models.items():
            print(f"[validate] {name}: {entry['checkpoint']} (epoch {entry['epoch']})")
        print(f"[validate] models loaded and warmed up on {self.device} in {self.load_s:.1f}s")

    def labels(self, recorded, force_df):
        """Per recorded frame: time, force label (frame_force_labels: filtered
        Fz, force_n = max(-Fz, 0), artifact flag), |dF/dt| and whether the
        frame is scored (no artifact, label <= max_label_force_n).
        """
        ts_ns = np.array([frame["host_ts_ns"] for frame in recorded], dtype=np.int64)
        _, force_n, artifact = self.fe_processor.frame_force_labels(force_df, ts_ns, self.config["processing"]["force"])
        t_s = (ts_ns - ts_ns[0]) / 1e9
        rate = np.abs(np.gradient(force_n.astype(np.float64), t_s)) if len(ts_ns) > 1 else np.zeros(len(ts_ns))
        return pd.DataFrame({
            "raw_frame_index": [frame["frame_id"] for frame in recorded], "host_ts_ns": ts_ns, "t_s": t_s,
            "true_n": force_n.astype(np.float64), "artifact": artifact, "force_rate_n_per_s": rate,
            "usable": ~artifact & (force_n <= self.metric_settings["max_label_force_n"]),
        })

    def validate(self, recorded, force_df, color_intrinsics, depth_scale, session_meta, session_name, out_dir, t_stop,
                 info, show=True):
        """Runs every model from scratch on the recorded frames (3D timer from
        t_stop, the others from the start of their own data creation), then
        writes the report (and shows timeline_all.png if show). Returns
        (report path, list of problems).
        """
        results, errors = {}, []
        for k, name in enumerate(self.trajectory.models):
            timer = Timer(t_stop if k == 0 else time.perf_counter())
            print(f"[validate] {name}: creating its data and estimating the force")
            try:
                predictions, extraction = self.trajectory.run_model(name, recorded, color_intrinsics, depth_scale,
                                                                    session_meta, timer, self.settings["temp_root"])
            except Exception as e:
                traceback.print_exc()
                errors.append(f"{name}: {e!r}")
                continue
            if predictions is None:
                errors.append(f"{name}: no valid fingertip frames found offline, no force estimated")
            results[name] = {"timer": timer, "predictions": predictions, "extraction": extraction}
        truth = self.labels(recorded, force_df)
        os.makedirs(out_dir, exist_ok=True)
        report = Report(self, session_name, out_dir, truth, force_df, results, info, errors)
        path = report.write()
        report.print_mae()
        if show:
            show_timeline(out_dir)
        return path, errors


def show_timeline(out_dir):
    """Opens the session's timeline_all.png in the 'Validation results' window."""
    image = cv2.imread(os.path.join(out_dir, "timeline_all.png"))
    if image is not None:
        cv2.imshow("Validation results", image)
        cv2.waitKey(1)


# ---------------------------------------------------------------- metrics + report

def series_metrics(pred, true, rate, metric_settings, regression_metrics):
    """Input: predicted / true force (N) and |dF/dt| (N/s) on the scored
    frames, config['comparison'], the projects' regression_metrics. Returns
    the metrics of compareModels.ipynb plus bias, max error and peak force.
    """
    if len(pred) == 0:
        return {"num_frames": 0}
    thr = metric_settings["contact_threshold_n"]
    error = pred - true
    abs_error = np.abs(error)
    pred_contact, true_contact = pred > thr, true > thr
    tp = np.sum(pred_contact & true_contact)
    precision = tp / max(pred_contact.sum(), 1)
    recall = tp / max(true_contact.sum(), 1)
    dynamic = rate > metric_settings["dynamic_force_rate_n_per_s"]
    no_contact = ~true_contact

    def masked_mae(mask):
        return float(abs_error[mask].mean()) if mask.any() else np.nan

    metrics = {
        **regression_metrics(pred, true, thr),
        "bias_n": float(error.mean()),
        "max_abs_error_n": float(abs_error.max()),
        "mae_no_contact_n": masked_mae(no_contact),
        "no_contact_mean_pred_n": float(pred[no_contact].mean()) if no_contact.any() else np.nan,
        "no_contact_false_force_rate": float(pred_contact[no_contact].mean()) if no_contact.any() else np.nan,
        "contact_precision": float(precision),
        "contact_recall": float(recall),
        "contact_f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "mae_dynamic_n": masked_mae(dynamic),
        "mae_static_n": masked_mae(~dynamic),
        "peak_true_n": float(true.max()),
        "peak_pred_n": float(pred.max()),
    }
    edges = metric_settings["force_bins_n"]
    for low, high in zip(edges[:-1], edges[1:]):
        metrics[f"mae_bin_{low:g}-{high:g}N"] = masked_mae((true > low) & (true <= high))
    return metrics


def _fmt(value, digits=3, signed=False):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


class Report:
    """Builds report.md, the plots and predictions.csv for one test."""

    METRIC_ROWS = [
        ("frames scored", "num_frames", 0, False), ("MAE (N)", "mae_n", 3, False), ("RMSE (N)", "rmse_n", 3, False),
        ("R²", "r2", 3, False), ("bias (N)", "bias_n", 3, True), ("max abs error (N)", "max_abs_error_n", 2, False),
        ("MAE contact (N)", "mae_contact_n", 3, False), ("MAE no contact (N)", "mae_no_contact_n", 3, False),
        ("mean prediction, no contact (N)", "no_contact_mean_pred_n", 3, False),
        ("false-force rate, no contact", "no_contact_false_force_rate", 3, False),
        ("contact accuracy", "contact_acc", 3, False), ("contact precision", "contact_precision", 3, False),
        ("contact recall", "contact_recall", 3, False), ("contact F1", "contact_f1", 3, False),
        ("MAE dynamic (N)", "mae_dynamic_n", 3, False), ("MAE static (N)", "mae_static_n", 3, False),
        ("peak force measured (N)", "peak_true_n", 2, False), ("peak force estimated (N)", "peak_pred_n", 2, False),
    ]

    def __init__(self, validator, session_name, out_dir, truth, force_df, results, info, errors):
        self.v = validator
        self.session_name, self.out_dir = session_name, out_dir
        self.truth, self.force_df, self.results, self.info, self.errors = truth, force_df, results, info, errors
        self.ms = validator.metric_settings
        self.frame_dt_s = float(np.median(np.diff(truth["t_s"]))) if len(truth) > 1 else np.nan
        self.predictions = {name: r["predictions"] for name, r in results.items() if r["predictions"] is not None}
        self.has_potts = any(df["pred_potts_n"].notna().any() for df in self.predictions.values())
        self.colors = {name: f"C{i}" for i, name in enumerate(validator.trajectory.models)}

    def scored(self, df):
        """Joins predictions with the labels and keeps the scored frames."""
        joined = df.merge(self.truth, on="raw_frame_index", how="inner")
        return joined[joined["usable"]]

    def metrics_for(self, df, column="pred_n", frames=None):
        joined = self.scored(df)
        if frames is not None:
            joined = joined[joined["raw_frame_index"].isin(frames)]
        return series_metrics(joined[column].to_numpy(dtype=float), joined["true_n"].to_numpy(),
                              joined["force_rate_n_per_s"].to_numpy(), self.ms, self.v.regression_metrics)

    # ---- sections

    def recording_section(self):
        truth = self.truth
        usable = truth[truth["usable"]]
        thr = self.ms["contact_threshold_n"]
        force_ts = self.force_df["host_ts_ns"].to_numpy()
        force_rate_hz = (len(force_ts) - 1) / ((force_ts[-1] - force_ts[0]) / 1e9) if len(force_ts) > 1 else np.nan
        frame_dt_ms = np.diff(truth["host_ts_ns"].to_numpy()) / 1e6
        contact = usable["true_n"] > thr
        rows = [
            ["duration", f"{truth['t_s'].iloc[-1]:.1f} s" if len(truth) else "-"],
            ["frames recorded", f"{len(truth)} (median interval {self.frame_dt_s * 1000:.1f} ms = "
                                f"{1 / self.frame_dt_s:.1f} fps; {int(np.sum(frame_dt_ms > 50))} gaps > 50 ms, "
                                f"longest {frame_dt_ms.max():.0f} ms)" if len(frame_dt_ms) else str(len(truth))],
            ["force samples", f"{len(force_ts)} ({force_rate_hz:.0f} Hz)"],
            [f"frames scored (no artifact, label <= {self.ms['max_label_force_n']:g} N)", f"{len(usable)} of {len(truth)}"],
            ["artifact frames (Fz > artifact threshold, or outside the force recording)", str(int(truth["artifact"].sum()))],
            [f"frames in contact (label > {thr:g} N)",
             f"{int(contact.sum())} ({contact.mean() * 100:.0f}% of scored)" if len(usable) else "-"],
            ["contact time", f"{contact.sum() * self.frame_dt_s:.1f} s" if len(usable) else "-"],
            ["measured force: median in contact / p95 / max",
             f"{usable['true_n'][contact].median():.2f} / {np.percentile(usable['true_n'], 95):.2f} / "
             f"{usable['true_n'].max():.2f} N" if contact.any() else "-"],
        ]
        rows += [list(row) for row in self.info.get("recording_rows", [])]
        return "## Recording\n\n" + _table(["", ""], rows)

    def tracking_section(self):
        extraction = next((r["extraction"] for r in self.results.values()), None)
        if extraction is None:
            return ""
        debug = pd.DataFrame(extraction["debug_rows"])
        frames = pd.DataFrame(extraction["frame_rows"])
        sources = debug["fingertip_source"].replace("", "none").value_counts().to_dict()
        reasons = debug.loc[debug["drop_reason"].astype(bool), "drop_reason"].value_counts().to_dict()
        pv = extraction.get("pv_crop") or {}
        lines = ["## Tracking and crops", "",
                 "Offline tracking (identical for every model; it is recomputed per model for the timing):", "",
                 f"- fingertip per frame: {', '.join(f'{k} {v}' for k, v in sorted(sources.items()))} "
                 "(offline = MediaPipe VIDEO on the full frame, live = live MediaPipe fallback, "
                 "interpolated = gap filling, none = no fingertip)",
                 f"- frames cropped: {len(frames)} of {len(debug)}; dropped: "
                 f"{', '.join(f'{k} {v}' for k, v in sorted(reasons.items())) or 'none'}"]
        if len(frames):
            lines.append(f"- median fingertip depth of the cropped frames {frames['fingertip_depth_m'].median() * 100:.1f} "
                         f"cm, crop side {frames['crop_side_px'].median():.0f} px")
        if pv:
            lines.append(f"- PressureVision crop scale: {pv['session_depth_m'] * 100:.1f} cm, {pv['crop_side_mm']:.0f} mm = "
                         f"{pv['crop_side_px']:.0f} px")
        return "\n".join(lines)

    def training_data_section(self):
        """This test next to the training sessions of the RGB-D checkpoints (data/processed/*/qc/qc.json)."""
        models = self.v.trajectory.models
        reference = next((m for m in models.values() if m["type"] != "pressurevision"), None)
        if reference is None:
            return ""
        data_root = repo_path(self.v.config["processing"]["output_root"])
        depths, p95s, maxes, contacts = [], [], [], []
        for name in reference["splits"].get("train", []):
            path = os.path.join(data_root, name, "qc", "qc.json")
            if not os.path.isfile(path):
                continue
            q = _read_json(path)
            depths.append(q["median_tip_depth_m"])
            if q.get("force_n_percentiles_usable"):
                p95s.append(q["force_n_percentiles_usable"]["p95"])
                maxes.append(q["force_n_percentiles_usable"]["max"])
            if q.get("contact_pct_usable") is not None:
                contacts.append(q["contact_pct_usable"])
        if not depths:
            return ""
        usable = self.truth[self.truth["usable"]]
        thr = self.ms["contact_threshold_n"]
        extraction = next((r["extraction"] for r in self.results.values()), None)
        frame_rows = pd.DataFrame(extraction["frame_rows"]) if extraction else pd.DataFrame()
        test_depth = float(frame_rows["fingertip_depth_m"].median() * 100) if len(frame_rows) else None
        test_p95 = float(np.percentile(usable["true_n"], 95)) if len(usable) else np.nan
        test_max = float(usable["true_n"].max()) if len(usable) else np.nan
        test_contact = float((usable["true_n"] > thr).mean() * 100) if len(usable) else np.nan

        def flag(value, low, high):
            if value is None or np.isnan(value):
                return "-"
            return "inside" if low <= value <= high else "**OUTSIDE**"

        rows = [
            ["median fingertip depth (cm)", _fmt(test_depth, 1), f"{min(depths) * 100:.1f} - {max(depths) * 100:.1f}",
             flag(test_depth, min(depths) * 100, max(depths) * 100)],
            ["force p95 (N)", _fmt(test_p95, 2), f"{min(p95s):.2f} - {max(p95s):.2f}", flag(test_p95, min(p95s), max(p95s))],
            ["force max (N)", _fmt(test_max, 2), f"{min(maxes):.2f} - {max(maxes):.2f}", flag(test_max, 0, max(maxes))],
            ["frames in contact (%)", _fmt(test_contact, 0), f"{min(contacts):.0f} - {max(contacts):.0f}",
             flag(test_contact, min(contacts), max(contacts))],
        ]
        lines = ["## Compared to the training data", "",
                 f"Range over the {len(depths)} training sessions of the RGB-D checkpoints (data/processed/*/qc/qc.json). "
                 "Values outside it mean the models extrapolate. The fingertip depth here is that of the cropped "
                 "(depth-validated) frames, in training that of every tracked frame.", "",
                 _table(["", "this test", "training sessions", ""], rows)]
        seen_in = [f"{name} ({split})" for name, m in models.items() for split, names in m["splits"].items()
                   if self.session_name in names]
        if seen_in:
            lines += ["", f"**This recording was used by: {', '.join(seen_in)} - not unseen data for those models.**"]
        return "\n".join(lines)

    def coverage_section(self):
        thr = self.ms["contact_threshold_n"]
        scored_contact = self.truth[self.truth["usable"] & (self.truth["true_n"] > thr)]["raw_frame_index"]
        rows = []
        for name, df in self.predictions.items():
            joined = self.scored(df)
            covered = np.isin(scored_contact, joined["raw_frame_index"]).mean() * 100 if len(scored_contact) else np.nan
            rows.append([name, len(df), len(joined), _fmt(covered, 0)])
        return ("## Coverage\n\nTrajectoryPipeline only crops frames whose smoothed fingertip passes the depth check "
                "(mostly contact frames), so not every frame gets an estimate.\n\n"
                + _table(["model", "frames estimated", "frames scored", "contact frames covered (%)"], rows))

    def metrics_tables(self):
        series = [(name, "pred_n", name) for name in self.predictions]
        if self.has_potts:
            series += [(name, "pred_potts_n", f"{name} Potts") for name in self.predictions]
        frame_sets = [set(self.scored(df)["raw_frame_index"]) for df in self.predictions.values()]
        common = set.intersection(*frame_sets) if frame_sets else set()
        tables = [("Metrics on each model's own frames", None, "Every model on all scored frames it estimated.")]
        if any(frames != common for frames in frame_sets):     # only when the models estimated different frames
            tables.append((f"Metrics on the common frames ({len(common)} frames estimated by every model)", common,
                           "Like-for-like comparison."))
        sections = []
        for title, frames, note in tables:
            metrics = {label: self.metrics_for(self.predictions[name], column, frames) for name, column, label in series}
            header = ["metric"] + [label for _, _, label in series]
            rows = [[row_label] + [_fmt(metrics[label].get(metric), digits, signed) if digits else
                                   str(metrics[label].get(metric, 0)) for _, _, label in series]
                    for row_label, metric, digits, signed in self.METRIC_ROWS]
            bins = [k for k in next(iter(metrics.values()), {}) if k.startswith("mae_bin_")]
            rows += [[f"MAE {b[8:]} (N)"] + [_fmt(metrics[label].get(b)) for _, _, label in series] for b in bins]
            sections.append(f"## {title}\n\n{note} \"Potts\" = after TrajectoryPipeline's Potts filter (what it plots "
                            f"and sends to Unity).\n\n" + _table(header, rows))
        return "\n\n".join(sections)

    def reference_section(self):
        rows = []
        for name, entry in self.v.trajectory.models.items():
            path = os.path.join(os.path.dirname(entry["checkpoint"]), "eval_test", "metrics.json")
            test = _read_json(path)["overall"] if os.path.isfile(path) else {}
            here = self.metrics_for(self.predictions[name]) if name in self.predictions else {}
            rows.append([name, os.path.relpath(entry["checkpoint"], REPO_ROOT), entry["epoch"],
                         _fmt(test.get("mae_n")), _fmt(test.get("r2")), _fmt(here.get("mae_n")), _fmt(here.get("r2"))])
        return ("## Compared to the held-out test set\n\nTest-set numbers from the checkpoint's eval_test/metrics.json "
                "(all test sessions, the training projects' own preprocessing). A much higher MAE here points at a "
                "shift between this recording (or TrajectoryPipeline's processing) and the training data.\n\n"
                + _table(["model", "checkpoint", "epoch", "test MAE (N)", "test R²", "this test MAE (N)",
                          "this test R²"], rows))

    def timing_section(self):
        stage_keys = [("extract", "offline tracking + crops"), ("save", "write session"),
                      ("estimate", "force estimation"), ("potts", "Potts filter")]
        rows = []
        for name, result in self.results.items():
            durations = {s["key"]: s["duration_s"] for s in result["timer"].stages}
            rows.append([name, len(result["extraction"]["frame_rows"])]
                        + [_fmt(durations.get(key), 2) for key, _ in stage_keys] + [_fmt(result["timer"].total_s(), 2)])
        start = "the moment the recording had been loaded from comparison/readyForProcessing" if self.info.get("live") \
            else "the end of the replayed window"
        return ("## Timing\n\n"
                "Each model runs on its own: it creates its data from the recorded frames and estimates, and its data "
                "is deleted before the next model starts with a fresh timer. The 3D timer starts at "
                f"{start}, so its total is the time from lift to estimates; the others start when their own data "
                "creation starts (total = the time from lift to estimates if that model ran alone). Models were "
                f"loaded and warmed up once at startup ({self.v.load_s:.1f} s, not included). Device: "
                f"{torch.cuda.get_device_name(self.v.device) if self.v.device.type == 'cuda' else 'CPU'}.\n\n"
                + _table(["model", "frames cropped"] + [label + " (s)" for _, label in stage_keys] + ["total (s)"], rows))

    # ---- plots

    def _plot_series(self, ax, name, column, style, label):
        t = self.truth["t_s"].to_numpy()
        index_of = {raw: i for i, raw in enumerate(self.truth["raw_frame_index"])}
        df = self.predictions[name]
        full = np.full(len(t), np.nan)
        full[[index_of[raw] for raw in df["raw_frame_index"]]] = df[column].to_numpy(dtype=float)
        mae = self.metrics_for(df, column).get("mae_n")
        ax.plot(t, full, style, ms=2.5, lw=0.9, color=self.colors[name], label=f"{label} (MAE {_fmt(mae, 2)} N)")

    def _figure(self, title, filename, series):
        """One timeline: measured force + the given (name, column, style, label) series. Returns the filename."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 4))
        t = self.truth["t_s"].to_numpy()
        ax.plot(t, self.truth["true_n"], color="black", lw=1.2, label="measured")
        for start, end in offlineExtract._run_slices(self.truth["artifact"].to_numpy()):
            ax.axvspan(t[start], t[end - 1], color="0.85", lw=0)
        for name, column, style, label in series:
            self._plot_series(ax, name, column, style, label)
        ax.set_title(title + " - gaps: no estimate; grey: artifact frames (not scored)")
        ax.set_xlabel("time since recording start (s)")
        ax.set_ylabel("force (N)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", markerscale=3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, filename), dpi=110)
        plt.close(fig)
        return filename

    def plots(self):
        files = [(name, self._figure(name, f"timeline_{name}.png", [(name, "pred_n", ".-", name)]))
                 for name in self.predictions]
        if self.predictions:
            files.append(("All models", self._figure("All models", "timeline_all.png",
                                                     [(name, "pred_n", ".-", name) for name in self.predictions])))
        if self.has_potts:
            files.append(("TrajectoryPipeline's Potts filter", self._figure(
                "After TrajectoryPipeline's Potts filter (pottsFilterL1)", "timeline_potts.png",
                [(name, "pred_potts_n", "-", f"{name} Potts") for name in self.predictions])))
        return "## Force over time\n\n" + "\n\n".join(f"### {title}\n\n![{title}]({path})" for title, path in files)

    def predictions_csv(self):
        table = self.truth[["raw_frame_index", "host_ts_ns", "t_s", "true_n", "artifact", "usable",
                            "force_rate_n_per_s"]].copy()
        for name, df in self.predictions.items():
            table = table.merge(df.rename(columns={"pred_n": f"{name}_n", "pred_potts_n": f"{name}_potts_n"}),
                                on="raw_frame_index", how="left")
        table.to_csv(os.path.join(self.out_dir, "predictions.csv"), index=False)

    def print_mae(self):
        """Prints every model's MAE (raw and Potts-filtered, on its own scored frames) to the terminal."""
        print(f"[validate] MAE - {self.session_name}")
        for name in self.v.trajectory.models:
            if name not in self.predictions:
                print(f"  {name:<16} no estimates")
                continue
            df = self.predictions[name]
            raw = self.metrics_for(df)
            line = f"  {name:<16} MAE {_fmt(raw.get('mae_n'))} N ({raw.get('num_frames', 0)} frames)"
            if self.has_potts:
                line += f"   Potts MAE {_fmt(self.metrics_for(df, 'pred_potts_n').get('mae_n'))} N"
            print(line)
        for error in self.errors:
            print(f"  problem: {error}")

    def write(self):
        self.predictions_csv()
        lines = [f"# Model validation - {self.session_name}",
                 f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {self.info.get('mode', '')}. Processing: TrajectoryPipeline "
                 "(extractSession, saveSession, estimateForce, pottsFilterL1) with the ForceEstimationPipeline / "
                 "ForceEstimationPressureVision checkpoints below."]
        if self.errors:
            lines.append("**Problems:**\n\n" + "\n".join(f"- {error}" for error in self.errors))
        body = [self.recording_section(), self.tracking_section(), self.training_data_section(),
                self.coverage_section(), self.metrics_tables(), self.reference_section(), self.timing_section(),
                self.plots(),
                "## Notes\n\n"
                "- Labels: force_n = max(-Fz, 0) of the low-passed force, resampled to the frame timestamps "
                "(preprocessing/dataProcessor.py). Scored frames: no artifact, label <= "
                f"{self.ms['max_label_force_n']:g} N; contact = label > {self.ms['contact_threshold_n']:g} N; "
                f"dynamic = |dF/dt| > {self.ms['dynamic_force_rate_n_per_s']:g} N/s.\n"
                "- The 3D model's clips are built from consecutive cropped frames (padded by repetition at the "
                "edges), without the training window rules.\n"
                "- predictions.csv: per recorded frame the label, flags and every estimate (NaN = no estimate)."]
        lines += [section for section in body if section]
        path = os.path.join(self.out_dir, "report.md")
        with open(path, "w") as f:
            f.write("\n\n".join(lines) + "\n")
        return path


# ---------------------------------------------------------------- capture

def _trail_color(force_n, max_force_n):
    fraction = float(np.clip(force_n / max_force_n, 0.0, 1.0))
    return 0, int(255 * (1 - fraction)), int(255 * fraction)


def _show_status(image, text):
    """Large centred status text (countdown, processing)."""
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)[0]
    origin = ((image.shape[1] - size[0]) // 2, (image.shape[0] + size[1]) // 2)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)


def save_screen_frames(screen_frames, out_dir):
    """Writes the (ns since recording start, preview) frames as out_dir/screen_<s>s.png."""
    os.makedirs(out_dir, exist_ok=True)
    for offset_ns, image in screen_frames:
        cv2.imwrite(os.path.join(out_dir, f"screen_{offset_ns / 1e9:06.2f}s.png"), image)
    print(f"[validate] {len(screen_frames)} screen frames written to {out_dir}")


# ---------------------------------------------------------------- readyForProcessing

RECORDING_FILE = "recording.pkl"      # written last: a folder without it is an unfinished save


def save_recording(ready_dir, recorded, force_samples, color_intrinsics, depth_scale, session_meta, session_name, info):
    """Writes one recording losslessly to ready_dir: color.npy and depth.npy
    (all frames stacked, uncompressed so the save is quick) and recording.pkl
    (per-frame timestamps + live results, force samples, intrinsics, depth
    scale, session meta, report info).
    """
    os.makedirs(ready_dir, exist_ok=True)
    for key, filename in (("color_bgr", "color.npy"), ("depth", "depth.npy")):
        first = recorded[0][key]
        stack = np.lib.format.open_memmap(os.path.join(ready_dir, filename), mode="w+", dtype=first.dtype,
                                          shape=(len(recorded),) + first.shape)
        for i, frame in enumerate(recorded):
            stack[i] = frame[key]
        stack.flush()
        del stack
    frames = [{k: v for k, v in frame.items() if k not in ("color_bgr", "depth")} for frame in recorded]
    recording = {"session_name": session_name, "frames": frames, "force_samples": force_samples,
                 "color_intrinsics": intrinsics_to_dict(color_intrinsics), "depth_scale": depth_scale,
                 "session_meta": session_meta, "info": info}
    with open(os.path.join(ready_dir, RECORDING_FILE), "wb") as f:
        pickle.dump(recording, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_recording(ready_dir):
    """Inverse of save_recording. Returns (recorded frames as run_live keeps
    them, force DataFrame, color intrinsics, depth scale, session meta,
    session name, info).
    """
    with open(os.path.join(ready_dir, RECORDING_FILE), "rb") as f:
        recording = pickle.load(f)
    color = np.load(os.path.join(ready_dir, "color.npy"))
    depth = np.load(os.path.join(ready_dir, "depth.npy"))
    recorded = [{**frame, "color_bgr": color[i], "depth": depth[i]} for i, frame in enumerate(recording["frames"])]
    force_df = pd.DataFrame(recording["force_samples"], columns=["host_ts_ns", "fz_N"])
    return (recorded, force_df, rs_intrinsics(recording["color_intrinsics"]), recording["depth_scale"],
            recording["session_meta"], recording["session_name"], recording["info"])


def run_processing(validator, settings):
    """Processes every recording in readyForProcessing (oldest first) into
    output_root/<session>/; a recording's folder is deleted once its report
    was written without problems.
    """
    ready_root = repo_path(settings["ready_root"])
    output_root = repo_path(settings["output_root"])
    names = sorted(name for name in os.listdir(ready_root)
                   if os.path.isfile(os.path.join(ready_root, name, RECORDING_FILE))) \
        if os.path.isdir(ready_root) else []
    if not names:
        print(f"[validate] nothing to process in {ready_root}")
        return
    print(f"[validate] {len(names)} recording(s) to process")
    kept = []
    for k, name in enumerate(names, 1):
        ready_dir = os.path.join(ready_root, name)
        print(f"[validate] ({k}/{len(names)}) {name}")
        try:
            recorded, force_df, color_intrinsics, depth_scale, session_meta, session_name, info = load_recording(ready_dir)
            t_stop = time.perf_counter()
            report, errors = validator.validate(recorded, force_df, color_intrinsics, depth_scale, session_meta,
                                                session_name, os.path.join(output_root, session_name), t_stop, info,
                                                show=False)
            del recorded
            print(f"[validate] report written to {report}")
        except Exception:
            traceback.print_exc()
            errors = ["processing failed"]
        if errors:
            kept.append(name)
        else:
            shutil.rmtree(ready_dir)
    print(f"[validate] done: {len(names) - len(kept)} processed and removed from {ready_root}"
          + (f"; kept because of problems: {', '.join(kept)}" if kept else ""))


# ---------------------------------------------------------------- capture

def run_live(trajectory, settings):
    """TrajectoryPipeline's capture loop (main.py, without the ArUco markers
    and Unity) with dataAcquisition's force sensor: 'a' -> countdown ->
    recording with a fingertip trail -> finger lifted -> recording saved to
    readyForProcessing (processed later with --process).
    """
    tp, config = trajectory.tp, trajectory.config
    hand_config = config["hand_tracking"]
    lift_stop_frames = config["force_estimation"]["lift_stop_frames"]
    output_root = repo_path(settings["output_root"])
    ready_root = repo_path(settings["ready_root"])

    pipeline, pipeline_profile, depth_scale = tp.startCamera(config)
    align = rs.align(rs.stream.color)
    color_intrinsics = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    session_meta = tp.buildSessionMeta(config, pipeline_profile, depth_scale)
    acquisition_config = load_acquisition_config()
    force_reader = forceHandler.build_force_reader(acquisition_config)
    force_reader.start()
    force_calibrator = forceHandler.build_force_calibrator(acquisition_config)
    landmarker = trajectory.live_landmarker()

    state, record_at, status = "idle", 0.0, "a: start a test"
    recorded, force_samples, trail = [], [], None
    screen_frames, last_screen_ts_ns = [], None    # displayed preview while recording, 1 per second
    has_touched, lift_count, last_touch_ts_ns = False, 0, None
    loop_index, latest_force_fz = 0, 0.0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            timestamp = int(time.perf_counter() * 1000)
            if not color_frame:
                continue
            loop_index += 1
            host_ts_ns = time.perf_counter_ns()
            frame = np.asanyarray(color_frame.get_data())
            clean_frame = frame.copy()
            depth_array = np.asanyarray(depth_frame.get_data())

            if state == "countdown" and time.perf_counter() >= record_at:
                state, recorded, force_samples = "record", [], []
                screen_frames, last_screen_ts_ns = [], None
                has_touched, lift_count, last_touch_ts_ns = False, 0, None
                print("[validate] recording - press, then lift the finger to stop")

            submit = loop_index % hand_config["live_mediapipe_every_n"] == 0
            vect_top, tip_px, depth_info, live = trajectory.live_step(clean_frame, depth_array, depth_scale,
                                                                      color_intrinsics, timestamp, submit, landmarker)
            touching = vect_top != (0, 0, 0)

            raw_force_samples = force_reader.drain()
            force_calibrator.update([fz for _, fz in raw_force_samples], depth_info["valid"])
            calibrated = [(ts, force_calibrator.apply(fz)) for ts, fz in raw_force_samples]
            latest_force_fz = calibrated[-1][1] if calibrated else latest_force_fz

            lifted = False
            if state == "record":
                force_samples.extend(calibrated)
                recorded.append({
                    "frame_id": len(recorded),
                    "color_bgr": clean_frame,
                    "depth": depth_array.copy(),
                    "host_ts_ns": host_ts_ns,
                    "color_device_ts_ms": color_frame.get_timestamp(),
                    "depth_device_ts_ms": depth_frame.get_timestamp(),
                    "live": live,
                })
                if touching:
                    has_touched, lift_count, last_touch_ts_ns = True, 0, host_ts_ns
                elif has_touched:
                    lift_count += 1
                    lifted = lift_count >= lift_stop_frames

            for sample_point in depth_info["sample_points"]:
                cv2.circle(frame, (sample_point["x"], sample_point["y"]), 3,
                           (0, 255, 0) if sample_point["valid"] else (0, 0, 255), -1)
            if trail is None or trail.shape != frame.shape:
                trail = np.zeros_like(frame)
            if state == "record" and touching and tip_px is not None:
                cv2.circle(trail, (int(tip_px[0]), int(tip_px[1])), 4,
                           _trail_color(-latest_force_fz, settings["trail_max_force_n"]), -1)
            mask = trail.any(axis=2)
            frame[mask] = trail[mask]
            preview_width = hand_config["live_preview_width"]
            preview = cv2.resize(frame, (preview_width, int(round(frame.shape[0] * preview_width / frame.shape[1]))),
                                 interpolation=cv2.INTER_AREA)
            idle_s = force_reader.seconds_since_last_packet()
            lines = [f"Fz {latest_force_fz:+.2f} N" + ("" if idle_s is not None and idle_s < 2 else "  (force sensor IDLE)"),
                     f"fingertip: {'touching' if touching else 'tracked' if tip_px else 'lost'}",
                     f"REC {len(recorded)} frames" if state == "record" else status,
                     "a: test  x: cancel  q: quit"]
            for k, line in enumerate(lines):
                cv2.putText(preview, line, (10, 20 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            if state == "countdown":
                _show_status(preview, f"{max(record_at - time.perf_counter(), 0):.1f}")
            elif state == "record" and not has_touched:
                _show_status(preview, "press")
            if lifted:
                _show_status(preview, "saving...")
            cv2.imshow("Validation", preview)
            if state == "record" and (last_screen_ts_ns is None or host_ts_ns - last_screen_ts_ns >= 1e9):
                screen_frames.append((host_ts_ns - recorded[0]["host_ts_ns"], preview.copy()))
                last_screen_ts_ns = host_ts_ns
            cv2.imshow("Force Gauge", cameraHandler.draw_force_gauge(latest_force_fz))
            key = cv2.waitKey(1) & 0xFF

            if lifted:
                session_name = "session_" + time.strftime("%Y%m%d_%H%M%S")
                frame_dt_ms = np.diff([f["host_ts_ns"] for f in recorded]) / 1e6
                info = {"live": True, "mode": "live recording",
                        "recording_rows": [["lift detection", f"{lift_stop_frames} frames without a valid fingertip, "
                                            f"{(host_ts_ns - last_touch_ts_ns) / 1e6:.0f} ms after the last valid one"],
                                           ["live loop", f"{1000 / np.median(frame_dt_ms):.1f} fps while recording"]]}
                ready_dir = os.path.join(ready_root, session_name)
                try:
                    save_started = time.perf_counter()
                    save_recording(ready_dir, recorded, force_samples, color_intrinsics, depth_scale, session_meta,
                                   session_name, info)
                    status = f"saved: {session_name}"
                    print(f"[validate] {len(recorded)} frames saved to {ready_dir} in "
                          f"{time.perf_counter() - save_started:.1f}s")
                except Exception:
                    traceback.print_exc()
                    status = "saving FAILED, see console"
                save_screen_frames(screen_frames, os.path.join(output_root, session_name, "screen_frames"))
                recorded, force_samples, screen_frames, state = [], [], [], "idle"
                continue

            if key in (ord("q"), 27):
                break
            if key == ord("a") and state == "idle":
                state, record_at, status = "countdown", time.perf_counter() + settings["countdown_s"], ""
                trail[:] = 0
            elif key == ord("x") and state != "idle":
                recorded, force_samples, state, status = [], [], "idle", "test cancelled"
    finally:
        force_reader.stop()
        landmarker.close()
        pipeline.stop()
        cv2.destroyAllWindows()


def run_replay(validator, settings, raw_dir, start_s, end_s):
    """Replays the frames of a raw recording (dataWriter.RawSessionWriter
    layout) between start_s and end_s through the same live code as
    run_live, at the recorded pace: live MediaPipe every n-th frame, depth
    ring, frames + live results kept in memory. Depth is aligned with the
    acquisition's numpy aligner (rs.align needs the camera). The 3D timer
    starts when the window has been replayed.
    """
    trajectory = validator.trajectory
    hand_config = trajectory.config["hand_tracking"]
    meta_raw = _read_json(os.path.join(raw_dir, "meta_raw.json"))
    calibration = meta_raw["camera"]["stream_calibration"]
    color_dict, depth_dict = calibration["color_intrinsics"], calibration["depth_intrinsics"]
    depth_scale = meta_raw["camera"]["depth"]["depth_scale_m"]
    width, height = color_dict["width"], color_dict["height"]
    if (width, height) != (trajectory.config["camera"]["color"]["width"], trajectory.config["camera"]["color"]["height"]):
        print(f"[validate] WARNING: recording is {width}x{height}, TrajectoryPipeline's camera config differs")
    aligner = cameraHandler.build_depth_aligner(calibration, depth_scale)
    color_intrinsics = rs_intrinsics(color_dict)
    frames_raw = pd.read_csv(os.path.join(raw_dir, "frames_raw.csv"))
    ts_ns = frames_raw["host_ts_ns"].to_numpy().astype(np.int64)
    t_rel = (ts_ns - ts_ns[0]) / 1e9
    window = np.flatnonzero((t_rel >= start_s) & (t_rel < end_s))
    if not window.size:
        raise ValueError(f"no frames between {start_s} and {end_s} s in {raw_dir}")
    first, last = int(window[0]), int(window[-1])

    color_frames = offlineExtract.read_video_frames(os.path.join(raw_dir, "color.mkv"), width, height, "rgb24", 3, np.uint8)
    depth_frames = offlineExtract.read_video_frames(os.path.join(raw_dir, "depth.mkv"), depth_dict["width"],
                                                    depth_dict["height"], "gray16le", 1, np.dtype("<u2"))
    landmarker = trajectory.live_landmarker()
    recorded, last_timestamp_ms, wall_start, behind_s = [], None, None, 0.0
    print(f"[validate] replaying frames {first}-{last} ({start_s:g}-{end_s:g} s) of {raw_dir}")
    try:
        for i, (color_rgb, depth_raw) in enumerate(zip(color_frames, depth_frames)):
            if i < first:
                continue
            if i > last:
                break
            if wall_start is None:
                wall_start = time.perf_counter()
            delay = wall_start + (ts_ns[i] - ts_ns[first]) / 1e9 - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                behind_s = max(behind_s, -delay)
            clean_frame = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            depth_array = aligner.align_roi(depth_raw, (0, 0, width, height))
            timestamp = cameraHandler.next_mediapipe_timestamp_ms(int(ts_ns[i]), last_timestamp_ms)
            last_timestamp_ms = timestamp
            submit = (i - first + 1) % hand_config["live_mediapipe_every_n"] == 0
            _, _, _, live = trajectory.live_step(clean_frame, depth_array, depth_scale, color_intrinsics, timestamp,
                                                 submit, landmarker)
            row = frames_raw.iloc[i]
            recorded.append({"frame_id": len(recorded), "color_bgr": clean_frame, "depth": depth_array,
                             "host_ts_ns": int(ts_ns[i]), "color_device_ts_ms": row["color_device_ts_ms"],
                             "depth_device_ts_ms": row["depth_device_ts_ms"], "live": live})
    finally:
        landmarker.close()
    t_stop = time.perf_counter()

    force_raw = pd.read_csv(os.path.join(raw_dir, "force_raw.csv"))
    force_df = force_raw[(force_raw["host_ts_ns"] >= ts_ns[first] - 1e9) & (force_raw["host_ts_ns"] <= ts_ns[last] + 1e9)]
    session_meta = {"timestamp_clock": meta_raw["timestamp_clock"], "camera": meta_raw["camera"], "crop": {}}
    session_name = os.path.basename(os.path.normpath(raw_dir))
    info = {"live": False,
            "mode": f"replay of {raw_dir}, {start_s:g}-{end_s:g} s (raw frames {first}-{last})",
            "recording_rows": [["replay", f"at the recorded pace, at most {behind_s * 1000:.0f} ms behind; depth aligned "
                                          "with the acquisition's numpy aligner"]]}
    out_dir = os.path.join(repo_path(settings["output_root"]), f"{session_name}_{start_s:g}-{end_s:g}s")
    report, _ = validator.validate(recorded, force_df, color_intrinsics, depth_scale, session_meta, session_name,
                                   out_dir, t_stop, info)
    print(f"[validate] report written to {report} - press any key in the results window to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--process", action="store_true",
                        help="process every recording saved in readyForProcessing instead of recording")
    parser.add_argument("--raw_session", default=None, help="replay a raw recording instead of using the camera")
    parser.add_argument("--start_s", type=float, default=0.0, help="replay: window start (s since the first frame)")
    parser.add_argument("--end_s", type=float, default=float("inf"), help="replay: window end (s)")
    args = parser.parse_args()

    with open(SETTINGS_PATH) as f:
        settings = yaml.safe_load(f)["validation"]
    if args.process:
        run_processing(Validator(settings), settings)
    elif args.raw_session:
        run_replay(Validator(settings), settings, os.path.abspath(args.raw_session), args.start_s, args.end_s)
    else:      # capture only: no models needed
        run_live(TrajectoryPipeline(settings["trajectory_pipeline_root"], {}, torch.device("cpu")), settings)


if __name__ == "__main__":
    main()
