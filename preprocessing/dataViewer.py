"""Review tool for the processed sessions written by dataProcessor.py - the
RGB-D counterpart of ForceEstimationPressureVision's preprocessing/dataViewer.py.

For every saved frame it shows, side by side:
  * the raw full video frame with the crop box and fingertip drawn on it
    (decoded from the raw color.mkv the session was made from),
  * the saved color crop exactly as stored (crops.h5 "color"),
  * the saved depth crop (crops.h5 "depth_raw" or "depth_filled"),
    colorized around the frame's depth reference, holes in black,
  * the network input as Model/dataset.py (3D) or Model2D/dataset.py (2D)
    produces it for the window whose label frame is this frame - optionally
    with the random training augmentation - color de-normalized and the
    relative-depth channel, with the tensor's statistics and label. For the
    3D model a strip of the clip's frames is shown underneath,
and below that the session's force: filtered Fz (original sign, pressing is
negative), raw Fz, and the label force_n with frames marked by whether a
dataset window uses them as label frame; the current window is shaded. A
slider (or clicking the plot, or the arrow keys) moves through the saved
frames; the session box switches sessions and shows which split each
session falls in with the current config.

Read-only: nothing on disk is changed.

Run with the trajPipeline conda env, from the repo root:
    ~/anaconda3/envs/trajPipeline/bin/python preprocessing/dataViewer.py
    ~/anaconda3/envs/trajPipeline/bin/python preprocessing/dataViewer.py data/processed/session_20261002_143513
"""

import argparse
import importlib.util
import json
import os
import sys

import cv2
import h5py
import numpy as np
import pandas as pd
import pyqtgraph as pg
from matplotlib import colormaps
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSlider, QVBoxLayout, QWidget,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name, path):
    """Both models' input pipelines are called dataset.py, so they are loaded
    under their own names instead of via sys.path.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset3d = _load_module("dataset3d", os.path.join(REPO_ROOT, "Model", "dataset.py"))
dataset2d = _load_module("dataset2d", os.path.join(REPO_ROOT, "Model2D", "dataset.py"))

# Model name -> (dataset module, config section, Dataset class).
MODELS = {
    "3D": (dataset3d, "model_training", dataset3d.ForceClipDataset),
    "2D": (dataset2d, "model2d_training", dataset2d.ForceFrameDataset),
}

FULL_FRAME_WIDTH_PX = 560          # on-screen width of the full-frame panel
CROP_DISPLAY_PX = 240              # on-screen size of the crop / network-input panels
STRIP_THUMB_PX = 64                # on-screen size of one clip-strip thumbnail
STRIP_EVERY_N = 2                  # every n-th clip frame is shown in the strip
FULL_FRAME_DEBOUNCE_MS = 120       # video seeking is slow; only decode once the slider settles
RAW_FORCE_PLOT_EVERY_N = 5         # raw Fz (~1 kHz) is decimated for plotting
DEPTH_CMAP = colormaps["turbo"]
REL_DEPTH_CMAP = colormaps["coolwarm"]   # same map as Model*/dataset.py's batch preview


def rgb_to_pixmap(rgb, width, height=None):
    """Input: (H, W, 3) uint8 RGB, target on-screen width (and optional
    height). Returns a QPixmap scaled to fit, aspect ratio kept.
    """
    rgb = np.ascontiguousarray(rgb)
    image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy()).scaled(width, height or width, Qt.KeepAspectRatio,
                                                  Qt.FastTransformation)


def colorize(values, cmap, vmin, vmax, invalid=None):
    """Input: (H, W) float array, matplotlib colormap, display range and an
    optional (H, W) bool mask drawn black. Returns (H, W, 3) uint8 RGB.
    """
    normalized = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    rgb = (cmap(normalized)[..., :3] * 255).astype(np.uint8)
    if invalid is not None:
        rgb[invalid] = 0
    return rgb


def network_color_to_rgb(color_chw, rgb_mean, rgb_std):
    """Input: (3, H, W) normalized color as the network receives it. Returns
    (H, W, 3) uint8 RGB, de-normalized for display.
    """
    rgb = np.clip(color_chw.transpose(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return (rgb * 255).astype(np.uint8)


class SessionData:
    """One processed session: frames.csv, qc/frames_all.csv, meta.json,
    crops.h5 (via Model/dataset.py's _SessionData - the same arrays both
    datasets read), the raw depth crops, the raw force and video (if the raw
    session is still on disk) and each model's view of which frames are
    used as label frames.
    """

    def __init__(self, session_dir, config, splits):
        self.session_dir = session_dir
        self.name = os.path.basename(session_dir)
        with open(os.path.join(session_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.frames = pd.read_csv(os.path.join(session_dir, "frames.csv"))
        self.all_frames = pd.read_csv(os.path.join(session_dir, "qc", "frames_all.csv"))
        self.t0 = int(self.all_frames["host_ts_ns"].iloc[0])
        self.frame_time_s = (self.frames["host_ts_ns"].to_numpy() - self.t0) / 1e9

        # _SessionData only depends on the window block's max_span_s; the two
        # sessions classes are identical, so one copy serves both datasets.
        self.data = dataset3d._SessionData(session_dir, config["model_training"])
        with h5py.File(os.path.join(session_dir, "crops.h5"), "r") as h5f:
            self.depth_raw = h5f["depth_raw"][:]
        self.depth_scale_mm = self.data.depth_scale_mm

        # Per model: is frame i the label frame of a window the dataset keeps?
        # Same rules as dataset._build_window_index: start = i - label_offset,
        # window inside the session, passes window_reject_reason, start on the stride.
        self.usage = {}
        for model_name, (module, section, _) in MODELS.items():
            model_config = config[section]
            window_config = model_config["window"]
            split = splits.get(model_name, {}).get(self.name, "val")
            stride = window_config["train_stride"] if split == "train" else window_config["eval_stride"]
            offset = module.label_frame_offset(window_config["length"])
            reasons = []
            for index in range(self.num_frames()):
                start = index - offset
                if start < 0 or start + window_config["length"] > self.num_frames():
                    reasons.append("edge")
                else:
                    reasons.append(module.window_reject_reason(self.data, start, window_config, offset))
            reasons = np.array(reasons, dtype=object)
            valid = np.array([reason is None for reason in reasons])
            on_stride = (np.arange(self.num_frames()) - offset) % stride == 0
            self.usage[model_name] = {
                "split": split, "stride": stride, "offset": offset, "window": window_config,
                "reasons": reasons, "valid": valid, "used": valid & on_stride,
            }

        self.raw_dir = self.meta.get("raw_session_dir")
        if not (self.raw_dir and os.path.isdir(self.raw_dir)):
            self.raw_dir = os.path.join(REPO_ROOT, config["processing"]["raw_root"], self.name)
        self.raw_force = None
        force_path = os.path.join(self.raw_dir, "force_raw.csv")
        if os.path.exists(force_path):
            raw = pd.read_csv(force_path).iloc[::RAW_FORCE_PLOT_EVERY_N]
            self.raw_force = ((raw["host_ts_ns"].to_numpy() - self.t0) / 1e9, raw["fz_N"].to_numpy())
        video_path = os.path.join(self.raw_dir, "color.mkv")
        self.video = cv2.VideoCapture(video_path) if os.path.exists(video_path) else None
        self._next_video_frame = None

    def num_frames(self):
        return len(self.frames)

    def nearest_frame_index(self, time_s):
        index = int(np.clip(np.searchsorted(self.frame_time_s, time_s), 0, self.num_frames() - 1))
        if index > 0 and abs(self.frame_time_s[index - 1] - time_s) < abs(self.frame_time_s[index] - time_s):
            return index - 1
        return index

    def full_frame_rgb(self, raw_frame_index):
        """Returns the raw color frame (RGB) or None if the raw video is not
        available. Reads sequentially when possible, seeks otherwise.
        """
        if self.video is None:
            return None
        if raw_frame_index != self._next_video_frame:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, raw_frame_index)
        ok, bgr = self.video.read()
        self._next_video_frame = raw_frame_index + 1 if ok else None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None

    def usage_text(self, model_name, index):
        usage = self.usage[model_name]
        window = usage["window"]
        reason = usage["reasons"][index]
        if reason == "edge":
            return "<b style='color:#c00'>EXCLUDED</b> (window runs past the session start/end)"
        if reason == "span":
            return f"<b style='color:#c00'>EXCLUDED</b> (window span > {window['max_span_s']} s: tracking gap)"
        if reason == "artifact":
            return "<b style='color:#c00'>EXCLUDED</b> (force artifact in window: arm lifted / outside force recording)"
        if reason == "label":
            return f"<b style='color:#c00'>EXCLUDED</b> (label > max_label_force_n = {window['max_label_force_n']} N)"
        if reason == "change":
            return f"<b style='color:#c00'>EXCLUDED</b> (|dF| over window > {window['max_force_change_n']} N)"
        if usage["used"][index]:
            return f"<b style='color:#080'>USED</b> ({usage['split']}, stride {usage['stride']})"
        return f"valid, skipped by {usage['split']} stride {usage['stride']}"

    def close(self):
        if self.video is not None:
            self.video.release()


class ImagePanel(QWidget):
    def __init__(self, title, width, height):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(width, height)
        self.width_px, self.height_px = width, height
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label)

    def set_rgb(self, rgb):
        self.image_label.setPixmap(rgb_to_pixmap(rgb, self.width_px, self.height_px))

    def set_message(self, text):
        self.image_label.clear()
        self.image_label.setText(text)


class ViewerWindow(QMainWindow):
    def __init__(self, config, session_dirs, splits, start_session_dir):
        super().__init__()
        self.config = config
        self.session_dirs = session_dirs
        self.splits = splits
        self.session = None
        self.input_datasets = {}

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        top.addWidget(QLabel("Session:"))
        self.session_box = QComboBox()
        for session_dir in session_dirs:
            name = os.path.basename(session_dir)
            split_3d = splits["3D"].get(name, "not in split")
            split_2d = splits["2D"].get(name, "not in split")
            label = split_3d if split_3d == split_2d else f"3D {split_3d} / 2D {split_2d}"
            self.session_box.addItem(f"{name}   [{label}]", session_dir)
        top.addWidget(self.session_box, 1)
        top.addWidget(QLabel("Model:"))
        self.model_box = QComboBox()
        self.model_box.addItems(list(MODELS))
        self.model_box.currentTextChanged.connect(self._on_model_changed)
        top.addWidget(self.model_box)
        root.addLayout(top)

        self.info_label = QLabel()
        self.info_label.setTextFormat(Qt.RichText)
        root.addWidget(self.info_label)

        images = QHBoxLayout()
        self.full_panel = ImagePanel("Raw frame + crop box (green) + fingertip (red)",
                                     FULL_FRAME_WIDTH_PX, int(FULL_FRAME_WIDTH_PX * 9 / 16))
        images.addWidget(self.full_panel)

        self.crop_panel = ImagePanel("Saved color crop (crops.h5)", CROP_DISPLAY_PX, CROP_DISPLAY_PX)
        images.addWidget(self.crop_panel)

        depth_column = QVBoxLayout()
        self.depth_panel = ImagePanel("Saved depth crop", CROP_DISPLAY_PX, CROP_DISPLAY_PX)
        depth_column.addWidget(self.depth_panel)
        self.depth_box = QComboBox()
        self.depth_box.addItems(["depth_filled (hole-filled)", "depth_raw (holes black)"])
        self.depth_box.currentIndexChanged.connect(lambda _: self._show_depth())
        depth_column.addWidget(self.depth_box)
        self.depth_stats_label = QLabel()
        self.depth_stats_label.setStyleSheet("font-family: monospace;")
        depth_column.addWidget(self.depth_stats_label)
        depth_column.addStretch(1)
        images.addLayout(depth_column)

        input_column = QVBoxLayout()
        input_images = QHBoxLayout()
        self.input_color_panel = ImagePanel("Network color (de-normalized)", CROP_DISPLAY_PX, CROP_DISPLAY_PX)
        self.input_depth_panel = ImagePanel("Network relative depth", CROP_DISPLAY_PX, CROP_DISPLAY_PX)
        input_images.addWidget(self.input_color_panel)
        input_images.addWidget(self.input_depth_panel)
        input_column.addLayout(input_images)
        augment_row = QHBoxLayout()
        self.augment_box = QCheckBox("training augmentation")
        self.augment_box.toggled.connect(lambda _: self._show_network_input())
        self.redraw_button = QPushButton("re-draw")
        self.redraw_button.clicked.connect(self._show_network_input)
        augment_row.addWidget(self.augment_box)
        augment_row.addWidget(self.redraw_button)
        augment_row.addStretch(1)
        input_column.addLayout(augment_row)
        self.input_stats_label = QLabel()
        self.input_stats_label.setStyleSheet("font-family: monospace;")
        input_column.addWidget(self.input_stats_label)
        input_column.addStretch(1)
        images.addLayout(input_column)
        root.addLayout(images)

        # 3D only: every STRIP_EVERY_N-th frame of the clip, color over relative depth.
        self.strip_title = QLabel()
        root.addWidget(self.strip_title)
        strip = QHBoxLayout()
        self.strip_labels = []
        for _ in range(0, max(cfg["window"]["length"] for cfg in
                              (config[section] for _, section, _ in MODELS.values())), STRIP_EVERY_N):
            column = QVBoxLayout()
            column.setSpacing(1)
            labels = []
            for _ in range(2):
                label = QLabel()
                label.setFixedSize(STRIP_THUMB_PX, STRIP_THUMB_PX)
                column.addWidget(label)
                labels.append(label)
            caption = QLabel()
            caption.setAlignment(Qt.AlignCenter)
            column.addWidget(caption)
            self.strip_labels.append((labels[0], labels[1], caption))
            strip.addLayout(column)
        strip.addStretch(1)
        root.addLayout(strip)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Force", units="N")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend(offset=(10, 10))
        self.plot_widget.setMinimumHeight(240)
        root.addWidget(self.plot_widget)
        self.click_proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseClicked, slot=self._on_plot_clicked)

        controls = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setFocusPolicy(Qt.NoFocus)
        self.slider.valueChanged.connect(self._show_frame)
        controls.addWidget(self.slider, 1)
        self.used_only_box = QCheckBox("step through used frames only")
        controls.addWidget(self.used_only_box)
        root.addLayout(controls)

        self.full_frame_timer = QTimer(self)
        self.full_frame_timer.setSingleShot(True)
        self.full_frame_timer.timeout.connect(self._show_full_frame)

        # Arrow keys step frames wherever the keyboard focus is.
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self._step(1))
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self._step(-1))

        start = session_dirs.index(start_session_dir) if start_session_dir in session_dirs else 0
        self.session_box.setCurrentIndex(start)
        self.session_box.currentIndexChanged.connect(self._on_session_changed)
        self._on_session_changed(start)
        self.resize(1500, 1000)

    def model_name(self):
        return self.model_box.currentText()

    # ---- session / plot ----
    def _on_session_changed(self, combo_index):
        if self.session is not None:
            self.session.close()
        session_dir = self.session_box.itemData(combo_index)
        name = os.path.basename(session_dir)
        self.session = SessionData(session_dir, self.config,
                                   {model: self.splits[model] for model in MODELS})
        self.input_datasets = {model: self._make_input_dataset(model) for model in MODELS}
        self.setWindowTitle(f"Processed Data Viewer - {name}")
        self._build_plot()
        self.slider.blockSignals(True)
        self.slider.setRange(0, self.session.num_frames() - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._show_frame(0)

    def _make_input_dataset(self, model_name):
        """Returns the model's Dataset object over this session only, used
        for its __getitem__ (window -> augment -> normalization): the
        attributes its __init__ sets are filled in directly, so no other
        session has to be loaded. windows holds one entry, set per frame.
        """
        module, section, dataset_class = MODELS[model_name]
        model_config = self.config[section]
        window_config = model_config["window"]
        input_dataset = dataset_class.__new__(dataset_class)
        input_dataset.split = "val"
        input_dataset.window_config = window_config
        input_dataset.window_length = window_config["length"]
        input_dataset.label_offset = module.label_frame_offset(window_config["length"])
        input_dataset.clip_mm = model_config["depth"]["clip_mm"]
        input_dataset.force_norm_scale_n = model_config["label"]["force_norm_scale_n"]
        input_dataset.augmentation_config = model_config["augmentation"]
        input_dataset.sessions = [self.session.data]
        input_dataset.windows = []
        return input_dataset

    def _on_model_changed(self, _):
        if self.session is None:
            return
        self._build_plot()
        self._show_frame(self.slider.value())

    def _build_plot(self):
        session = self.session
        usage = session.usage[self.model_name()]
        plot = self.plot_widget
        plot.clear()
        all_t = (session.all_frames["host_ts_ns"].to_numpy() - session.t0) / 1e9
        if session.raw_force is not None:
            plot.plot(*session.raw_force, pen=pg.mkPen((200, 200, 200), width=1), name="raw Fz (sensor sign)")
        plot.plot(all_t, session.all_frames["fz_filtered_n"].to_numpy(), pen=pg.mkPen((90, 90, 90), width=1),
                  name="filtered Fz (sensor sign)")

        force_n = session.frames["force_n"].to_numpy()
        valid_label = np.where(usage["valid"], force_n, np.nan)
        plot.plot(session.frame_time_s, valid_label, pen=pg.mkPen((30, 100, 220), width=2), connect="finite",
                  name=f"label force_n (valid {self.model_name()} label frames)")
        excluded = ~usage["valid"]
        if excluded.any():
            plot.plot(session.frame_time_s[excluded], force_n[excluded], pen=None,
                      symbol="o", symbolSize=3, symbolBrush=(220, 30, 30), symbolPen=None,
                      name="saved but excluded")
        not_saved = session.all_frames["saved"].to_numpy() == 0
        if not_saved.any():
            plot.plot(all_t[not_saved], np.full(not_saved.sum(), -1.0), pen=None, symbol="s", symbolSize=2,
                      symbolBrush=(240, 150, 0), symbolPen=None, name="no crop (hand not tracked), at -1 N")
        artifact_n = self.config["processing"]["force"]["artifact_positive_n"]
        plot.addItem(pg.InfiniteLine(pos=artifact_n, angle=0, pen=pg.mkPen((220, 30, 30), style=Qt.DashLine)))
        plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("k", width=1)))
        self.window_region = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(255, 0, 0, 30),
                                                 pen=pg.mkPen(None))
        plot.addItem(self.window_region)
        self.cursor_line = pg.InfiniteLine(pos=session.frame_time_s[0], angle=90, pen=pg.mkPen("r", width=2))
        plot.addItem(self.cursor_line)

    def _on_plot_clicked(self, event):
        mouse_event = event[0]
        if mouse_event.button() != Qt.LeftButton:
            return
        point = self.plot_widget.plotItem.vb.mapSceneToView(mouse_event.scenePos())
        self.slider.setValue(self.session.nearest_frame_index(point.x()))

    def _step(self, step):
        """Moves one saved frame forward/back, or to the next/previous used
        label frame when "used frames only" is ticked.
        """
        index = self.slider.value() + step
        if self.used_only_box.isChecked():
            used = np.flatnonzero(self.session.usage[self.model_name()]["used"])
            candidates = used[used > self.slider.value()] if step > 0 else used[used < self.slider.value()][::-1]
            index = int(candidates[0]) if candidates.size else self.slider.value()
        self.slider.setValue(int(np.clip(index, 0, self.session.num_frames() - 1)))

    # ---- frame display ----
    def _show_frame(self, index):
        session = self.session
        row = session.frames.iloc[index]
        usage = session.usage[self.model_name()]
        self.cursor_line.setPos(session.frame_time_s[index])
        start = int(np.clip(index - usage["offset"], 0, session.num_frames() - 1))
        end = int(np.clip(start + usage["window"]["length"] - 1, 0, session.num_frames() - 1))
        self.window_region.setRegion((session.frame_time_s[start], session.frame_time_s[end]))

        self.crop_panel.set_rgb(session.data.color[index])
        self._show_depth()
        self._show_network_input()
        self.full_frame_timer.start(FULL_FRAME_DEBOUNCE_MS)

        flags = []
        if row["interpolated"]:
            flags.append("fingertip interpolated")
        if not row["detected"]:
            flags.append("not detected this frame")
        if not row["fingertip_depth_valid"]:
            flags.append("fingertip depth check failed")
        if row["artifact"]:
            flags.append("<b style='color:#c00'>force artifact</b>")
        self.info_label.setText(
            f"Frame {index + 1}/{session.num_frames()} (raw #{int(row['raw_frame_index'])})   "
            f"t = {session.frame_time_s[index]:.2f} s   "
            f"Fz = {row['fz_filtered_n']:+.2f} N  &rarr;  label <b>{row['force_n']:.2f} N</b>   "
            f"{self.model_name()}: {session.usage_text(self.model_name(), index)}   "
            f"split: <b>{usage['split']}</b><br>"
            f"crop centre ({row['crop_cx']:.0f}, {row['crop_cy']:.0f}) px, side {row['crop_side_px']:.0f} px "
            f"= {self.config['processing']['crop']['side_mm']:.0f} mm, angle {row['crop_angle_deg']:.0f} deg, "
            f"tip depth {row['tip_depth_m'] * 1000:.0f} mm, "
            f"outside image {max(row['outside_fraction'], 0.0) * 100:.0f}%   "
            f"{', '.join(flags)}")

    def _show_depth(self):
        session = self.session
        index = self.slider.value()
        row = session.frames.iloc[index]
        raw_mm = session.depth_raw[index].astype(np.float32) * session.depth_scale_mm
        holes = session.depth_raw[index] == 0
        if self.depth_box.currentIndex() == 0:
            depth_mm, invalid = session.data.depth_mm(index), None
        else:
            depth_mm, invalid = raw_mm, holes
        clip_mm = self.config[MODELS[self.model_name()][1]]["depth"]["clip_mm"]
        reference = float(row["depth_ref_mm"])
        self.depth_panel.set_rgb(colorize(depth_mm, DEPTH_CMAP, reference - clip_mm, reference + clip_mm, invalid))
        valid_mm = depth_mm[~holes] if (~holes).any() else depth_mm
        self.depth_stats_label.setText(
            f"ref (median valid) {reference:.1f} mm\n"
            f"color range ref +/- {clip_mm:.0f} mm\n"
            f"valid range [{valid_mm.min():.0f}, {valid_mm.max():.0f}] mm\n"
            f"holes {holes.mean() * 100:.1f}% (csv {row['depth_invalid_fraction'] * 100:.1f}%)")

    def _show_network_input(self):
        index = self.slider.value()
        model_name = self.model_name()
        module = MODELS[model_name][0]
        usage = self.session.usage[model_name]
        input_dataset = self.input_datasets[model_name]
        augment = self.augment_box.isChecked()
        self.redraw_button.setEnabled(augment)
        # The datasets augment exactly when split == "train".
        input_dataset.split = "train" if augment else "val"
        start = index - usage["offset"]
        if usage["reasons"][index] == "edge":
            for panel in (self.input_color_panel, self.input_depth_panel):
                panel.set_message("no full window\naround this frame")
            self.input_stats_label.setText("")
            self._clear_strip()
            return
        input_dataset.windows = [(0, start)]
        try:
            sample = input_dataset[0]
        except Exception as error:
            self.input_color_panel.set_message(f"dataset error:\n{error}")
            self.input_depth_panel.set_message("")
            self._clear_strip()
            return

        tensor = (sample["clip"] if model_name == "3D" else sample["image"]).numpy()
        # (4, T, H, W) for 3D, (4, H, W) for 2D - show the label frame.
        label_frame = tensor[:, input_dataset.label_offset] if tensor.ndim == 4 else tensor
        self.input_color_panel.set_rgb(network_color_to_rgb(label_frame[:3], module.RGB_MEAN, module.RGB_STD))
        self.input_depth_panel.set_rgb(colorize(label_frame[3], REL_DEPTH_CMAP, -1, 1))

        axes = tuple(range(1, tensor.ndim))
        means = ", ".join(f"{v:+.2f}" for v in tensor.mean(axis=axes))
        stds = ", ".join(f"{v:.2f}" for v in tensor.std(axis=axes))
        used_start = int(sample["start_frame"])
        shift = used_start - start
        self.input_stats_label.setText(
            f"tensor {tuple(tensor.shape)} {tensor.dtype}\n"
            f"range [{tensor.min():+.2f}, {tensor.max():+.2f}]\n"
            f"ch mean [{means}]\nch std  [{stds}]\n"
            f"label {sample['label_n'].item():.2f} N "
            f"(normalized {sample['label_normalized'].item():.3f})\n"
            f"window frames {used_start}..{used_start + input_dataset.window_length - 1}"
            f"{f' (temporal shift {shift:+d})' if shift else ''}\n"
            f"depth: blue = closer, red = farther (+/-1 = {input_dataset.clip_mm:.0f} mm)")

        if tensor.ndim == 4:
            self._show_strip(tensor, module, used_start, input_dataset.label_offset)
        else:
            self._clear_strip("2D model: only the label frame reaches the network "
                              "(the window only decides which label frames are kept)")

    def _show_strip(self, clip, module, start, label_offset):
        self.strip_title.setText(f"3D clip frames 0, {STRIP_EVERY_N}, {2 * STRIP_EVERY_N}, ...: color (top) / relative depth (bottom); "
                                 f"label frame = offset {label_offset}")
        num_frames = clip.shape[1]
        for slot, (color_label, depth_label, caption) in enumerate(self.strip_labels):
            t = slot * STRIP_EVERY_N
            if t >= num_frames:
                color_label.clear(), depth_label.clear(), caption.clear()
                continue
            color = network_color_to_rgb(clip[:3, t], module.RGB_MEAN, module.RGB_STD)
            depth = colorize(clip[3, t], REL_DEPTH_CMAP, -1, 1)
            color_label.setPixmap(rgb_to_pixmap(color, STRIP_THUMB_PX))
            depth_label.setPixmap(rgb_to_pixmap(depth, STRIP_THUMB_PX))
            force = self.session.data.frame_force_n[start + t]
            caption.setText(f"<b>{t}</b><br>{force:.1f}N" if t == label_offset else f"{t}<br>{force:.1f}N")

    def _clear_strip(self, message=""):
        self.strip_title.setText(message)
        for color_label, depth_label, caption in self.strip_labels:
            color_label.clear(), depth_label.clear(), caption.clear()

    def _show_full_frame(self):
        session = self.session
        row = session.frames.iloc[self.slider.value()]
        frame = session.full_frame_rgb(int(row["raw_frame_index"]))
        if frame is None:
            self.full_panel.set_message("raw color.mkv not available")
            return
        frame = frame.copy()
        # Same corner mapping as offlineExtract.crop_frame: centre + R(angle) * (+/-half, +/-half).
        half = row["crop_side_px"] / 2
        angle = np.deg2rad(row["crop_angle_deg"])
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]]) @ rotation.T
        corners += [row["crop_cx"], row["crop_cy"]]
        cv2.polylines(frame, [np.round(corners).astype(np.int32)], True, (0, 255, 0), 2)
        cv2.circle(frame, (int(round(row["tip_x"])), int(round(row["tip_y"]))), 5, (255, 0, 0), -1)
        self.full_panel.set_rgb(frame)

    def closeEvent(self, event):
        if self.session is not None:
            self.session.close()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", nargs="?", default=None, help="processed session to open first")
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)
    config = dataset3d.load_config()
    model_config = config["model_training"]
    data_root = os.path.join(REPO_ROOT, model_config["data_root"])
    session_dirs = [os.path.join(data_root, name) for name in dataset3d.discover_sessions(model_config)]
    if not session_dirs:
        QMessageBox.critical(None, "No data", f"No processed sessions (with PROCESSED_OK) in {data_root}")
        return
    splits = {}
    for model_name, (module, section, _) in MODELS.items():
        try:
            splits[model_name] = {name: split for split, names in module.resolve_splits(config[section]).items()
                                  for name in names}
        except ValueError as error:   # too few sessions for a split yet
            print(f"[viewer] {model_name}: {error}")
            splits[model_name] = {}

    start = os.path.abspath(args.session_dir) if args.session_dir else None
    window = ViewerWindow(config, session_dirs, splits, start)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
