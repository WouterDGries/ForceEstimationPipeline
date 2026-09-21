"""Manual review tool for a recorded session (see Data/session_*/README.md
for the file layout). Plots the whole session's force trace with a slider
underneath; scrubbing the slider jumps to the nearest recorded frame and
shows its color crop and depth crop.

Run with the trajPipeline conda env (has PySide6/pyqtgraph/h5py):

    /home/wouterdg/anaconda3/envs/trajPipeline/bin/python dataViewer.py [session_dir]
    python DataCollection/dataViewer.py [path]
If session_dir is omitted, a folder picker opens in DataCollection/Data.

Note: this pipeline only ever saves the cropped color/depth images
(video.h5 has no full-frame dataset), so there is no "full frame" to show -
only the two crops are displayed.
"""

import argparse
import json
import os
import sys

import cv2
import h5py
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QSlider, QVBoxLayout, QWidget,
)

DEFAULT_DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
# On-screen panel size in pixels. Crops are scaled UP or DOWN to fit this, whatever
# their native resolution (112, 224, ...) - never hardcode the crop size here.
DISPLAY_SIZE_PX = 340
DEPTH_REFERENCE_FRAME_INDEX = 1127
DEPTH_RANGE_MARGIN_M = 0.02


class SessionData:
    """Loads one session_*/ directory and answers frame/force lookups."""

    def __init__(self, session_dir):
        self.session_dir = session_dir
        self.frames = pd.read_csv(os.path.join(session_dir, "frames.csv"))
        self.force = pd.read_csv(os.path.join(session_dir, "force.csv"))
        with open(os.path.join(session_dir, "meta.json")) as f:
            self.meta = json.load(f)

        self.h5 = h5py.File(os.path.join(session_dir, "video.h5"), "r")
        self.color = self.h5["color"]
        self.depth = self.h5["depth"]
        self.depth_scale_m = self.meta["camera"]["depth"]["depth_scale_m"]
        reference_depth = self.depth[DEPTH_REFERENCE_FRAME_INDEX]
        valid_reference_depth = reference_depth[reference_depth > 0] * self.depth_scale_m
        if valid_reference_depth.size == 0:
            raise ValueError(f"Reference frame {DEPTH_REFERENCE_FRAME_INDEX} has no valid depth pixels")
        self.depth_limits_m = (
            float(valid_reference_depth.min() - DEPTH_RANGE_MARGIN_M),
            float(valid_reference_depth.max() + DEPTH_RANGE_MARGIN_M),
        )

        t0 = min(self.frames["host_ts_ns"].min(), self.force["host_ts_ns"].min())
        self.t0 = int(t0)
        self.force_ts_ns = self.force["host_ts_ns"].to_numpy()
        self.force_time_s = (self.force_ts_ns - self.t0) / 1e9
        self.force_fz = self.force["fz_N"].to_numpy()
        self.frame_time_s = (self.frames["host_ts_ns"].to_numpy() - self.t0) / 1e9

    def num_frames(self):
        return len(self.frames)

    def nearest_force_index(self, frame_ts_ns):
        idx = np.searchsorted(self.force_ts_ns, frame_ts_ns)
        if idx <= 0:
            return 0
        if idx >= len(self.force_ts_ns):
            return len(self.force_ts_ns) - 1
        before, after = idx - 1, idx
        if frame_ts_ns - self.force_ts_ns[before] <= self.force_ts_ns[after] - frame_ts_ns:
            return before
        return after

    def nearest_frame_index(self, time_s):
        idx = np.searchsorted(self.frame_time_s, time_s)
        idx = min(max(idx, 0), self.num_frames() - 1)
        if idx > 0 and abs(self.frame_time_s[idx - 1] - time_s) < abs(self.frame_time_s[idx] - time_s):
            return idx - 1
        return idx

    def close(self):
        self.h5.close()


def numpy_rgb_to_pixmap(rgb_array, display_size):
    height, width, _ = rgb_array.shape
    image = QImage(np.ascontiguousarray(rgb_array).data, width, height, 3 * width, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(image)
    return pixmap.scaled(display_size, display_size, Qt.KeepAspectRatio, Qt.FastTransformation)


def depth_crop_to_pixmap(depth_u16, display_size, depth_scale_m, depth_limits_m):
    low_m, high_m = depth_limits_m
    if not np.any(depth_u16 > 0):
        normalized = np.zeros_like(depth_u16, dtype=np.uint8)
    else:
        depth_m = depth_u16.astype(np.float32) * depth_scale_m
        normalized = np.clip((depth_m - low_m) / max(high_m - low_m, 1e-6), 0, 1)
        normalized = (normalized * 255).astype(np.uint8)
        normalized[depth_u16 == 0] = 0
    colorized_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)
    return numpy_rgb_to_pixmap(colorized_rgb, display_size)


class ImagePanel(QWidget):
    def __init__(self, title):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(DISPLAY_SIZE_PX, DISPLAY_SIZE_PX)
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label)

    def set_pixmap(self, pixmap):
        self.image_label.setPixmap(pixmap)

    def set_message(self, text):
        self.image_label.clear()
        self.image_label.setText(text)


class ViewerWindow(QMainWindow):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self.setWindowTitle(f"Data Viewer - {os.path.basename(session.session_dir)}")

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        self.info_label = QLabel()
        root_layout.addWidget(self.info_label)

        images_layout = QHBoxLayout()
        self.color_panel = ImagePanel("Color crop")
        self.depth_panel = ImagePanel("Depth crop")
        images_layout.addWidget(self.color_panel)
        images_layout.addWidget(self.depth_panel)
        root_layout.addLayout(images_layout)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Fz", units="N")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.plot(session.force_time_s, session.force_fz, pen=pg.mkPen(color="b", width=1))
        self.cursor_line = pg.InfiniteLine(pos=session.frame_time_s[0], angle=90,
                                            pen=pg.mkPen(color="r", width=2))
        self.plot_widget.addItem(self.cursor_line)
        self.plot_widget.setMinimumHeight(250)
        root_layout.addWidget(self.plot_widget)

        self.click_proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseClicked,
                                           slot=self._on_plot_clicked)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(session.num_frames() - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        root_layout.addWidget(self.slider)

        self._show_frame(0)
        self.resize(900, 750)

    def _on_plot_clicked(self, event):
        mouse_event = event[0]
        view_point = self.plot_widget.plotItem.vb.mapSceneToView(mouse_event.scenePos())
        frame_index = self.session.nearest_frame_index(view_point.x())
        self.slider.setValue(frame_index)

    def _on_slider_changed(self, frame_index):
        self._show_frame(frame_index)

    def _show_frame(self, frame_index):
        session = self.session
        row = session.frames.iloc[frame_index]
        frame_ts_ns = int(row["host_ts_ns"])
        force_idx = session.nearest_force_index(frame_ts_ns)
        force_value = session.force_fz[force_idx]
        time_s = session.frame_time_s[frame_index]

        self.cursor_line.setPos(time_s)

        color_crop = session.color[frame_index]
        depth_crop = session.depth[frame_index]
        self.color_panel.set_pixmap(numpy_rgb_to_pixmap(color_crop, DISPLAY_SIZE_PX))
        self.depth_panel.set_pixmap(depth_crop_to_pixmap(
            depth_crop, DISPLAY_SIZE_PX, session.depth_scale_m, session.depth_limits_m
        ))

        detected = "yes" if row["fingertip_detected"] else "no"
        depth_valid = "yes" if row["fingertip_depth_valid"] else "no"
        fingertip_depth_m = row["fingertip_depth_m"]
        depth_text = f"{fingertip_depth_m * 100:.1f} cm" if pd.notna(fingertip_depth_m) else "n/a"

        self.info_label.setText(
            f"Frame {frame_index + 1}/{session.num_frames()}   t = {time_s:.2f} s   "
            f"Fz = {force_value:+.3f} N   "
            f"fingertip detected: {detected}   depth valid: {depth_valid} ({depth_text})   "
            f"crop origin: ({int(row['crop_x0'])}, {int(row['crop_y0'])})"
        )


def resolve_session_dir(argument):
    if argument:
        return os.path.abspath(argument)

    app = QApplication.instance() or QApplication(sys.argv)
    start_dir = DEFAULT_DATA_ROOT if os.path.isdir(DEFAULT_DATA_ROOT) else os.getcwd()
    chosen = QFileDialog.getExistingDirectory(None, "Select a session directory", start_dir)
    return chosen or None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", nargs="?", default=None,
                         help="Path to a session_* directory. If omitted, a folder picker opens.")
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)

    session_dir = resolve_session_dir(args.session_dir)
    if not session_dir:
        return

    required_files = ["frames.csv", "force.csv", "meta.json", "video.h5"]
    missing = [name for name in required_files if not os.path.exists(os.path.join(session_dir, name))]
    if missing:
        QMessageBox.critical(None, "Invalid session directory",
                              f"{session_dir} is missing: {', '.join(missing)}")
        return

    session = SessionData(session_dir)
    if session.num_frames() == 0:
        QMessageBox.warning(None, "Empty session", "This session has no recorded frames.")
        session.close()
        return

    window = ViewerWindow(session)
    window.show()
    app.aboutToQuit.connect(session.close)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
