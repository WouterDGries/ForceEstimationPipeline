"""ATI NetFT UDP force sensor reader and gravity-offset calibration for the
data-collection recorder. Only Fz (downward force) is used; Fx/Fy/Tx/Ty/Tz
are unpacked (to get the byte offsets right) and dropped. All tunable
values come from the config dict loaded from config.yaml.
"""

import queue
import socket
import struct
import threading
import time

_CMD_START_STREAM = struct.pack(">HHI", 0x1234, 0x0002, 0)  # header, cmd=start stream, count=infinite
_CMD_STOP_STREAM = struct.pack(">HHI", 0x1234, 0x0000, 0)   # header, cmd=stop stream


class ForceSensorReader:
    """Background-thread UDP reader for the ATI NetFT RDT streaming
    protocol. Runs continuously regardless of recording state; call drain()
    to collect the Fz samples that arrived since the last call.
    """

    def __init__(self, ip, port, counts_per_newton, send_keepalive, keepalive_interval_s):
        self._addr = (ip, port)
        self._counts_per_n = counts_per_newton
        self._send_keepalive = send_keepalive
        self._keepalive_interval_s = keepalive_interval_s
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.5)
        self._sample_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = None
        self.last_packet_monotonic = None  # time.monotonic() of the last packet actually received

    def start(self):
        """Sends the RDT 'start stream' command and launches the reader
        thread. Returns nothing.
        """
        self._sock.sendto(_CMD_START_STREAM, self._addr)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def seconds_since_last_packet(self):
        """Returns seconds since the last packet was received, or None if
        no packet has ever arrived yet.
        """
        if self.last_packet_monotonic is None:
            return None
        return time.monotonic() - self.last_packet_monotonic

    def _run(self):
        last_keepalive = time.monotonic()
        while not self._stop_event.is_set():
            if self._send_keepalive and time.monotonic() - last_keepalive >= self._keepalive_interval_s:
                try:
                    self._sock.sendto(_CMD_START_STREAM, self._addr)
                except OSError:
                    pass
                last_keepalive = time.monotonic()

            try:
                data, _ = self._sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError as e:
                # e.g. WinError 10054 / ConnectionResetError: a well-known Windows Winsock quirk
                # where a UDP socket raises this after an earlier packet triggered an ICMP "port
                # unreachable". MUST NOT let this silently kill the reader thread - any error
                # other than socket.timeout escaping this loop kills the thread silently, which
                # looks exactly like "force stuck at 0 forever".
                print(f"[ForceSensorReader] socket error (ignored, still reading): {e!r}")
                continue

            self.last_packet_monotonic = time.monotonic()
            host_ts_ns = time.perf_counter_ns()
            if len(data) >= 36:
                # header (I), status (I), then 6 int32 counts: Fx, Fy, Fz, Tx, Ty, Tz - only Fz is kept
                _, _, fz, _, _, _ = struct.unpack(">iiiiii", data[12:36])
                self._sample_queue.put((host_ts_ns, fz / self._counts_per_n))

    def drain(self):
        """Non-blocking. Returns and removes every (host_ts_ns, fz_N) sample
        buffered since the last call, as a list.
        """
        samples = []
        while True:
            try:
                samples.append(self._sample_queue.get_nowait())
            except queue.Empty:
                break
        return samples

    def stop(self):
        """Stops the reader thread, sends the RDT 'stop stream' command, and
        closes the socket. Returns nothing.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self._sock.sendto(_CMD_STOP_STREAM, self._addr)
        except OSError:
            pass
        self._sock.close()


class ForceCalibrator:
    """Continuously re-baselines Fz to the mean of the samples seen over each
    interval_s window, so gravity's constant pull on the arm is subtracted
    out - this runs during recording too, not just while idle. Recalibration
    is suspended only while the fingertip is validated as touching the
    surface (the depth-validation ring's 5-of-8 rule), so an intentional
    push isn't zeroed out.
    """

    def __init__(self, interval_s):
        self._interval_s = interval_s
        self._offset_n = 0.0
        self._window_samples = []
        self._window_start = None
        self._was_touching = False

    def update(self, raw_fz_samples, fingertip_touching):
        """Input: list of raw Fz floats that arrived since the last call,
        and whether the fingertip is currently validated as touching (i.e.
        fingertip_depth_info["valid"]). Recalibrates the zero offset once a
        fresh, uninterrupted interval_s window has been seen. Returns
        nothing.
        """
        if fingertip_touching:
            self._window_samples.clear()
            self._window_start = None
        else:
            if self._was_touching or self._window_start is None:
                self._window_samples.clear()
                self._window_start = time.monotonic()

            self._window_samples.extend(raw_fz_samples)
            if time.monotonic() - self._window_start >= self._interval_s and self._window_samples:
                self._offset_n = sum(self._window_samples) / len(self._window_samples)
                self._window_samples.clear()
                self._window_start = time.monotonic()

        self._was_touching = fingertip_touching

    def apply(self, raw_fz):
        """Input: one raw Fz float. Returns it with the current gravity
        offset subtracted.
        """
        return raw_fz - self._offset_n


def build_force_reader(config):
    """Input: full config dict (uses config['force_sensor']). Returns a new
    ForceSensorReader (not yet started).
    """
    force_config = config["force_sensor"]
    return ForceSensorReader(
        force_config["ip"], force_config["port"], force_config["counts_per_newton"],
        force_config["send_keepalive"], force_config["keepalive_interval_s"],
    )


def build_force_calibrator(config):
    """Input: full config dict (uses config['force_sensor']). Returns a new
    ForceCalibrator.
    """
    return ForceCalibrator(config["force_sensor"]["calibration_interval_s"])


class ForceHistogram:
    """Tracks how the |Fz| samples seen during the current recording are
    spread across NUM_BINS fixed-width magnitude bins (0-2N, 2-4N, ...,
    18-20N), so the live preview can show the operator which force ranges
    still need more presses for a balanced dataset. Samples above the top
    bin's upper edge fall into that top bin. Call reset() when a new
    recording starts; counts otherwise persist (including after recording
    stops) so the final balance stays visible until the next take.
    """

    NUM_BINS = 10
    BIN_WIDTH_N = 2.0

    def __init__(self):
        self.bin_counts = [0] * self.NUM_BINS
        self.total_count = 0

    def reset(self):
        """Clears all counts, e.g. at the start of a new recording. Returns nothing."""
        self.bin_counts = [0] * self.NUM_BINS
        self.total_count = 0

    def add_samples(self, fz_values):
        """Input: iterable of calibrated Fz floats. Bins each by |Fz| and
        updates the running counts. Returns nothing.
        """
        for fz in fz_values:
            bin_index = min(int(abs(fz) // self.BIN_WIDTH_N), self.NUM_BINS - 1)
            self.bin_counts[bin_index] += 1
            self.total_count += 1

    def percentages(self):
        """Returns a list of NUM_BINS floats: each bin's share of
        total_count as a percentage (0-100), or all zeros if empty.
        """
        if self.total_count == 0:
            return [0.0] * self.NUM_BINS
        return [count / self.total_count * 100.0 for count in self.bin_counts]
