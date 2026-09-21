"""Depth preprocessing for the fingertip-contact network's depth input.

Raw depth crops in video.h5 are uint16 sensor counts (see meta.json's
camera.depth.depth_scale_m for the meters-per-count factor), with a
meaningful fraction of zero pixels where the sensor couldn't get a valid
reading (holes, out-of-range, low-reflectivity surfaces). Before a depth
crop reaches the network it must be:

  1. converted to meters using depth_scale_m,
  2. have invalid (raw == 0) pixels masked out,
  3. normalized on a fixed physical scale, so the same real-world distance
     always maps to the same network input value across frames and sessions
     (a per-frame min/max normalization would make e.g. 5mm mean something
     different in every frame), and
  4. have pixels farther than MAX_DEPTH_M zeroed out, since the fingertip
     contact region this network cares about is always within a few
     centimeters of the camera.

Invalid and out-of-range pixels are unified into a single "no reading" value
(0.0 in both the metric and normalized output), which is unambiguous because
a real surface can never be exactly at the camera's focal point (raw == 0).
"""

import numpy as np

MAX_DEPTH_M = 0.30  # pixels farther than this aren't part of the fingertip contact region


def preprocess_depth(depth_raw, depth_scale_m, max_depth_m=MAX_DEPTH_M):
    """Input: raw uint16 depth crop(s) - any shape, e.g. a single (H, W) crop
    or a batch (N, H, W) - and the session's camera.depth.depth_scale_m from
    meta.json. Returns (normalized, valid_mask):

      normalized - float32 array, same shape as depth_raw, in [0, 1].
                   0.0 means "no valid in-range reading"; a value v > 0
                   means the surface was at v * max_depth_m meters.
      valid_mask - bool array, same shape, True where depth_raw held a
                   genuine in-range reading (i.e. where normalized > 0).
    """
    depth_m = depth_raw.astype(np.float32) * depth_scale_m
    valid_mask = (depth_raw != 0) & (depth_m <= max_depth_m)
    depth_m = np.where(valid_mask, depth_m, 0.0)
    normalized = depth_m / max_depth_m
    return normalized, valid_mask
