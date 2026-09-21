# Recording session: session_20260921_134612

## Files in this session

| File | Contents |
|---|---|
| `video.h5` | datasets `color` (N,224,224,3 uint8, RGB) and `depth` (N,224,224 uint16, raw counts) |
| `frames.csv` | one row per saved video frame, including fingertip detection/depth validation measurements and crop location |
| `force.csv` | one row per force sample: `sample_index, host_ts_ns, fz_N` (zeroed against the gravity baseline) |
| `meta.json` | machine-readable copy of the settings below |

Frame count: 512  |  Force sample count: 80464

`frames.csv` and `force.csv` share the same clock (`time.perf_counter_ns`, monotonic, not wall-clock), so rows from the two files can be time-aligned by `host_ts_ns` during offline post-processing.

## Camera settings used

- Color: 1920x1080 @ 30 fps, auto_exposure=False, exposure=156, gain=64, auto_white_balance=False, white_balance=3000K, brightness=0, contrast=50
- Depth: 1280x720 @ 30 fps, depth_units=0.0001 m/count, actual depth_scale=9.999999747378752e-05 m/count, emitter_enabled=True, laser_power=150, visual_preset=high_accuracy
- Crop: 224x224 px, centered on the Left hand's MediaPipe landmark 8 (index fingertip)
- Force sensor: ATI NetFT over UDP at 192.168.1.1:49152, Fz only, counts_per_newton=1000000.0 (**placeholder value — verify against the sensor's calibration file**), auto-zeroed to the mean Fz every 3s whenever the fingertip isn't validated as touching (depth-validation 5-of-8 rule)
