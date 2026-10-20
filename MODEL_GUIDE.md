# Force-estimation models and training data

This document is the implementation reference for the three models used in
this repository's comparison:

1. **3D RGB-D model** in `Model/`
2. **2D RGB-D model** in `Model2D/`
3. **PressureVision force model** in the neighboring
   `../ForceEstimationPressureVision/Model/` project

The names “3D” and “2D” refer to the convolution type and input time
dimension. The first two estimate one scalar force in newtons per input
sample. PressureVision internally estimates a dense pressure map, then
integrates it to produce one scalar force. All three are compared against the
same measured normal-force labels, but their input crops and preprocessing
differ.

## At a glance

| Model | Network input | Prediction | Starting weights | Training target |
|---|---|---|---|---|
| 3D RGB-D | 20-frame clip, `(B, 4, 20, 112, 112)` | One normalized force per clip; report in N | TorchVision R(2+1)D-18, Kinetics-400 | Centre-frame measured force |
| 2D RGB-D | One frame, `(B, 4, 112, 112)` | One normalized force per frame; report in N | TorchVision ResNet-18, ImageNet | Same centre-frame force and selected frames as 3D |
| PressureVision force model | One RGB crop, normally `(B, 3, 256, 256)`, plus crop pixel area | Dense 9-class pressure logits/map and integrated force in N | Official PressureVisionNet checkpoint (`paper_59.pt`) by default | Measured scalar force; no pixelwise ground-truth pressure map |

For the 3D and 2D models, input channels are RGB followed by relative depth.
For PressureVision the input is RGB only.

## Shared force labels and split policy

### Force label

The acquisition rig records the normal force component `Fz` in newtons.
Pressing is negative on this sensor. Preprocessing low-pass filters `Fz` with
a zero-phase, order-4 Butterworth filter (12 Hz in the checked-in configs),
interpolates it to camera-frame timestamps, and defines the training label as:

```text
force_n = max(-Fz_filtered, 0)
```

Thus zero means no measured press and positive values are press force in N.
Frames during strong positive `Fz` (typically the forearm being lifted off
the plate) and a 0.5 s margin around those events are marked as artifacts.
Frames outside the force recording are also artifacts. Artifact frames are
saved for provenance/QC, but excluded from model samples.

### Splits and leakage prevention

The standard split is 70% train, 15% validation, and 15% test, with split
seed 0. Sessions—not individual frames or windows—are assigned to splits.
This prevents nearly identical adjacent frames or overlapping clips from
crossing between train and test. Processed sessions are discoverable only
when their `PROCESSED_OK` marker exists. The resolved session lists are
written to `splits.json` and stored in the model checkpoints.

The two RGB-D model configs are intended to remain aligned on the same
sessions, windows, label definition, force threshold, and depth transform.
The PressureVision project uses the same session-based fraction/seed policy,
but its own preprocessed crop files and frame-selection configuration.

## 1. 3D RGB-D clip model

### Architecture and contract

Implementation: `Model/model.py`; dataset: `Model/dataset.py`; training:
`Model/train.py`; entry point: `Model/mainModel.py`.

- Backbone: TorchVision `r2plus1d_18` pretrained on Kinetics-400.
- Input stem is changed from three to four channels. The pretrained RGB
  kernels are copied unchanged; the new depth kernel starts as the mean of
  the RGB kernels.
- Replaces the original classification layer with a regression head:
  dropout (0.4) → linear (512 to 128) → GELU → dropout (0.4) → linear
  (128 to 1).
- The final layer is initialized to predict the mean normalized training
  force.
- `layer1` and `layer2` are frozen, including their BatchNorm running
  statistics. The stem, `layer3`, `layer4`, and regression head are trained.
- Output is one normalized force per clip. Convert to N by multiplying by
  `force_norm_scale_n` (20 N in the checked-in config). Evaluation clamps
  predictions to the physically valid side of zero.

### Sample construction

A sample starts as a sequence of 20 consecutive saved frames. Its label is
the force at local offset `20 // 2 = 10` (the 11th frame in zero-based
counting). Each candidate window is rejected if:

- its first-to-last timestamp span exceeds 0.8 s;
- any frame in it is marked as a force artifact; or
- its centre-frame force exceeds 30 N.

`max_force_change_n` is currently null, so no force-change filter is applied.
The training index uses window-start stride 3; validation and test use stride
1. The 20 frames inside each accepted clip are consecutive saved samples.

Each frame contributes:

- RGB: `uint8` RGB converted to `[0, 1]`, then normalized with Kinetics-400
  channel means `[0.43216, 0.394666, 0.37645]` and standard deviations
  `[0.22803, 0.22145, 0.216989]`;
- relative depth: hole-filled depth in mm minus the depth reference from the
  clip's centre frame, clipped to `[-40, 40]` mm and divided by 40. Positive
  means farther from the camera than the reference; negative means closer.

The resulting per-sample tensor is `float32`, `(4, T, H, W) =
(4, 20, 112, 112)`. In a batch it is `(B, 4, 20, 112, 112)`.

### Training

Run from the repository root with the project's configured Python
environment:

```bash
python Model/mainModel.py
```

The default seed is 42; `--seed N` selects a separate
`Model/checkpoints/seed_N/` output folder. Training uses:

- AdamW with separate learning rates: head `1e-3`, `layer4` `1e-4`,
  `layer3` `3e-5`, stem `1e-4`; weight decay `1e-3` except biases and
  normalization parameters;
- Huber loss with delta 1.5 N, converted to normalized-label units;
- linear warmup for 1.5 epochs followed by cosine decay, stepped per optimizer
  update;
- batch size and effective batch size 16, gradient norm clipping at 1.0;
- up to 60 epochs, stopping after 10 epochs without improved validation MAE;
- horizontal flips with probability 0.5 and temporal-offset augmentation
  (try a shifted window of up to ±2 frames, keeping its label synchronized).
  Rotation and depth noise are disabled by default.

On supported NVIDIA GPUs, autocast uses bfloat16 on Ampere-or-newer hardware
and float16 with gradient scaling on older hardware. CPU training uses
float32. The best checkpoint is selected by validation MAE in N and contains
weights, epoch, validation MAE, resolved splits, and inference/preprocessing
metadata. Unless `--skip_test` is supplied, the entry point evaluates it on
the held-out test sessions.

## 2. 2D RGB-D single-frame model

### Architecture and contract

Implementation: `Model2D/model.py`; dataset: `Model2D/dataset.py`; training:
`Model2D/train.py`; entry point: `Model2D/mainModel.py`.

- Backbone: TorchVision ResNet-18 pretrained on ImageNet.
- ResNet `conv1` is adapted to four input channels by copying its RGB kernels
  and initializing the depth kernel to their channel mean.
- Uses the same 512→128 regression head, dropout (0.4), GELU, and mean-label
  output-bias initialization as the 3D model.
- Freezes `layer1` and `layer2`, including BatchNorm statistics. The stem,
  `layer3`, `layer4`, and head are trainable.
- Input is one RGB+relative-depth frame `(B, 4, 112, 112)`. Output is a
  normalized scalar force; multiply by 20 N in the current config to recover
  N.

### Sample construction

It uses the same 20-frame candidate windows, filters, split logic, and
centre-frame labels as the 3D model. The network receives only the centre
frame, not the other 19 frames. The window exists to choose and filter label
frames consistently with the 3D model. Its four-channel layout,
relative-depth definition, crop size, and label normalization are aligned
with the 3D dataset. RGB normalization is the deliberate exception: 2D uses
ImageNet means/stds, while 3D uses Kinetics-400 means/stds, matching each
backbone's pretraining.

This makes 2D versus 3D a comparison of single-frame versus temporal visual
context, rather than a change in target or sample-selection policy. Keep the
two `model_training` and `model2d_training` sections in `config.yaml` aligned
when comparing them.

### Training

Run from the repository root:

```bash
python Model2D/mainModel.py
```

Training settings intentionally mirror the 3D run: seed 42, batch/effective
batch size 16, up to 60 epochs, early-stopping patience 10, Huber delta 1.5 N,
weight decay `1e-3`, gradient clipping at 1.0, 1.5-epoch linear warmup then
cosine decay, and the same horizontal flip and temporal-offset augmentation.
The per-module learning rates are head `1e-3`, `layer4` `1e-4`, `layer3`
`3e-5`, and stem (`conv1` + `bn1`) `1e-4`. The best checkpoint is chosen by
validation MAE and test evaluation runs unless `--skip_test` is passed.

## 3. PressureVision force model

### Which PressureVision implementation this means

The model used in the comparison is the force-supervised adaptation in
`../ForceEstimationPressureVision/Model/`, not the original PressureVision
paper training loop as-is. It reuses the upstream PressureVision FPN and its
9-class pressure head, then converts that head to total force by integrating
the predicted pressure over the known physical crop area.

The upstream source is vendored/configured under
`../ForceEstimationPressureVision/PressureVision/`. Its original task predicts
pressure maps from full RGB camera views and pixelwise pressure-sensor maps.
The adaptation described here instead trains on fingertip RGB crops and only
has scalar force labels. Do not substitute the upstream `prediction/trainer.py`
recipe or its original data layout for the force-fine-tuning recipe below.

Implementation: `../ForceEstimationPressureVision/Model/model.py`; dataset:
`../ForceEstimationPressureVision/Model/dataset.py`; training:
`../ForceEstimationPressureVision/Model/train.py`; entry point:
`../ForceEstimationPressureVision/Model/mainModel.py`.

### Architecture, physical output, and initialization

- Backbone/decoder: PressureVision's `segmentation_models_pytorch` FPN with
  an SE-ResNeXt-50 32×4d encoder. It predicts 9 logits per pixel.
- Default initialization loads the official PressureVisionNet
  `paper_59.pt` checkpoint. If that file is absent, the project's downloader
  fetches it. The alternative `init_weights: imagenet` starts with an
  ImageNet-pretrained encoder and a randomly initialized FPN decoder/head.
- The PressureVision pressure classes use thresholds
  `[0, 0.5, 1, 2, 4, 8, 16, 32, 64]` kPa. The original class-to-scalar
  conversion maps class 0 to 0, intermediate classes to the midpoint of
  adjacent thresholds, and the final class to 80 kPa.
- During training, per-pixel softmax probabilities are converted to
  expected pressure; a learnable positive gain multiplies that map. The map
  is integrated over crop pixels:

  ```text
  pressure_kpa[y, x] = gain * sum_k(probability[k, y, x] * class_centre_kpa[k])
  force_n = sum_yx(pressure_kpa[y, x]) * 1000 * pixel_area_m2
  ```

- `pixel_area_m2` is supplied with each image. The model returns predicted
  force in N, a continuous pressure map in kPa, and logits. A separate
  hard/argmax pressure-map integral is reported for comparison.
- There is no per-pixel pressure ground truth in the forearm-press data. Only
  the integrated force is supervised, so the pressure map's spatial
  distribution is not directly identified by the label.
- By default encoder stages `layer0`, `layer1`, and `layer2` (including
  BatchNorm statistics) are frozen. Later encoder stages, FPN decoder,
  segmentation head, and scalar gain are trainable.

### Data and sample construction

This project has its own processed session format, generated by
`../ForceEstimationPressureVision/preprocessing/dataProcessor.py`. The
default crop is a 160 mm square resized to 256×256, approximately
0.625 mm/pixel. It is centred on the tracked fingertip. Its per-frame RGB
JPEGs are stored in `data/processed/session_*/frames/`; they are not the
112×112 RGB-D HDF5 crops consumed by `Model/` and `Model2D/`.

Each row in that project's `frames.csv` describes a saved crop, including
`raw_frame_index`, relative JPEG path (`image`), `host_ts_ns`, filtered `Fz`,
`force_n`, `artifact`, tracking/interpolation flags, fingertip/crop
coordinates, `crop_side_px`, `pixel_size_mm`, and `outside_fraction`.
`meta.json` records preprocessing/camera details; `PROCESSED_OK` marks a
complete session. Each sample is one RGB crop and its scalar `force_n`.

Rows are removed if they are artifacts or if `force_n > 30 N`. The default
training/evaluation stride is every third usable row. The loader reads RGB,
scales to `[0, 1]`, and applies PressureVision's ImageNet normalization
(means `[0.485, 0.456, 0.406]`, standard deviations
`[0.229, 0.224, 0.225]`). It also returns the per-pixel area in m². Since
`pixel_size_mm = crop_side_mm / crop_output_px`, the unaugmented value is
`(pixel_size_mm / 1000)²`. During geometric scale augmentation, pixel area
is divided by the square of the applied image scale so the physical
integral remains consistent.

Training augmentation uses horizontal flips, rotations up to ±15°, scales
from 0.85 to 1.15, translations up to 8% of image width, brightness/contrast
jitter of 0.15, Gaussian blur (probability 0.2, sigma up to 1 px), and
additive pixel noise (standard deviation 0.01).

### Training

Run from the sibling project root:

```bash
python Model/mainModel.py
```

`--seed N` and `--init_weights pressurevision|imagenet` select run-specific
checkpoint directories; `--max_epochs N` overrides the epoch limit.
Defaults in that project's `config.yaml`:

- seed 42; batch size 12; 30 epochs maximum; up to 30,000 samples per epoch;
- AdamW with encoder learning rate `5e-5`, decoder/head `2e-4`, gain `1e-2`,
  and weight decay `1e-4` (no decay for bias and normalization parameters);
- Huber loss on integrated force with delta 1 N;
- an additional location prior of weight 1.0, penalizing predicted pressure
  outside a 25 mm radius from the crop centre on frames whose true force is
  above 0.5 N. This helps address the fact that only the pressure-map
  integral has a direct label. It is not added to validation loss/MAE;
- one epoch of linear warmup followed by cosine decay, gradient clipping at
  1.0, and early stopping after six epochs without validation-MAE
  improvement.

The initial model (epoch 0) is validated and saved as a baseline, which is
zero-shot PressureVisionNet when using `paper_59.pt`. The best fine-tuned
checkpoint is chosen by validation MAE in N. The evaluation reports soft
integral force and hard/argmax-map force, as well as overall and contact
metrics.

## Processed data layouts

### RGB-D models (`ForceEstimationPipeline`)

`preprocessing/dataProcessor.py` turns raw recordings into one
`data/processed/session_<timestamp>/` directory per session:

```text
session_<timestamp>/
  crops.h5          # color: (N, 112, 112, 3) uint8 RGB
                    # depth_raw and depth_filled: (N, 112, 112) uint16 counts
  frames.csv        # one row per saved crop/frame, including timing,
                    # force_n, fz_filtered_n, artifact, depth_ref_mm,
                    # raw_frame_index, and depth quality/crop metadata
  meta.json         # camera/depth scale, crop and processing settings
  qc/               # plots, crop strip, QC metadata and complete frame table
  PROCESSED_OK      # written last; signals that processing completed
```

`depth_filled` is generated by per-frame hole filling and filtering;
multiply counts by the session depth scale to obtain mm. `depth_ref_mm` is
the median of originally valid depth pixels in that frame. All tracked and
mostly-in-bounds crops are retained, including no-contact frames. The
original depth-invalid fraction is recorded for QC; the models use the
hole-filled depth channel.

Raw acquisition consists of aligned-by-metadata RGB/depth video streams
(`color.mkv`, `depth.mkv`), `frames_raw.csv`, timestamped `force_raw.csv`,
`meta_raw.json`, and alignment-check samples. Raw RGB is encoded video; raw
depth is lossless uint16 counts. The preprocessor tracks fingertip/DIP with
MediaPipe, smooths detections and interpolates short gaps, aligns depth to
color, extracts a crop that covers 40 mm × 40 mm at the fingertip depth,
and resizes it to 112×112.

### PressureVision force fine-tuning

`../ForceEstimationPressureVision/preprocessing/dataProcessor.py` writes a
different `data/processed/session_<timestamp>/` structure:

```text
session_<timestamp>/
  frames/             # <raw_frame_index>.jpg RGB fingertip crops
  frames.csv          # image path, timing, force_n, artifact, crop geometry,
                      # pixel_size_mm, tracking/interpolation metadata
  meta.json           # session crop scale, camera and processing metadata
  qc/
  PROCESSED_OK
```

This dataset stores RGB JPEGs only; the PressureVision force fine-tuner does
not load depth. Its default crop is 160 mm and 256×256 pixels, deliberately
much larger than the 40 mm, 112×112 RGB-D crops. Both processed formats use
the same basic nonnegative `force_n` label and raw frame indexing convention,
but their image dimensions, physical fields of view, storage formats, and
input channels are not interchangeable.

## Checkpoints, evaluation, and comparisons

- RGB-D checkpoints are `best_model.pt` under `Model/checkpoints/` or
  `Model2D/checkpoints/` (or their seed subdirectories). They include a
  `model_state_dict`, epoch, validation MAE, splits, and input/label metadata.
- PressureVision fine-tuning checkpoints are under
  `../ForceEstimationPressureVision/Model/checkpoints/`; they include
  `model_state_dict`, the full config, epoch, validation metrics, and splits.
- Each model writes an `eval_test/predictions.csv` and metrics. The
  comparison script expects those predictions and checks that the
  session splits and ground-truth labels match before comparing:
  `comparison/compareModels.ipynb`.
- The common frame key is `(session, raw_frame_index)`. For a valid
  comparison, preserve shared session identities, force labels, and test
  sessions. Do not interpret equal numerical inputs or equal crop geometry:
  PressureVision's RGB crop intentionally has a different physical scale
  from the RGB-D crops.

## Important implementation constraints for future agents

1. Do not silently change the force sign, unit, temporal label alignment, or
   the per-session split policy.
2. The 3D model's label belongs to clip offset `window_length // 2`; any
   inference code must use that exact frame convention.
3. Keep 2D and 3D window/filter/split settings aligned if claiming a
   controlled temporal-context comparison.
4. Do not feed depth to PressureVision or RGB-only data to the RGB-D models
   without deliberately changing the model, preprocessing, training, and
   checkpoint metadata together.
5. PressureVision's force result depends on correct crop pixel area and the
   preprocessing scale. A pressure-map display is not a pixelwise calibrated
   pressure measurement on the forearm; it is a map whose integral is
   trained against the total force.
6. Whenever training data, split policy, normalization, crop geometry, class
   centres, or output interpretation changes, retrain or establish
   checkpoint compatibility explicitly.
