"""Evaluates a trained checkpoint on a split - by default the held-out test
sessions, densely (every window start, eval stride 1) - in the same output
format as ForceEstimationPressureVision's Model/evaluate.py, so
comparison/compareModels.ipynb can match the predictions of all three models
frame by frame (session + raw_frame_index).

The same file lives in Model/ (3D, clips) and Model2D/ (2D, single frames);
it works with whichever dataset/model/train modules sit next to it.

Writes to <checkpoint_dir>/eval_<split>/:
    metrics.json            overall + per-session metrics, trivial baseline
    predictions.csv         one row per scored window: session, raw_frame_index (of the label frame), time, true / predicted force
    timeline_<session>.png  measured vs predicted force over time

Usage (trajPipeline env, from the repo root):
    python Model/evaluate.py                                    # best_model.pt on the test split
    python Model/evaluate.py --split val --checkpoint Model/checkpoints/seed_7/best_model.pt
    python Model2D/evaluate.py                                  # same for the 2D model
"""

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import dataset
import model as model_module
import train

CONFIG_SECTION = getattr(dataset, "CONFIG_SECTION", "model_training")
CONTACT_THRESHOLD_N = 0.5   # same as PressureVision's training.contact_threshold_n


def regression_metrics(pred_n, true_n, contact_threshold_n=CONTACT_THRESHOLD_N):
    """Input: 1D numpy arrays of predicted / true force (N) and the contact
    threshold. Returns a dict: mae, rmse, r2 over all frames, mae_contact
    over frames whose true force exceeds the threshold, and contact_acc
    (agreement of pred > threshold with true > threshold). Identical to
    ForceEstimationPressureVision's train.regression_metrics.
    """
    error = pred_n - true_n
    ss_tot = np.sum((true_n - true_n.mean()) ** 2)
    contact = true_n > contact_threshold_n
    return {
        "mae_n": float(np.mean(np.abs(error))),
        "rmse_n": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1.0 - np.sum(error ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "mae_contact_n": float(np.mean(np.abs(error[contact]))) if contact.any() else float("nan"),
        "contact_acc": float(np.mean((pred_n > contact_threshold_n) == contact)),
        "num_frames": int(len(true_n)),
    }


def mean_train_force_n(model_config, train_sessions):
    """Mean label over every usable training frame (no artifact, label <=
    max_label_force_n) - the trivial baseline, defined like PressureVision's
    (mean of its usable train frames at stride 1).
    """
    data_root = os.path.join(dataset.REPO_ROOT, model_config["data_root"])
    labels = []
    for name in train_sessions:
        frames = pd.read_csv(os.path.join(data_root, name, "frames.csv"))
        usable = (frames["artifact"] == 0) & (frames["force_n"] <= model_config["window"]["max_label_force_n"])
        labels.append(frames.loc[usable, "force_n"].to_numpy())
    return float(np.concatenate(labels).mean())


def _timeline_plots(predictions, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for session, df in predictions.groupby("session"):
        t = (df["host_ts_ns"] - df["host_ts_ns"].min()) / 1e9
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t, df["true_n"], ".", ms=1.5, label="measured")
        ax.plot(t, df["pred_n"], ".", ms=1.5, label="predicted")
        mae = np.mean(np.abs(df["pred_n"] - df["true_n"]))
        ax.set_title(f"{session}  MAE {mae:.2f} N")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("force (N)")
        ax.legend(markerscale=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"timeline_{session}.png"), dpi=90)
        plt.close(fig)


def predict(net, data, config, device):
    """Input: model, a dataset (eval mode, no augmentation), the full config
    dict, device. Runs every window in data.windows (in order) through the
    model. Returns a DataFrame with one row per window: session,
    raw_frame_index and host_ts_ns of the label frame, true_n, pred_n.
    """
    model_config = config[CONFIG_SECTION]
    loader = DataLoader(data, batch_size=model_config["loader"]["batch_size"], shuffle=False,
                        num_workers=model_config["loader"]["num_workers"], pin_memory=True)
    autocast_dtype, _ = train.select_precision(device)
    force_norm_scale_n = model_config["label"]["force_norm_scale_n"]
    preds = []
    net.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["clip" if "clip" in batch else "image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                prediction = net(inputs).squeeze(1).float()
            prediction = train._clamp_predictions(prediction, data.clamp_sign)
            preds.append(prediction.cpu().numpy() * force_norm_scale_n)
    rows = []
    for session_index, start in data.windows:
        session = data.sessions[session_index]
        label_index = start + data.label_offset
        rows.append((session.name, int(session.raw_frame_index[label_index]), int(session.frame_ts_ns[label_index]),
                     float(session.frame_force_n[label_index])))
    predictions = pd.DataFrame(rows, columns=["session", "raw_frame_index", "host_ts_ns", "true_n"])
    predictions["pred_n"] = np.concatenate(preds) if preds else np.array([])
    return predictions


def evaluate(checkpoint_path=None, split="test", all_windows=False, config=None):
    """Input: checkpoint path (default <checkpoint_dir>/best_model.pt), split
    name, all_windows (True = score every window start, ignoring the window
    rules), optional config dict. Uses the split and window rules stored in
    the checkpoint, so later config/data changes can't silently change what
    is scored. Writes the outputs listed in the module docstring. Returns
    the metrics dict.
    """
    config = copy.deepcopy(config or dataset.load_config())
    model_config = config[CONFIG_SECTION]
    checkpoint_path = checkpoint_path or os.path.join(
        dataset.REPO_ROOT, model_config["training"]["checkpoint_dir"], "best_model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata = checkpoint["metadata"]

    splits = checkpoint.get("splits") or dataset.resolve_splits(model_config)
    model_config["splits"] = {f"{name}_sessions": sessions for name, sessions in splits.items()}
    if "window_rules" in metadata:
        model_config["window"] = dict(metadata["window_rules"])
    if "data_root" in metadata:
        model_config["data_root"] = metadata["data_root"]
    model_config["window"]["eval_stride"] = 1

    out_dir = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), f"eval_{split}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[evaluate] {checkpoint_path} (epoch {checkpoint['epoch']}) on {split}: {splits[split]}")

    data = dataset.build_dataset(split, config) if split != "train" else None
    if data is None:   # "train" without augmentation / temporal jitter: build as an eval split
        model_config["splits"]["test_sessions"] = splits["train"]
        data = dataset.build_dataset("test", config)
    if all_windows:
        data.windows = [(session_index, start) for session_index, session in enumerate(data.sessions)
                        for start in range(session.num_frames - data.window_length + 1)]
    data.clamp_sign = metadata["prediction_clamp_sign"]

    net = model_module.build_model(mean_normalized_label=0.0).to(device)
    net.load_state_dict(checkpoint["model_state_dict"])
    predictions = predict(net, data, config, device)
    predictions.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    pred_n, true_n = predictions["pred_n"].to_numpy(), predictions["true_n"].to_numpy()
    mean_train_n = mean_train_force_n(model_config, splits["train"])
    metrics = {
        "model": f"{CONFIG_SECTION} ({metadata.get('backbone', 'r2plus1d_18')})",
        "checkpoint": os.path.abspath(checkpoint_path),
        "epoch": checkpoint["epoch"],
        "split": split,
        "all_windows": all_windows,
        "window_rules": model_config["window"],
        "overall": regression_metrics(pred_n, true_n),
        "trivial_baseline_mean_train_force": regression_metrics(np.full(len(true_n), mean_train_n), true_n),
        "per_session": {session: regression_metrics(df["pred_n"].to_numpy(), df["true_n"].to_numpy())
                        for session, df in predictions.groupby("session")},
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _timeline_plots(predictions, out_dir)

    overall, trivial = metrics["overall"], metrics["trivial_baseline_mean_train_force"]
    print(f"[evaluate] {split}: MAE {overall['mae_n']:.3f} N  RMSE {overall['rmse_n']:.3f} N  R2 {overall['r2']:.3f}  "
          f"contact-MAE {overall['mae_contact_n']:.3f} N  contact-acc {overall['contact_acc']:.3f}  "
          f"(trivial baseline MAE {trivial['mae_n']:.3f} N, {overall['num_frames']} frames)")
    for session, session_metrics in metrics["per_session"].items():
        print(f"  [evaluate] {session}: MAE {session_metrics['mae_n']:.3f} N  R2 {session_metrics['r2']:.3f}")
    print(f"[evaluate] Outputs written to {out_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None, help="default: <checkpoint_dir>/best_model.pt")
    parser.add_argument("--split", default="test", choices=dataset.SPLIT_NAMES)
    parser.add_argument("--all_windows", action="store_true", help="score every window start, ignoring the window rules")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.split, args.all_windows)
