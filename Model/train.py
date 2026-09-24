"""Training loop for the fingertip force estimator.

Reads config.yaml's model_training block, builds the train/val
ForceClipDataset + DataLoaders (dataset.py) and the adapted r2plus1d_18
(model.py), then trains with AdamW (per-module learning rates), a
warmup-then-cosine schedule, and HuberLoss under bf16/fp16 autocast.
Validates every epoch and checkpoints whenever validation MAE improves.
Produces a checkpoint (.pt, with everything needed to use it correctly
later - see _save_checkpoint) and a metrics-history CSV that
explore.ipynb reads. train() is called by mainModel.py; running this file
directly instead runs the Section 8 sanity checks (frozen BatchNorm,
overfit-one-batch, trivial baseline) without a full training run.
"""

import copy
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dataset
import model as model_module

METRICS_FIELDS = ["epoch", "train_loss", "val_loss", "val_mae_n", "val_rmse_n", "val_r2", "lr", "epoch_time_s"]


def _split_decay_no_decay(named_parameters):
    """Input: an iterable of (name, param) from a module's
    named_parameters(). Splits into (decay, no_decay) parameter lists -
    biases and 1D (BatchNorm/LayerNorm) parameters get no weight decay,
    everything else does. Returns (decay, no_decay).
    """
    decay, no_decay = [], []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith("bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def build_optimizer(net, optimizer_config):
    """Input: the adapted model and config['model_training']['training']
    ['optimizer']. Builds AdamW with one (decay, no_decay) parameter-group
    pair per trainable module - head/layer4/layer3/stem - at that module's
    learning rate; decay groups use weight_decay, no_decay groups use 0.
    layer1/layer2 are frozen in model.py and never appear here. Returns
    the optimizer.
    """
    weight_decay = optimizer_config["weight_decay"]
    module_lrs = [
        (net.fc, optimizer_config["head_lr"]),
        (net.layer4, optimizer_config["layer4_lr"]),
        (net.layer3, optimizer_config["layer3_lr"]),
        (net.stem, optimizer_config["stem_lr"]),
    ]
    param_groups = []
    for module, lr in module_lrs:
        decay, no_decay = _split_decay_no_decay(module.named_parameters())
        param_groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        param_groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return torch.optim.AdamW(param_groups)


def build_scheduler(optimizer, warmup_steps, total_steps):
    """Input: optimizer, warmup step count, total step count
    (max_epochs * steps_per_epoch). Returns a LambdaLR meant to be stepped
    once per optimizer step: linear warmup to each group's base LR, then
    cosine decay to 0 over the remaining steps.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def select_precision(device):
    """Input: torch.device. Returns (autocast_dtype, use_grad_scaler):
    bfloat16 with no GradScaler on Ampere-or-newer GPUs (bf16 has enough
    exponent range that overflow isn't a practical concern), float16 with
    GradScaler on older GPUs, or (None, False) - plain float32, no autocast -
    off CUDA entirely.
    """
    if device.type != "cuda":
        return None, False
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8:
        return torch.bfloat16, False
    return torch.float16, True


def _prediction_clamp_sign(train_dataset):
    """Input: the training ForceClipDataset. Returns +1.0 or -1.0. Section
    8 asks to "clamp predictions to >= 0 at evaluation time", which assumes
    a sensor convention where pressing reads positive; this rig's Fz reads
    NEGATIVE while pressing (confirmed from the data - see the training
    report), so the literal >=0 clamp would clamp every real prediction to
    0 and make the evaluation meaningless. This infers the rig's actual
    sign from the training set's mean label once, so the clamp keeps its
    intent (a prediction can't cross past the zero/no-contact baseline)
    on whichever side is physically correct here.
    """
    return 1.0 if train_dataset.mean_normalized_label() >= 0 else -1.0


def _clamp_predictions(prediction, clamp_sign):
    """Input: a prediction tensor (normalized units) and the sign from
    _prediction_clamp_sign(). Returns the prediction clamped toward zero
    from the physically-impossible side.
    """
    return prediction.clamp(min=0) if clamp_sign > 0 else prediction.clamp(max=0)


def _regression_metrics(pred_n, true_n):
    """Input: 1D numpy arrays of predicted/true force in newtons. Returns
    (mae, rmse, r2).
    """
    error = pred_n - true_n
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((true_n - true_n.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(mae), float(rmse), float(r2)


def _trivial_baseline_mae_n(train_dataset, eval_dataset):
    """Input: the training dataset (for its mean label) and any other split
    to evaluate against. Returns the MAE in newtons of always predicting
    the training split's mean force - Section 8's sanity floor that a real
    model must clear by a wide margin.
    """
    mean_train_n = train_dataset.mean_normalized_label() * train_dataset.force_norm_scale_n
    eval_labels_n = np.array([
        eval_dataset.sessions[session_index].frame_fz_n[start + eval_dataset.label_offset]
        for session_index, start in eval_dataset.windows
    ])
    return float(np.mean(np.abs(eval_labels_n - mean_train_n)))


def run_validation(net, loader, device, autocast_dtype, loss_fn, force_norm_scale_n, clamp_sign):
    """Input: model, a validation/test DataLoader, device, autocast dtype
    (or None), loss function, the label normalization scale, and the
    clamp sign from _prediction_clamp_sign(). Runs one full pass under
    torch.inference_mode, converting predictions and labels back to
    newtons. Returns (mean_loss, mae_n, rmse_n, r2).
    """
    net.eval()
    losses, preds_n, trues_n = [], [], []
    with torch.inference_mode():
        for batch in loader:
            clip = batch["clip"].to(device, non_blocking=True)
            label = batch["label_normalized"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                prediction = net(clip).squeeze(1).float()
            loss = loss_fn(prediction, label)
            losses.append(loss.item())
            preds_n.append(_clamp_predictions(prediction, clamp_sign).cpu().numpy() * force_norm_scale_n)
            trues_n.append(label.cpu().numpy() * force_norm_scale_n)
    preds_n = np.concatenate(preds_n)
    trues_n = np.concatenate(trues_n)
    mae_n, rmse_n, r2 = _regression_metrics(preds_n, trues_n)
    return float(np.mean(losses)), mae_n, rmse_n, r2


def _init_metrics_file(path):
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=METRICS_FIELDS).writeheader()


def _append_metrics_row(path, row):
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=METRICS_FIELDS).writerow(row)


def _save_checkpoint(path, net, config, epoch, val_mae_n, clamp_sign):
    """Input: destination path, the trained model, the full config dict,
    the epoch number, this checkpoint's validation MAE, and the prediction
    clamp sign. Saves the model weights plus everything needed to use the
    checkpoint correctly later (Section 6): window length, temporal
    stride, label-frame convention, force component, crop size, colour
    normalization constants, depth reference/clip definition, and the
    label normalization scale. Returns nothing.
    """
    model_config = config["model_training"]
    metadata = {
        "window_length": model_config["window"]["length"],
        "temporal_stride_within_window": 1,     # frames inside a window are always consecutive
        "label_frame_offset": dataset.label_frame_offset(model_config["window"]["length"]),
        "force_component": "Fz",                 # the only component this rig measures - see dataAcquisition/forceHandler.py
        "force_filter_cutoff_hz": model_config["force_filter"]["cutoff_hz"],
        "force_filter_order": model_config["force_filter"]["order"],
        "crop_size_px": 112,
        "crop_physical_size_mm": None,   # not tracked by this rig - fixed pixel box, physical size is distance-dependent
        "rgb_mean": dataset.RGB_MEAN.tolist(),
        "rgb_std": dataset.RGB_STD.tolist(),
        "depth_reference_definition": "median of the originally-valid depth pixels (mm) at the window's "
                                       "label (centre) frame, subtracted from every frame in the window",
        "depth_clip_mm": model_config["depth"]["clip_mm"],
        "depth_sign_convention": "positive = farther from camera than the window reference, negative = closer",
        "force_norm_scale_n": model_config["label"]["force_norm_scale_n"],
        "prediction_clamp_sign": clamp_sign,
    }
    torch.save({"model_state_dict": net.state_dict(), "epoch": epoch, "val_mae_n": val_mae_n,
                "metadata": metadata}, path)


def train(config=None):
    """Input: optionally a pre-loaded config dict (loads config.yaml if
    omitted). Runs the full training loop described in the module
    docstring. Returns the best validation MAE in newtons.
    """
    config = config or dataset.load_config()
    model_config = config["model_training"]
    training_config = model_config["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    print("[train] Building datasets")
    train_dataset, train_loader = dataset.build_dataloader("train", config)
    val_dataset, val_loader = dataset.build_dataloader("val", config)

    clamp_sign = _prediction_clamp_sign(train_dataset)
    baseline_mae_n = _trivial_baseline_mae_n(train_dataset, val_dataset)
    print(f"[train] Trivial baseline (always predict the mean train force): val MAE = {baseline_mae_n:.3f} N "
          f"- the trained model must beat this by a wide margin")

    print("[train] Building model")
    net = model_module.build_model(train_dataset.mean_normalized_label()).to(device)
    num_trainable = sum(p.numel() for p in model_module.trainable_parameters(net))
    print(f"  [train] {num_trainable:,} trainable parameters (layer1/layer2 frozen)")

    optimizer = build_optimizer(net, training_config["optimizer"])
    loader_config = model_config["loader"]
    accumulation_steps = max(1, training_config["effective_batch_size"] // loader_config["batch_size"])
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    warmup_steps = int(training_config["schedule"]["warmup_epochs"] * steps_per_epoch)
    total_steps = training_config["max_epochs"] * steps_per_epoch
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    autocast_dtype, use_grad_scaler = select_precision(device)
    scaler = torch.amp.GradScaler(device="cuda" if device.type == "cuda" else "cpu", enabled=use_grad_scaler)

    delta_normalized = training_config["loss"]["huber_delta_n"] / model_config["label"]["force_norm_scale_n"]
    loss_fn = nn.HuberLoss(delta=delta_normalized)

    checkpoint_dir = os.path.join(dataset.REPO_ROOT, training_config["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics_path = os.path.join(checkpoint_dir, training_config["metrics_history_file"])
    _init_metrics_file(metrics_path)

    best_val_mae_n = float("inf")
    epochs_without_improvement = 0

    print(f"[train] Starting training: max_epochs={training_config['max_epochs']} "
          f"batch_size={loader_config['batch_size']} effective_batch_size={training_config['effective_batch_size']} "
          f"accumulation_steps={accumulation_steps} steps_per_epoch={steps_per_epoch} "
          f"precision={autocast_dtype or torch.float32}")

    for epoch in range(1, training_config["max_epochs"] + 1):
        epoch_start = time.time()
        net.train()
        model_module.set_frozen_layers_eval(net)   # undo .train()'s effect on layer1/layer2's frozen BatchNorm

        train_losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            clip = batch["clip"].to(device, non_blocking=True)
            label = batch["label_normalized"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                prediction = net(clip).squeeze(1).float()   # cast to float32 before the loss - see module docstring
                loss = loss_fn(prediction, label) / accumulation_steps

            scaler.scale(loss).backward()

            is_last_batch = (batch_index + 1) == len(train_loader)
            if (batch_index + 1) % accumulation_steps == 0 or is_last_batch:
                if use_grad_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_module.trainable_parameters(net),
                                                training_config["grad_clip_max_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            train_losses.append(loss.item() * accumulation_steps)
            report_every = max(1, len(train_loader) // 5)
            if (batch_index + 1) % report_every == 0:
                print(f"  [train] epoch {epoch} batch {batch_index + 1}/{len(train_loader)} "
                      f"loss={np.mean(train_losses[-report_every:]):.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        val_loss, val_mae_n, val_rmse_n, val_r2 = run_validation(
            net, val_loader, device, autocast_dtype, loss_fn, model_config["label"]["force_norm_scale_n"], clamp_sign)
        epoch_time_s = time.time() - epoch_start
        train_loss_mean = float(np.mean(train_losses))
        print(f"[train] epoch {epoch}/{training_config['max_epochs']}  "
              f"train_loss={train_loss_mean:.4f}  val_loss={val_loss:.4f}  "
              f"val_MAE={val_mae_n:.3f}N  val_RMSE={val_rmse_n:.3f}N  val_R2={val_r2:.3f}  ({epoch_time_s:.1f}s)")
        _append_metrics_row(metrics_path, {
            "epoch": epoch, "train_loss": train_loss_mean, "val_loss": val_loss, "val_mae_n": val_mae_n,
            "val_rmse_n": val_rmse_n, "val_r2": val_r2, "lr": scheduler.get_last_lr()[0], "epoch_time_s": epoch_time_s,
        })

        if val_mae_n < best_val_mae_n:
            best_val_mae_n = val_mae_n
            epochs_without_improvement = 0
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            _save_checkpoint(checkpoint_path, net, config, epoch, val_mae_n, clamp_sign)
            print(f"  [train] validation MAE improved to {val_mae_n:.3f}N - checkpoint saved to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config["early_stopping_patience"]:
                print(f"[train] Early stopping: no val MAE improvement in {epochs_without_improvement} epochs")
                break

    print(f"[train] Done. Best val MAE = {best_val_mae_n:.3f}N. "
          f"Checkpoint: {os.path.join(checkpoint_dir, 'best_model.pt')}  Metrics history: {metrics_path}")
    return best_val_mae_n


def check_frozen_batchnorm(net, train_loader, device, optimizer, loss_fn, autocast_dtype, use_grad_scaler, grad_clip_max_norm):
    """Section 8: after one training step, layer1/layer2 weights AND
    running statistics must be unchanged. Snapshots layer1/layer2's full
    state_dict, runs exactly one optimizer step on one real batch, and
    compares. Prints the result. Returns nothing; raises AssertionError on
    a mismatch.
    """
    before = {name: tensor.clone() for name, tensor in net.state_dict().items()
              if name.startswith("layer1.") or name.startswith("layer2.")}

    net.train()
    model_module.set_frozen_layers_eval(net)
    batch = next(iter(train_loader))
    clip = batch["clip"].to(device)
    label = batch["label_normalized"].to(device)

    scaler = torch.amp.GradScaler(device="cuda" if device.type == "cuda" else "cpu", enabled=use_grad_scaler)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        prediction = net(clip).squeeze(1).float()
        loss = loss_fn(prediction, label)
    scaler.scale(loss).backward()
    if use_grad_scaler:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model_module.trainable_parameters(net), grad_clip_max_norm)
    scaler.step(optimizer)
    scaler.update()

    after = net.state_dict()
    max_diff = max((before[name] - after[name].to(before[name].device)).abs().max().item() for name in before)
    print(f"  [train] frozen BatchNorm check: max |before-after| over layer1/layer2 = {max_diff:.3e}")
    assert max_diff == 0.0, "layer1/layer2 changed after a training step despite being frozen"


def check_overfit_one_batch(config, device, num_iterations=300, batch_size=6):
    """Section 8: augmentation off, a handful of clips - training loss must
    fall to near zero within a few hundred iterations, otherwise something
    is wrong in the model, the loss, or the labels. Builds a tiny fixed
    batch from the training split with augmentation disabled and repeatedly
    trains on just that batch. Prints the loss trajectory. Returns nothing.
    """
    config = copy.deepcopy(config)
    aug = config["model_training"]["augmentation"]
    aug["horizontal_flip_prob"] = 0.0
    aug["rotation_enabled"] = False
    aug["depth_noise_enabled"] = False
    aug["temporal_offset_max_frames"] = 0

    train_dataset = dataset.build_dataset("train", config)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    batch = next(iter(loader))
    clip = batch["clip"].to(device)
    label = batch["label_normalized"].to(device)

    net = model_module.build_model(train_dataset.mean_normalized_label()).to(device)
    optimizer = torch.optim.AdamW(model_module.trainable_parameters(net), lr=1e-3)
    delta_normalized = (config["model_training"]["training"]["loss"]["huber_delta_n"] /
                         config["model_training"]["label"]["force_norm_scale_n"])
    loss_fn = nn.HuberLoss(delta=delta_normalized)
    autocast_dtype, _ = select_precision(device)

    print(f"[train] Overfit-one-batch check: {batch_size} clips, {num_iterations} iterations, augmentation off")
    net.train()
    model_module.set_frozen_layers_eval(net)
    final_loss = None
    for iteration in range(1, num_iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            prediction = net(clip).squeeze(1).float()
            loss = loss_fn(prediction, label)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if iteration == 1 or iteration % 50 == 0:
            print(f"  [train] iteration {iteration}/{num_iterations}  loss={final_loss:.5f}")
    verdict = "PASS" if final_loss < 0.01 else "CHECK MANUALLY - did not fall near zero"
    print(f"[train] Overfit-one-batch final loss: {final_loss:.5f} ({verdict})")


if __name__ == "__main__":
    run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {run_device}" +
          (f" ({torch.cuda.get_device_name(run_device)})" if run_device.type == "cuda" else ""))
    run_config = dataset.load_config()

    print("[train] Building train dataset for the checks below")
    check_train_dataset, check_train_loader = dataset.build_dataloader("train", run_config)
    check_val_dataset, _ = dataset.build_dataloader("val", run_config)
    print(f"[train] Trivial baseline (always predict the mean train force): val MAE = "
          f"{_trivial_baseline_mae_n(check_train_dataset, check_val_dataset):.3f} N")

    check_net = model_module.build_model(check_train_dataset.mean_normalized_label()).to(run_device)
    check_optimizer = build_optimizer(check_net, run_config["model_training"]["training"]["optimizer"])
    check_delta = (run_config["model_training"]["training"]["loss"]["huber_delta_n"] /
                   run_config["model_training"]["label"]["force_norm_scale_n"])
    check_loss_fn = nn.HuberLoss(delta=check_delta)
    check_autocast_dtype, check_use_scaler = select_precision(run_device)

    check_frozen_batchnorm(check_net, check_train_loader, run_device, check_optimizer, check_loss_fn,
                            check_autocast_dtype, check_use_scaler, run_config["model_training"]["training"]["grad_clip_max_norm"])

    check_overfit_one_batch(run_config, run_device)

    print("[train] All train.py checks done. Call train.train() (see mainModel.py) to run the full training loop.")
