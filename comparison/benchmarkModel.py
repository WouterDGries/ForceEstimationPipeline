"""Measures the inference cost of ONE trained model - parameters, FLOPs,
GPU/CPU latency, throughput and memory per force prediction. Called by
compareModels.ipynb (RUN_BENCHMARK = True) in a separate process per model, because the
three projects all have top-level modules named dataset/model/train that
would clash in one interpreter.

What one "prediction" is differs per model, so the input is that model's
own unit of work, with random values (timing does not depend on content):
    clip3d          (1, 4, T, 112, 112) RGB-D clip -> one force (T = the checkpoint's window_length);
                    in streaming use this runs once per new frame on the last T frames
    frame2d         (1, 4, 112, 112) RGB-D frame -> one force
    pressurevision  (1, 3, px, px) RGB crop + pixel area -> one force (px from the checkpoint's config)
Fingertip tracking and cropping (MediaPipe etc.) are the same for all three
and are not included.

Usage (normally via compareModels.ipynb):
    python comparison/benchmarkModel.py --kind clip3d --code_dir Model --checkpoint Model/checkpoints/best_model.pt \
        --settings '{"batch_size": 16, ...}' --out result.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch


def _autocast_dtype(device, precision):
    """bf16 on Ampere or newer, fp16 on older GPUs - as the projects'
    train.select_precision() - or None for fp32 / CPU.
    """
    if precision != "autocast" or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def build(kind, checkpoint_path, device):
    """Returns (model in eval mode, make_inputs(batch_size) -> tuple of input
    tensors, description dict) for the given model kind.
    """
    if kind in ("clip3d", "frame2d"):
        import model as model_module
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        net = model_module.build_model(mean_normalized_label=0.0)
        net.load_state_dict(checkpoint["model_state_dict"])
        crop_px = checkpoint["metadata"].get("crop_size_px", 112)
        if kind == "clip3d":
            frames = int(checkpoint["metadata"]["window_length"])
            shape = (4, frames, crop_px, crop_px)
        else:
            frames = 1
            shape = (4, crop_px, crop_px)

        def make_inputs(batch_size):
            return (torch.randn((batch_size,) + shape, device=device),)
        description = {"input": f"{shape} RGB + relative depth", "frames_per_prediction": frames}
    elif kind == "pressurevision":
        import model as model_module
        net, checkpoint = model_module.load_checkpoint(checkpoint_path, torch.device("cpu"))
        crop = checkpoint["config"]["processing"]["crop"]
        crop_px = crop["output_px"]
        pixel_area_m2 = (crop["side_mm"] / crop_px / 1000.0) ** 2

        def make_inputs(batch_size):
            return (torch.randn((batch_size, 3, crop_px, crop_px), device=device),
                    torch.full((batch_size,), pixel_area_m2, device=device))
        description = {"input": f"(3, {crop_px}, {crop_px}) RGB", "frames_per_prediction": 1}
    else:
        raise ValueError(f"unknown model kind '{kind}'")
    return net.to(device).eval(), make_inputs, description


def count_flops(net, inputs):
    """FLOPs of one forward pass (multiply + add counted separately, so
    FLOPs = 2 x MACs), via torch's FlopCounterMode.
    """
    from torch.utils.flop_counter import FlopCounterMode
    counter = FlopCounterMode(display=False)
    with torch.inference_mode(), counter:
        net(*inputs)
    return counter.get_total_flops()


def time_forward(net, inputs, device, autocast_dtype, warmup, iterations):
    """Runs warmup forward passes, then times `iterations` single passes
    (synchronizing the GPU around each). Returns per-pass times in ms.
    """
    times_ms = []
    with torch.inference_mode():
        for i in range(warmup + iterations):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                net(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if i >= warmup:
                times_ms.append((time.perf_counter() - start) * 1000.0)
    return np.array(times_ms)


def peak_memory_mb(net, inputs, device, autocast_dtype):
    """Peak GPU memory (MB) allocated during one forward pass, weights included."""
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=autocast_dtype,
                                                enabled=autocast_dtype is not None):
        net(*inputs)
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 2 ** 20


def benchmark(kind, checkpoint_path, settings):
    """Input: model kind, checkpoint path, config['comparison']['benchmark'].
    Returns a dict of cost metrics (see the module docstring).
    """
    torch.backends.cudnn.benchmark = True
    use_cuda = settings["device"] == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    net, make_inputs, description = build(kind, checkpoint_path, device)
    autocast_dtype = _autocast_dtype(device, settings["precision"])

    result = {
        "kind": kind,
        "checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_mb": os.path.getsize(checkpoint_path) / 2 ** 20,
        **description,
        "parameters": sum(p.numel() for p in net.parameters()),
        "device": torch.cuda.get_device_name(device) if use_cuda else "cpu",
        "precision": str(autocast_dtype).replace("torch.", "") if autocast_dtype else "float32",
        "gflops_per_prediction": count_flops(net, make_inputs(1)) / 1e9,
    }

    single = time_forward(net, make_inputs(1), device, autocast_dtype,
                          settings["warmup_iterations"], settings["timed_iterations"])
    result.update({"latency_ms_median": float(np.median(single)), "latency_ms_p95": float(np.percentile(single, 95))})
    batch_size = settings["batch_size"]
    batched = time_forward(net, make_inputs(batch_size), device, autocast_dtype,
                           settings["warmup_iterations"], settings["timed_iterations"])
    result.update({"throughput_batch_size": batch_size,
                   "throughput_predictions_per_s": float(batch_size * 1000.0 / np.median(batched))})
    if use_cuda:
        result["peak_gpu_memory_mb_batch1"] = peak_memory_mb(net, make_inputs(1), device, autocast_dtype)
        result[f"peak_gpu_memory_mb_batch{batch_size}"] = peak_memory_mb(net, make_inputs(batch_size), device,
                                                                         autocast_dtype)

    if settings.get("also_cpu") and use_cuda:
        torch.set_num_threads(settings["cpu_threads"])
        cpu = torch.device("cpu")
        net_cpu = net.to(cpu).float()
        cpu_times = time_forward(net_cpu, tuple(x.to(cpu) for x in make_inputs(1)), cpu, None,
                                 max(2, settings["warmup_iterations"] // 5), settings["cpu_timed_iterations"])
        result.update({"cpu_threads": settings["cpu_threads"],
                       "cpu_latency_ms_median": float(np.median(cpu_times)),
                       "cpu_latency_ms_p95": float(np.percentile(cpu_times, 95))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", required=True, choices=["clip3d", "frame2d", "pressurevision"])
    parser.add_argument("--code_dir", required=True, help="folder with the model's model.py (imported from there)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--settings", required=True, help="config['comparison']['benchmark'] as JSON")
    parser.add_argument("--out", required=True, help="where the result JSON is written")
    args = parser.parse_args()

    checkpoint = os.path.abspath(args.checkpoint)
    sys.path.insert(0, os.path.abspath(args.code_dir))
    result = benchmark(args.kind, checkpoint, json.loads(args.settings))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
