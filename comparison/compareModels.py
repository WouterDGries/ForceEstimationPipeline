"""Compares the 3D (Model/), 2D (Model2D/) and PressureVision models on the
test frames they ALL predicted, from the predictions.csv files their
evaluate.py scripts write (one per trained run, e.g. one per seed).

Checks first that every run used the same test sessions (splits.json next
to each checkpoint) and that the true labels agree, then reports per model
(mean +/- std over its runs, i.e. seeds):

  * overall MAE / RMSE / R2 / bias, next to the trivial baseline (always
    predict the mean training force)
  * no-contact frames (true <= contact threshold): MAE, mean predicted force
    and false-force rate (prediction above the threshold while not touching)
  * contact frames: MAE, contact accuracy / precision / recall / F1
  * MAE per force bin
  * dynamic frames (|dF/dt| above a threshold: press onsets and releases)
    vs static frames - where temporal context should matter
  * MAE per test session
  * uncertainty: 95% bootstrap intervals over the test SESSIONS (resampled
    with replacement) for every model's MAE and every pairwise MAE
    difference, plus the seed-to-seed spread

With --benchmark it also measures every model's inference cost
(benchmarkModel.py, one subprocess per model: parameters, FLOPs, latency,
throughput, GPU memory, optionally CPU latency) and adds it to the report;
--benchmark_only does just that (no predictions needed) and writes
benchmark.md / benchmark.json.

Settings and the run folders per model are in config.yaml's comparison
block. Writes <output_dir>/summary.md, summary.json, common_predictions.csv,
force_bin_mae.png and timeline_<session>.png.

Usage (trajPipeline env, from the repo root, after evaluating every model):
    python comparison/compareModels.py
    python comparison/compareModels.py --benchmark        # + inference time / compute per model
    python comparison/compareModels.py --benchmark_only   # only the inference cost
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "config.yaml")


def repo_path(path):
    """Returns path resolved against the repo root (absolute paths unchanged)."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(REPO_ROOT, path))


def find_runs(model_patterns):
    """Input: {model name: [glob patterns of eval_<split> folders]}. Returns
    {model name: [eval folder, ...]} for the folders that contain a
    predictions.csv (models without any run are left out, with a warning).
    """
    runs = {}
    for name, patterns in model_patterns.items():
        found = sorted({os.path.normpath(path) for pattern in patterns for path in glob.glob(repo_path(pattern))
                        if os.path.isfile(os.path.join(path, "predictions.csv"))})
        if found:
            runs[name] = found
        else:
            print(f"[compare] WARNING: no predictions.csv found for {name} ({patterns}) - left out")
    return runs


def check_splits(runs):
    """Every run's checkpoint folder (parent of eval_<split>) holds the
    splits.json it was trained with. Returns the shared split; exits if
    any two runs differ (a comparison over different test sessions is
    meaningless).
    """
    splits_by_run = {}
    for name, folders in runs.items():
        for folder in folders:
            path = os.path.join(os.path.dirname(folder), "splits.json")
            if not os.path.isfile(path):
                sys.exit(f"[compare] {path} missing - cannot check that {name} used the same split")
            with open(path) as f:
                splits_by_run[f"{name}: {folder}"] = {split: sorted(names) for split, names in json.load(f).items()}
    reference_run, reference = next(iter(splits_by_run.items()))
    for run, splits in splits_by_run.items():
        for split in ("train", "val", "test"):
            if splits.get(split) != reference.get(split):
                only_here = sorted(set(splits.get(split, [])) - set(reference.get(split, [])))
                only_there = sorted(set(reference.get(split, [])) - set(splits.get(split, [])))
                sys.exit(f"[compare] {split} split differs between\n  {reference_run}\n  {run}\n"
                         f"  only in the second: {only_here}\n  only in the first: {only_there}\n"
                         f"Make sure all projects have the same PROCESSED_OK sessions and split settings, then retrain.")
    return reference


def load_truth(data_root, sessions):
    """Labels from the shared processed data: force_n and time for every
    saved frame, plus the force rate |dF/dt| (N/s) from neighbouring frames
    of the same session. Returns a DataFrame keyed by (session, raw_frame_index).
    """
    frames = []
    for name in sessions:
        df = pd.read_csv(os.path.join(data_root, name, "frames.csv"),
                         usecols=["raw_frame_index", "host_ts_ns", "force_n", "artifact"])
        t_s = df["host_ts_ns"].to_numpy() / 1e9
        df["force_rate_n_per_s"] = np.abs(np.gradient(df["force_n"].to_numpy(), t_s)) if len(df) > 1 else 0.0
        df["session"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def mean_train_force(data_root, train_sessions, max_label_force_n):
    """Trivial baseline: mean label over the usable training frames (no
    artifact, label <= max_label_force_n), as in every model's evaluate.py.
    """
    labels = []
    for name in train_sessions:
        df = pd.read_csv(os.path.join(data_root, name, "frames.csv"), usecols=["force_n", "artifact"])
        labels.append(df.loc[(df["artifact"] == 0) & (df["force_n"] <= max_label_force_n), "force_n"].to_numpy())
    return float(np.concatenate(labels).mean())


def frame_masks(common, settings):
    """Boolean masks over the common frames for every reported subset."""
    thr = settings["contact_threshold_n"]
    true_n = common["true_n"].to_numpy()
    dynamic = common["force_rate_n_per_s"].to_numpy() > settings["dynamic_force_rate_n_per_s"]
    masks = {"all": np.ones(len(common), dtype=bool), "no_contact": true_n <= thr, "contact": true_n > thr,
             "dynamic": dynamic, "static": ~dynamic}
    edges = settings["force_bins_n"]
    for low, high in zip(edges[:-1], edges[1:]):
        masks[f"bin_{low:g}-{high:g}N"] = (true_n > low) & (true_n <= high)
    return masks


def run_metrics(pred_n, true_n, masks, sessions, thr):
    """All scalar metrics of one run on the common frames."""
    error = pred_n - true_n
    abs_error = np.abs(error)
    ss_tot = np.sum((true_n - true_n.mean()) ** 2)
    pred_contact, true_contact = pred_n > thr, true_n > thr
    tp = np.sum(pred_contact & true_contact)
    precision = tp / max(pred_contact.sum(), 1)
    recall = tp / max(true_contact.sum(), 1)
    metrics = {
        "mae_n": abs_error.mean(),
        "rmse_n": np.sqrt(np.mean(error ** 2)),
        "r2": 1.0 - np.sum(error ** 2) / ss_tot if ss_tot > 0 else np.nan,
        "bias_n": error.mean(),
        "contact_acc": np.mean(pred_contact == true_contact),
        "contact_precision": precision,
        "contact_recall": recall,
        "contact_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "no_contact_mean_pred_n": pred_n[masks["no_contact"]].mean() if masks["no_contact"].any() else np.nan,
        "no_contact_false_force_rate": pred_contact[masks["no_contact"]].mean() if masks["no_contact"].any() else np.nan,
    }
    for subset, mask in masks.items():
        if subset != "all":
            metrics[f"mae_{subset}_n"] = abs_error[mask].mean() if mask.any() else np.nan
    for session in np.unique(sessions):
        metrics[f"mae_session_{session}_n"] = abs_error[sessions == session].mean()
    return {key: float(value) for key, value in metrics.items()}


def bootstrap(per_run_errors, masks, sessions, settings):
    """Session-level bootstrap. per_run_errors: {model: [abs error array per
    run]}. Resamples the test sessions with replacement; a model's MAE in a
    resample is the mean over its runs. Returns {subset: {"models": {model:
    (low, high)}, "differences": {"A - B": (mean, low, high, P(A<B))}}}.
    """
    rng = np.random.default_rng(settings["bootstrap_seed"])
    session_names = np.unique(sessions)
    session_index = np.searchsorted(session_names, sessions)
    draws = rng.integers(0, len(session_names), size=(settings["bootstrap_samples"], len(session_names)))
    weights = np.stack([np.bincount(draw, minlength=len(session_names)) for draw in draws])   # (B, S)
    results = {}
    for subset in ("all", "no_contact", "contact", "dynamic", "static"):
        mask = masks[subset]
        counts = np.bincount(session_index[mask], minlength=len(session_names)).astype(float)
        frame_total = weights @ counts
        valid = frame_total > 0
        model_mae = {}
        for model, errors_per_run in per_run_errors.items():
            run_maes = [(weights @ np.bincount(session_index[mask], weights=errors[mask], minlength=len(session_names)))
                        / np.where(valid, frame_total, 1) for errors in errors_per_run]
            model_mae[model] = np.mean(run_maes, axis=0)[valid]
        names = list(model_mae)
        results[subset] = {
            "models": {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in model_mae.items()},
            "differences": {
                f"{a} - {b}": [float(np.mean(model_mae[a] - model_mae[b])),
                               float(np.percentile(model_mae[a] - model_mae[b], 2.5)),
                               float(np.percentile(model_mae[a] - model_mae[b], 97.5)),
                               float(np.mean(model_mae[a] < model_mae[b]))]
                for i, a in enumerate(names) for b in names[i + 1:]
            },
        }
    return results


def _plots(common, models, masks, settings, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = [key for key in masks if key.startswith("bin_")]
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.8 / len(models)
    for k, model in enumerate(models):
        abs_err = np.abs(common[f"pred_{model}"] - common["true_n"]).to_numpy()
        ax.bar(np.arange(len(bins)) + (k - (len(models) - 1) / 2) * width,
               [abs_err[masks[b]].mean() if masks[b].any() else np.nan for b in bins], width, label=model)
    no_contact_err = {m: np.abs(common[f"pred_{m}"] - common["true_n"]).to_numpy()[masks["no_contact"]].mean()
                      for m in models}
    ax.set_xticks(np.arange(len(bins)), [b[4:] for b in bins])
    ax.set_xlabel("true force bin")
    ax.set_ylabel("MAE (N)")
    ax.set_title("MAE per force bin (no contact: " + ", ".join(f"{m} {v:.2f} N" for m, v in no_contact_err.items()) + ")")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "force_bin_mae.png"), dpi=100)
    plt.close(fig)

    for session, df in common.groupby("session"):
        t = (df["host_ts_ns"] - df["host_ts_ns"].min()) / 1e9
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t, df["true_n"], ".", ms=1.5, color="black", label="measured")
        for model in models:
            mae = np.mean(np.abs(df[f"pred_{model}"] - df["true_n"]))
            ax.plot(t, df[f"pred_{model}"], ".", ms=1.5, label=f"{model} (MAE {mae:.2f} N)")
        ax.set_title(f"{session} - common frames, prediction averaged over runs")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("force (N)")
        ax.legend(markerscale=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"timeline_{session}.png"), dpi=90)
        plt.close(fig)


def _summary_markdown(summary):
    models = list(summary["models"])
    lines = [f"# Model comparison - {summary['num_common_frames']} common test frames, "
             f"{len(summary['test_sessions'])} test sessions", ""]
    lines += ["## Coverage", "", "| run | frames predicted | in common set |", "|---|---|---|"]
    lines += [f"| {run} | {n} | {summary['num_common_frames'] / n * 100:.1f}% |" for run, n in summary["frames_per_run"].items()]
    lines += ["", f"Label check (max |true_n - processed force_n| over common frames): "
                  f"{summary['max_label_difference_n']:.4f} N", ""]

    def row(label, key, fmt="{:.3f}"):
        cells = []
        for m in models:
            s = summary["models"][m]["mean"].get(key, np.nan)
            sd = summary["models"][m]["std"].get(key, np.nan)
            cells.append(fmt.format(s) + (f" ± {fmt.format(sd)}" if summary["models"][m]["runs"] > 1 else ""))
        return f"| {label} | " + " | ".join(cells) + f" | {fmt.format(summary['baseline'].get(key, np.nan))} |"

    lines += ["## Metrics (mean ± std over runs)", "",
              "| metric | " + " | ".join(f"{m} ({summary['models'][m]['runs']} run(s))" for m in models) + " | trivial baseline |",
              "|---" * (len(models) + 2) + "|"]
    for label, key in [("MAE (N)", "mae_n"), ("RMSE (N)", "rmse_n"), ("R²", "r2"), ("bias (N)", "bias_n"),
                       ("MAE no contact (N)", "mae_no_contact_n"), ("mean prediction, no contact (N)", "no_contact_mean_pred_n"),
                       ("false-force rate, no contact", "no_contact_false_force_rate"),
                       ("MAE contact (N)", "mae_contact_n"), ("contact accuracy", "contact_acc"),
                       ("contact F1", "contact_f1"), ("MAE dynamic (N)", "mae_dynamic_n"), ("MAE static (N)", "mae_static_n")]:
        lines.append(row(label, key))
    for key in summary["force_bins"]:
        lines.append(row(f"MAE {key[4:]} (N)", f"mae_{key}_n"))
    lines += ["", f"Frames: {summary['subset_sizes']}", "",
              "## 95% bootstrap intervals over test sessions", ""]
    for subset, result in summary["bootstrap"].items():
        lines.append(f"**{subset}** - MAE: " + ", ".join(f"{m} [{lo:.3f}, {hi:.3f}]" for m, (lo, hi) in result["models"].items()))
        lines.append("")
        lines += ["| difference | mean (N) | 95% interval | P(first < second) |", "|---|---|---|---|"]
        lines += [f"| {pair} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.2f} |"
                  for pair, (d, lo, hi, p) in result["differences"].items()]
        lines.append("")
    lines += ["An interval that contains 0 means the difference is not resolved by these test sessions. "
              "Compare differences with the seed-to-seed std above as well.", "",
              "## MAE per test session (N, mean over runs)", "",
              "| session | " + " | ".join(models) + " |", "|---" * (len(models) + 1) + "|"]
    for session in summary["test_sessions"]:
        lines.append(f"| {session} | " + " | ".join(
            f"{summary['models'][m]['mean'].get(f'mae_session_{session}_n', np.nan):.3f}" for m in models) + " |")
    return "\n".join(lines) + "\n"


def _benchmark_checkpoint(name, model_settings, runs):
    """Checkpoint to benchmark: the benchmark block's explicit `checkpoint`,
    else best_model.pt next to the model's first evaluated run, else
    <code_dir>/checkpoints/best_model.pt. Weights don't change the timing,
    only which architecture/input size gets built.
    """
    candidates = [model_settings.get("checkpoint")]
    if runs.get(name):
        candidates.append(os.path.join(os.path.dirname(runs[name][0]), "best_model.pt"))
    candidates.append(os.path.join(model_settings["code_dir"], "checkpoints", "best_model.pt"))
    for candidate in candidates:
        if candidate and os.path.isfile(repo_path(candidate)):
            return repo_path(candidate)
    return None


def run_benchmarks(benchmark_settings, runs):
    """Runs comparison/benchmarkModel.py once per model in
    benchmark_settings['models'] (separate processes: the projects' modules
    share names). Returns {model: result dict}; models without a checkpoint
    or whose benchmark fails are reported and left out.
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarkModel.py")
    options = {k: v for k, v in benchmark_settings.items() if k != "models"}
    results = {}
    for name, model_settings in benchmark_settings["models"].items():
        checkpoint = _benchmark_checkpoint(name, model_settings, runs)
        if checkpoint is None:
            print(f"[compare] WARNING: no checkpoint found to benchmark {name} - left out")
            continue
        print(f"[compare] benchmarking {name} ({model_settings['kind']}) from {checkpoint}")
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "result.json")
            completed = subprocess.run(
                [sys.executable, script, "--kind", model_settings["kind"], "--code_dir", repo_path(model_settings["code_dir"]),
                 "--checkpoint", checkpoint, "--settings", json.dumps(options), "--out", out_path],
                cwd=repo_path(model_settings["code_dir"]))
            if completed.returncode != 0 or not os.path.isfile(out_path):
                print(f"[compare] WARNING: benchmark of {name} failed (exit code {completed.returncode}) - left out")
                continue
            with open(out_path) as f:
                results[name] = json.load(f)
    return results


def _benchmark_markdown(results, settings):
    if not results:
        return "## Inference cost\n\nNo model could be benchmarked.\n"
    models = list(results)
    batch_size = settings["batch_size"]
    first = results[models[0]]
    lines = ["## Inference cost per force prediction", "",
             f"Device: {first['device']}, precision: {first['precision']}, random inputs of each model's own size, "
             f"{settings['timed_iterations']} timed passes after {settings['warmup_iterations']} warm-up. "
             "Fingertip tracking and cropping are shared by all models and not included.", "",
             "| | " + " | ".join(models) + " |", "|---" * (len(models) + 1) + "|"]

    def row(label, key, fmt):
        return f"| {label} | " + " | ".join(fmt.format(results[m][key]) if key in results[m] else "-"
                                           for m in models) + " |"
    lines += [
        f"| input | " + " | ".join(results[m]["input"] for m in models) + " |",
        row("frames per prediction", "frames_per_prediction", "{}"),
        f"| parameters (M) | " + " | ".join(f"{results[m]['parameters'] / 1e6:.2f}" for m in models) + " |",
        row("checkpoint size (MB)", "checkpoint_mb", "{:.1f}"),
        row("GFLOPs per prediction", "gflops_per_prediction", "{:.2f}"),
        row("GPU latency, batch 1, median (ms)", "latency_ms_median", "{:.2f}"),
        row("GPU latency, batch 1, p95 (ms)", "latency_ms_p95", "{:.2f}"),
        f"| max rate, batch 1 (predictions/s) | " + " | ".join(f"{1000.0 / results[m]['latency_ms_median']:.0f}"
                                                              for m in models) + " |",
        row(f"throughput, batch {batch_size} (predictions/s)", "throughput_predictions_per_s", "{:.0f}"),
        row("peak GPU memory, batch 1 (MB)", "peak_gpu_memory_mb_batch1", "{:.0f}"),
        row(f"peak GPU memory, batch {batch_size} (MB)", f"peak_gpu_memory_mb_batch{batch_size}", "{:.0f}"),
    ]
    if any("cpu_latency_ms_median" in results[m] for m in models):
        lines += [row(f"CPU latency, batch 1, median (ms, {settings['cpu_threads']} threads)", "cpu_latency_ms_median", "{:.1f}"),
                  row("CPU latency, batch 1, p95 (ms)", "cpu_latency_ms_p95", "{:.1f}")]
    frame_budget_ms = 1000.0 / settings["camera_fps"]
    lines += ["", f"Real time at {settings['camera_fps']} fps (one prediction per frame, budget {frame_budget_ms:.1f} ms): " +
              ", ".join(f"{m} {'yes' if results[m]['latency_ms_median'] <= frame_budget_ms else 'no'}" for m in models) +
              " (GPU, batch 1). The 3D model re-processes its whole window for every new frame.", "",
              "GPU latency at batch 1 is largely fixed per-call overhead (Python, kernel launches), so for small models "
              "it says little about compute - compare GFLOPs, batched throughput and CPU latency for that. "
              "Run on the deployment machine for real-time numbers.", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", action="store_true", help="also measure inference time / compute per model")
    parser.add_argument("--benchmark_only", action="store_true", help="only measure inference time / compute")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    settings = config["comparison"]
    thr = settings["contact_threshold_n"]
    out_dir = repo_path(settings["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    runs = find_runs(settings["models"])
    if args.benchmark_only:
        results = run_benchmarks(settings["benchmark"], runs)
        with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
            json.dump(results, f, indent=2)
        markdown = _benchmark_markdown(results, settings["benchmark"])
        with open(os.path.join(out_dir, "benchmark.md"), "w") as f:
            f.write(markdown)
        print(markdown)
        print(f"[compare] Outputs written to {out_dir}")
        return
    if len(runs) < 2:
        sys.exit("[compare] need predictions from at least two models")
    splits = check_splits(runs)
    print(f"[compare] all runs share the split; test sessions: {splits['test']}")

    predictions, frames_per_run = {}, {}
    for model, folders in runs.items():
        for folder in folders:
            df = pd.read_csv(os.path.join(folder, "predictions.csv"), usecols=["session", "raw_frame_index", "true_n", "pred_n"])
            df = df[df["session"].isin(splits["test"])].drop_duplicates(["session", "raw_frame_index"])
            predictions[(model, folder)] = df.set_index(["session", "raw_frame_index"])
            frames_per_run[f"{model}: {os.path.relpath(folder, REPO_ROOT)}"] = len(df)
            print(f"[compare] {model}: {folder} - {len(df)} frames")

    common_index = None
    for df in predictions.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    if common_index is None or len(common_index) == 0:
        sys.exit("[compare] no frames in common - are the predictions from the same processed sessions?")

    data_root = repo_path(settings["data_root"])
    truth = load_truth(data_root, splits["test"]).set_index(["session", "raw_frame_index"])
    missing = len(common_index.difference(truth.index))
    if missing:
        print(f"[compare] WARNING: {missing} common frames are not in {data_root} - left out")
    common = truth.loc[common_index.intersection(truth.index).sort_values()].reset_index()
    common = common.rename(columns={"force_n": "true_n"})
    key = pd.MultiIndex.from_frame(common[["session", "raw_frame_index"]])
    max_label_difference = max(float(np.max(np.abs(df.loc[key, "true_n"].to_numpy() - common["true_n"].to_numpy())))
                               for df in predictions.values())
    if max_label_difference > 0.05:
        print(f"[compare] WARNING: labels differ by up to {max_label_difference:.3f} N between a model and the processed data")

    masks = frame_masks(common, settings)
    true_n = common["true_n"].to_numpy()
    sessions = common["session"].to_numpy()
    per_run_metrics, per_run_errors = {}, {}
    for (model, folder), df in predictions.items():
        pred_n = df.loc[key, "pred_n"].to_numpy()
        per_run_metrics.setdefault(model, []).append(run_metrics(pred_n, true_n, masks, sessions, thr))
        per_run_errors.setdefault(model, []).append(np.abs(pred_n - true_n))
    for model in runs:
        common[f"pred_{model}"] = np.mean([predictions[(model, f)].loc[key, "pred_n"].to_numpy() for f in runs[model]], axis=0)

    baseline_n = mean_train_force(data_root, splits["train"], settings["max_label_force_n"])
    summary = {
        "test_sessions": splits["test"],
        "num_common_frames": int(len(common)),
        "frames_per_run": frames_per_run,
        "max_label_difference_n": max_label_difference,
        "settings": settings,
        "force_bins": [k for k in masks if k.startswith("bin_")],
        "subset_sizes": {k: int(v.sum()) for k, v in masks.items()},
        "baseline": {"mean_train_force_n": baseline_n,
                     **run_metrics(np.full(len(common), baseline_n), true_n, masks, sessions, thr)},
        "models": {model: {"runs": len(metrics_list),
                           "run_folders": runs[model],
                           "mean": pd.DataFrame(metrics_list).mean().to_dict(),
                           "std": pd.DataFrame(metrics_list).std(ddof=1).fillna(0.0).to_dict(),
                           "per_run": metrics_list}
                   for model, metrics_list in per_run_metrics.items()},
        "bootstrap": bootstrap(per_run_errors, masks, sessions, settings),
    }
    if args.benchmark:
        summary["inference_cost"] = run_benchmarks(settings["benchmark"], runs)

    common.to_csv(os.path.join(out_dir, "common_predictions.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    markdown = _summary_markdown(summary)
    if args.benchmark:
        markdown += "\n" + _benchmark_markdown(summary["inference_cost"], settings["benchmark"])
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(markdown)
    _plots(common, list(runs), masks, settings, out_dir)
    print(markdown)
    print(f"[compare] Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
