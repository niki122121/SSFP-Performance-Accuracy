#!/usr/bin/env python3
"""Drive successive SSFP/Noir SGD steps through Prover.toml.

Requirements: Python >=3.11, numpy, nargo in PATH, MNIST IDX .gz files.
The Noir debugging circuit must return a *flat* [Field; TOTAL_PARAMETERS+1]:
W1, W2, W3, b1, b2, b3 in column-major order, then loss at level 1.
See fcnn_debug_output_patch.md for the required Noir-side changes.

This is an execution/training harness, not a proof verifier. It executes the
same Noir program for each batch, without generating cryptographic proofs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

# Noir's default BN254 scalar-field modulus. Change --field-modulus if needed.
BN254_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617


INPUT_SIZE = 784
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32
OUTPUT_SIZE = 10
BATCH_SIZE = 4
MAX_TRAINING_BATCHES = 9380  # Matches the current Noir configuration.

PARAMETERS = (
    ("W1", "W1_raw", (INPUT_SIZE, HIDDEN_SIZE_1), 2),
    ("W2", "W2_raw", (HIDDEN_SIZE_1, HIDDEN_SIZE_2), 1),
    ("W3", "W3_raw", (HIDDEN_SIZE_2, OUTPUT_SIZE), 1),
    ("b1", "b1_raw", (1, HIDDEN_SIZE_1), 1),
    ("b2", "b2_raw", (1, HIDDEN_SIZE_2), 1),
    ("b3", "b3_raw", (1, OUTPUT_SIZE), 1),
)
TOTAL_PARAMETERS = sum(math.prod(shape) for _, _, shape, _ in PARAMETERS)
EXPECTED_OUTPUT_COUNT = TOTAL_PARAMETERS + 1
METRIC_COLUMNS = (
    "step", "epoch", "batch", "elapsed_s", "noir_loss_pre", "numpy_loss_pre",
    "numpy_loss_post", "train_accuracy", "train_precision_macro",
    "train_recall_macro", "train_f1_macro", "test_accuracy",
    "test_precision_macro", "test_recall_macro", "test_f1_macro",
)


def load_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        header = handle.read(16)
        if len(header) != 16:
            raise ValueError(f"Invalid MNIST image header: {path}")
        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051 or rows * cols != INPUT_SIZE:
            raise ValueError(f"Unexpected MNIST image format: {path}")
        raw = np.frombuffer(handle.read(), dtype=np.uint8)
    if raw.size != count * INPUT_SIZE:
        raise ValueError(f"Image payload length mismatch: {path}")
    return raw.reshape(count, INPUT_SIZE).copy()


def load_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        header = handle.read(8)
        if len(header) != 8:
            raise ValueError(f"Invalid MNIST label header: {path}")
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Unexpected MNIST label format: {path}")
        raw = np.frombuffer(handle.read(), dtype=np.uint8)
    if raw.size != count:
        raise ValueError(f"Label payload length mismatch: {path}")
    return raw.copy()


def load_mnist(mnist_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        load_idx_images(mnist_dir / "train-images-idx3-ubyte.gz"),
        load_idx_labels(mnist_dir / "train-labels-idx1-ubyte.gz"),
        load_idx_images(mnist_dir / "t10k-images-idx3-ubyte.gz"),
        load_idx_labels(mnist_dir / "t10k-labels-idx1-ubyte.gz"),
    )


def balanced_subset_indices(labels: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Choose a deterministic approximately class-balanced subset."""
    if size > len(labels) or size < BATCH_SIZE:
        raise ValueError("Subset size must be >= batch size and <= dataset size")
    rng = np.random.default_rng(seed)
    per_class = [size // OUTPUT_SIZE + (c < size % OUTPUT_SIZE) for c in range(OUTPUT_SIZE)]
    chunks = []
    for cls, take in enumerate(per_class):
        candidates = np.flatnonzero(labels == cls)
        if len(candidates) < take:
            raise ValueError(f"Class {cls} has insufficient examples")
        chunks.append(rng.choice(candidates, size=take, replace=False))
    indices = np.concatenate(chunks)
    rng.shuffle(indices)
    return indices


def field_int(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("Unexpected Boolean instead of Field")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        return int(value, 16 if value.lower().startswith("0x") else 10)
    raise ValueError(f"Unexpected field representation: {type(value).__name__}")


def field_to_signed(value: object, modulus: int) -> int:
    n = field_int(value) % modulus
    return n - modulus if n > modulus // 2 else n


def encode_field(value: int, modulus: int) -> str:
    return f'"{int(value) % modulus}"'


def quantize(values: np.ndarray, level: int, d: int) -> np.ndarray:
    """Nearest-integer quantization; store as Python arbitrary-precision integers."""
    scaled = np.rint(np.asarray(values, dtype=np.float64) * (1 << (level * d)))
    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite weights during quantization")
    return np.fromiter((int(v) for v in scaled.flat), dtype=object, count=scaled.size).reshape(scaled.shape)


def initial_parameters(seed: int, d: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = {
        "W1": (rng.standard_normal((INPUT_SIZE, HIDDEN_SIZE_1)).astype(np.float32) * np.float32(np.sqrt(2.0 / INPUT_SIZE))),
        "W2": (rng.standard_normal((HIDDEN_SIZE_1, HIDDEN_SIZE_2)).astype(np.float32) * np.float32(np.sqrt(2.0 / HIDDEN_SIZE_1))),
        "W3": (rng.standard_normal((HIDDEN_SIZE_2, OUTPUT_SIZE)).astype(np.float32) * np.float32(np.sqrt(2.0 / HIDDEN_SIZE_2))),
        "b1": np.zeros((1, HIDDEN_SIZE_1), dtype=np.float32),
        "b2": np.zeros((1, HIDDEN_SIZE_2), dtype=np.float32),
        "b3": np.zeros((1, OUTPUT_SIZE), dtype=np.float32),
    }
    return {name: quantize(weights[name], level, d) for name, _, _, level in PARAMETERS}


def write_toml_array(handle, key: str, values: np.ndarray, modulus: int) -> None:
    if values.ndim == 1:
        handle.write(f"{key} = [")
        handle.write(", ".join(encode_field(v, modulus) for v in values))
        handle.write("]\n")
    elif values.ndim == 2:
        handle.write(f"{key} = [\n")
        for row in values:
            handle.write("  [")
            handle.write(", ".join(encode_field(v, modulus) for v in row))
            handle.write("],\n")
        handle.write("]\n")
    else:
        raise ValueError(f"Expected 1D or 2D array for {key}")


def write_prover(path: Path, pixel_batch: np.ndarray, label_batch: np.ndarray,
                 parameters: dict[str, np.ndarray], modulus: int) -> None:
    """Rewrite Prover.toml atomically and deliberately omit any previous return value."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
        write_toml_array(handle, "data_train", pixel_batch, modulus)
        write_toml_array(handle, "labels_train", label_batch, modulus)
        for name, toml_name, shape, _ in PARAMETERS:
            assert parameters[name].shape == shape
            values = parameters[name][0] if name.startswith("b") else parameters[name]
            write_toml_array(handle, toml_name, values, modulus)
    os.replace(tmp, path)


def flatten_toml_fields(obj: object) -> list[object]:
    if isinstance(obj, list):
        result = []
        for item in obj:
            result.extend(flatten_toml_fields(item))
        return result
    return [obj]


def read_updated_parameters(path: Path, modulus: int) -> tuple[dict[str, np.ndarray], int]:
    with path.open("rb") as handle:
        doc = tomllib.load(handle)
    if "return" not in doc:
        raise RuntimeError("No 'return' in Prover.toml. Confirm --overwrite-return and the debug circuit output.")
    flat = flatten_toml_fields(doc["return"])
    if len(flat) != EXPECTED_OUTPUT_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_OUTPUT_COUNT} returned Field elements, got {len(flat)}. "
                           "Check the Noir debug output order and train_batch return type.")
    result = {}
    at = 0
    for name, _, shape, _ in PARAMETERS:
        count = math.prod(shape)
        raw = np.fromiter((field_to_signed(x, modulus) for x in flat[at:at + count]),
                          dtype=object, count=count)
        result[name] = raw.reshape(shape, order="F")
        at += count
    loss_raw = field_to_signed(flat[-1], modulus)
    return result, loss_raw


def validate_level_bounds(parameters: dict[str, np.ndarray], w: int, d: int) -> None:
    for name, _, _, level in PARAMETERS:
        bound = 1 << (w + level * d)
        if any(abs(int(v)) >= bound for v in parameters[name].flat):
            raise ValueError(f"{name} exceeds the assumed W+level*D magnitude bound")


def decoded_float_params(parameters: dict[str, np.ndarray], d: int) -> dict[str, np.ndarray]:
    return {name: np.asarray(parameters[name], dtype=np.float64) / float(1 << (level * d))
            for name, _, _, level in PARAMETERS}


def probabilities(parameters: dict[str, np.ndarray], pixels: np.ndarray, d: int, chunk: int = 512) -> np.ndarray:
    p = decoded_float_params(parameters, d)
    result = np.empty((len(pixels), OUTPUT_SIZE), dtype=np.float64)
    for start in range(0, len(pixels), chunk):
        x = pixels[start:start + chunk].astype(np.float64) / 256.0  # Matches the Noir circuit.
        a1 = np.maximum(x @ p["W1"] + p["b1"], 0.0)
        a2 = np.maximum(a1 @ p["W2"] + p["b2"], 0.0)
        logits = a2 @ p["W3"] + p["b3"]
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        result[start:start + len(x)] = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return result


def cross_entropy(prob: np.ndarray, labels: np.ndarray) -> float:
    correct = np.clip(prob[np.arange(len(labels)), labels.astype(int)], 1e-12, 1.0)
    return float(-np.log(correct).mean())


def classification_metrics(prob: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = prob.argmax(axis=1)
    cm = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.int64)
    np.add.at(cm, (labels.astype(int), predictions), 1)
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    actual = cm.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros(OUTPUT_SIZE), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros(OUTPUT_SIZE), where=actual > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(OUTPUT_SIZE), where=(precision + recall) > 0)
    return {
        "accuracy": float((predictions == labels).mean()),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
    }


def write_checkpoint(path: Path, parameters: dict[str, np.ndarray], next_step: int,
                     config: dict, subset_digest: str) -> None:
    payload = {"next_step": next_step, "config": config, "subset_sha256": subset_digest,
               "parameters": {name: [str(int(v)) for v in parameters[name].flatten(order="F")]
                              for name, _, _, _ in PARAMETERS}}
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    os.replace(tmp, path)


def read_checkpoint(path: Path, config: dict, subset_digest: str) -> tuple[dict[str, np.ndarray], int]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload["config"] != config or payload["subset_sha256"] != subset_digest:
        raise ValueError("Checkpoint configuration or sampled MNIST subset changed")
    params = {}
    for name, _, shape, _ in PARAMETERS:
        raw = payload["parameters"][name]
        params[name] = np.fromiter((int(v) for v in raw), dtype=object).reshape(shape, order="F")
    return params, int(payload["next_step"])


def run_nargo(project: Path, nargo: str, log_path: Path, action: str) -> float:
    if action == "compile":
        command = [nargo, "compile"]
    elif action == "execute":
        command = [nargo, "execute", "--overwrite-return", "ssfp_train_step"]
    else:
        raise ValueError(action)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT, check=False)
    duration = time.perf_counter() - start
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise RuntimeError(f"Nargo {action} exited {completed.returncode}. Log: {log_path}\n{tail}")
    return duration


def append_metrics(path: Path, row: dict[str, object]) -> None:
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def save_plots(metrics_path: Path, workdir: Path) -> None:
    """Plot per-step loss and accuracy if matplotlib is installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; metrics.csv still contains the full history")
        return
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    step = np.array([int(r["step"]) for r in rows])
    noir_loss = np.array([float(r["noir_loss_pre"]) for r in rows])
    np_pre = np.array([float(r["numpy_loss_pre"]) for r in rows])
    np_post = np.array([float(r["numpy_loss_post"]) for r in rows])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(step, noir_loss, label="SSFP loss (pre-update)", alpha=0.7)
    ax.plot(step, np_pre, label="NumPy loss (pre-update)", alpha=0.6)
    ax.plot(step, np_post, label="NumPy loss (post-update)", alpha=0.6)
    if len(step) >= 5:
        window = min(10, len(step))
        running = np.convolve(noir_loss, np.ones(window) / window, mode="valid")
        ax.plot(step[window - 1:], running, label=f"SSFP rolling mean ({window})", linewidth=2)
    ax.set(xlabel="Training step", ylabel="Cross-entropy loss", title="SSFP FCNN training loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(workdir / "loss_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(step, [float(r["train_accuracy"]) for r in rows], label="Train accuracy")
    ax.plot(step, [float(r["test_accuracy"]) for r in rows], label="Test accuracy")
    ax.set(xlabel="Training step", ylabel="Accuracy", title="SSFP FCNN classification accuracy", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(workdir / "accuracy_curve.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, required=True, help="Path to the Noir project (Nargo.toml)")
    ap.add_argument("--mnist-dir", type=Path, required=True, help="Path to the 4 MNIST .gz files")
    ap.add_argument("--train-samples", type=int, default=40, help="Multiple of 4 (default: 40)")
    ap.add_argument("--test-samples", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--w", type=int, default=32)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--field-modulus", type=int, default=BN254_MODULUS)
    ap.add_argument("--model-seed", type=int, default=42)
    ap.add_argument("--shuffle-seed", type=int, default=42)
    ap.add_argument("--subset-seed", type=int, default=42)
    ap.add_argument("--nargo", default="nargo")
    ap.add_argument("--workdir", type=Path, default=None, help="Outputs/checkpoints directory (default: <project>/ssfp_training)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Only write one Prover.toml, without invoking nargo")
    args = ap.parse_args()

    project = args.project.expanduser().resolve()
    mnist_dir = args.mnist_dir.expanduser().resolve()
    workdir = (args.workdir or (project / "ssfp_training")).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if not (project / "Nargo.toml").exists():
        ap.error(f"No Nargo.toml in {project}")
    if args.train_samples % BATCH_SIZE != 0 or args.train_samples <= 0:
        ap.error("--train-samples must be a positive multiple of BATCH_SIZE=4")
    if args.epochs < 1 or args.d < 1 or args.w < 1 or args.test_samples < 1:
        ap.error("Invalid epoch, precision or test subset size")
    if args.field_modulus <= 2 ** (args.w + 2 * args.d + 2):
        ap.error("Field modulus is too small for chosen fixed-point configuration")

    x_train_all, y_train_all, x_test_all, y_test_all = load_mnist(mnist_dir)
    train_indices = balanced_subset_indices(y_train_all, args.train_samples, args.subset_seed)
    test_indices = balanced_subset_indices(y_test_all, args.test_samples, args.subset_seed + 1)
    subset_digest = hashlib.sha256(train_indices.tobytes() + test_indices.tobytes()).hexdigest()
    x_train, y_train = x_train_all[train_indices], y_train_all[train_indices]
    x_test, y_test = x_test_all[test_indices], y_test_all[test_indices]

    # Do not include epoch count here, to support extending training after a resume.
    config = {"w": args.w, "d": args.d, "field_modulus": str(args.field_modulus),
              "train_samples": args.train_samples, "test_samples": args.test_samples,
              "model_seed": args.model_seed, "subset_seed": args.subset_seed,
              "shuffle_seed": args.shuffle_seed, "batch_size": BATCH_SIZE,
              "network": [INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE]}
    checkpoint_path = workdir / "checkpoint.json.gz"
    metrics_path = workdir / "metrics.csv"
    if args.resume:
        if not checkpoint_path.exists():
            ap.error(f"No checkpoint at {checkpoint_path}")
        parameters, next_step = read_checkpoint(checkpoint_path, config, subset_digest)
    else:
        if checkpoint_path.exists() and not args.dry_run:
            ap.error(f"Checkpoint already exists at {checkpoint_path}. Use --resume or change --workdir")
        parameters, next_step = initial_parameters(args.model_seed, args.d), 0
    validate_level_bounds(parameters, args.w, args.d)
    steps_per_epoch = args.train_samples // BATCH_SIZE
    total_steps = steps_per_epoch * args.epochs
    if total_steps > MAX_TRAINING_BATCHES:
        ap.error(f"Training requests {total_steps} steps, but the W1 range proof supports at most {MAX_TRAINING_BATCHES}")
    if next_step >= total_steps:
        print(f"Nothing to do: already completed {next_step} of {total_steps} steps")
        return

    prover_path = project / "Prover.toml"
    print(f"Network: {INPUT_SIZE}->{HIDDEN_SIZE_1}->{HIDDEN_SIZE_2}->{OUTPUT_SIZE}; "
          f"parameters: {TOTAL_PARAMETERS}; batch size: {BATCH_SIZE}")
    print(f"MNIST: train={len(x_train)}, test={len(x_test)}; "
          f"steps per epoch={steps_per_epoch}; total steps={total_steps}")
    print(f"Fixed-point: W={args.w}, D={args.d}, W1 level=2, others level=1")
    if not args.dry_run:
        compile_s = run_nargo(project, args.nargo, workdir / "compile.log", "compile")
        print(f"Compilation completed in {compile_s:.2f}s", flush=True)

    for step in range(next_step, total_steps):
        epoch, batch_index = divmod(step, steps_per_epoch)
        # Independent epoch-specific, deterministic permutations; resume-safe.
        order = np.random.default_rng(args.shuffle_seed + epoch).permutation(args.train_samples)
        selected = order[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
        batch_pixels = x_train[selected]
        batch_labels = y_train[selected]
        write_prover(prover_path, batch_pixels, batch_labels, parameters, args.field_modulus)
        if args.dry_run:
            print(f"Wrote {prover_path} ({prover_path.stat().st_size:,} bytes) for step {step}")
            return

        pre_prob = probabilities(parameters, batch_pixels, args.d)
        pre_loss = cross_entropy(pre_prob, batch_labels)
        elapsed = run_nargo(project, args.nargo, workdir / "last_execute.log", "execute")
        updated, loss_raw = read_updated_parameters(prover_path, args.field_modulus)
        validate_level_bounds(updated, args.w, args.d)
        noir_loss = loss_raw / float(1 << args.d)  # Debug circuit resets loss to level 1.
        post_loss = cross_entropy(probabilities(updated, batch_pixels, args.d), batch_labels)
        parameters = updated

        # Save raw integer parameter state before recording metrics.
        write_checkpoint(checkpoint_path, parameters, step + 1, config, subset_digest)
        train_scores = classification_metrics(probabilities(parameters, x_train, args.d), y_train)
        test_scores = classification_metrics(probabilities(parameters, x_test, args.d), y_test)
        row = {"step": step + 1, "epoch": epoch + 1, "batch": batch_index + 1,
               "elapsed_s": round(elapsed, 3), "noir_loss_pre": round(noir_loss, 7),
               "numpy_loss_pre": round(pre_loss, 7), "numpy_loss_post": round(post_loss, 7),
               **{f"train_{k}": round(v, 6) for k, v in train_scores.items()},
               **{f"test_{k}": round(v, 6) for k, v in test_scores.items()}}
        append_metrics(metrics_path, row)
        print(f"epoch {epoch + 1}/{args.epochs} batch {batch_index + 1}/{steps_per_epoch} "
              f"| Noir loss(pre) {noir_loss:.5f} | NumPy loss {pre_loss:.5f}->{post_loss:.5f} "
              f"| train acc {train_scores['accuracy']:.1%} | test acc {test_scores['accuracy']:.1%} "
              f"| {elapsed:.2f}s", flush=True)
        if batch_index == steps_per_epoch - 1:
            save_plots(metrics_path, workdir)
            print(f"  epoch {epoch + 1}: test macro precision={test_scores['precision_macro']:.3f}, "
                  f"recall={test_scores['recall_macro']:.3f}, F1={test_scores['f1_macro']:.3f}", flush=True)

    save_plots(metrics_path, workdir)
    print(f"Completed {total_steps} steps. Metrics: {metrics_path}; checkpoint: {checkpoint_path}; plots: {workdir}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
