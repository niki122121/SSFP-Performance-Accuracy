#!/usr/bin/env python3
"""Drive successive SSFP/Noir SGD steps through Prover.toml.

Requirements: Python >=3.11, numpy, nargo in PATH, MNIST IDX .gz files.
By default the Noir circuit can keep returning its original three Fields.
The Python harness instead reads the eight consecutive runtime prints:
loss, probabilities, W1, b1, W2, b2, W3, b3.
The prints are emitted by ssfp_context::append_ctx_println and the default
loss level is 2 (original SSFP loss). Optionally, --output-source return
supports a debug circuit returning flattened updated weights and loss.

This is an execution/training harness, not a proof verifier. It executes the
same Noir program for each batch, without generating cryptographic proofs.
Classification metrics use NumPy inference on Noir-updated fixed-point weights;
they are not direct measurements of Noir inference accuracy.
"""

from __future__ import annotations

import argparse
import ast
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
BATCH_SIZE = 16
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
DEFAULT_PRINT_LOSS_LEVEL = 2  # Verified from the original Noir runtime print.
METRIC_COLUMNS = (
    "step", "epoch", "batch", "elapsed_s", "noir_loss_pre", "numpy_loss_pre",
    "numpy_loss_post", "train_accuracy", "train_precision_macro",
    "train_recall_macro", "train_f1_macro", "test_accuracy",
    "test_precision_macro", "test_recall_macro", "test_f1_macro",
    "validation_loss",
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
    if np.any(raw >= OUTPUT_SIZE):
        raise ValueError(f"Invalid MNIST class label in {path}")
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
    n = field_int(value)
    if not (0 <= n < modulus):
        raise ValueError(f"Noncanonical returned Field value: {n}")
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
    if loss_raw < 0:
        raise ValueError(f"Noir returned a negative cross-entropy loss: {loss_raw}")
    return result, loss_raw



def read_printed_parameters(path: Path, modulus: int) -> tuple[dict[str, np.ndarray], int]:
    """Parse the eight original ssfp_context::append_ctx_println outputs.

    Expected order: loss, probabilities[BATCH_SIZE][OUTPUT_SIZE], W1, b1,
    W2, b2, W3, b3. Matrices are printed in their natural 2-D row-major
    shape, unlike the optional flattened return values (column-major).
    Compiler warnings and the three-element circuit return are ignored.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    names = ("probabilities", "W1", "b1", "W2", "b2", "W3", "b3")
    expected = {"probabilities": (BATCH_SIZE, OUTPUT_SIZE)}
    expected.update({name: shape for name, _, shape, _ in PARAMETERS})
    for i in range(len(lines) - 7):
        if not lines[i].strip().startswith("0x"):
            continue
        if not all(lines[i + j].lstrip().startswith("[[") for j in range(1, 8)):
            continue
        try:
            loss_raw = field_to_signed(int(lines[i].strip(), 16), modulus)
            if loss_raw < 0:
                raise ValueError("Negative cross-entropy loss")
            tensors = {}
            for offset, name in enumerate(names, 1):
                value = ast.literal_eval(lines[i + offset].strip())
                rows, cols = expected[name]
                if not isinstance(value, list) or len(value) != rows:
                    raise ValueError(f"{name} expected {rows} rows, got {len(value) if isinstance(value, list) else type(value).__name__}")
                if any(not isinstance(row, list) or len(row) != cols for row in value):
                    raise ValueError(f"{name} expected {cols} columns per row")
                tensors[name] = value
            result = {}
            for name, _, shape, _ in PARAMETERS:
                # Here the logs preserve rows and columns directly: no order='F'.
                result[name] = np.fromiter(
                    (field_to_signed(v, modulus) for row in tensors[name] for v in row),
                    dtype=object, count=math.prod(shape)).reshape(shape)
            # Check all prints were decoded; this catches noncanonical Fields.
            for row in tensors["probabilities"]:
                for v in row:
                    field_to_signed(v, modulus)
            return result, loss_raw
        except (ValueError, TypeError, SyntaxError, MemoryError) as exc:
            raise RuntimeError(f"Malformed SSFP runtime prints near line {i + 1} of {path}: {exc}") from exc
    raise RuntimeError(
        f"Could not find eight consecutive SSFP prints in {path}. "
        "Keep append_ctx_println(loss.v), probabilities.mat, W1, b1, W2, b2, W3, b3 "
        "in that exact order and ensure nargo execute prints runtime values."
    )


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


def noir_source_fingerprint(project: Path) -> str:
    """Detect changes to the local Noir program across resume operations.

    External git/path dependency versions are not included; pin those separately.
    """
    source_files = [project / "Nargo.toml", *sorted((project / "src").rglob("*.nr"))]
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(str(path.relative_to(project)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


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


def reconcile_metrics(path: Path, completed_steps: int) -> None:
    """Rollback metrics rows that were written before an interrupted checkpoint.

    The checkpoint is committed after the corresponding CSV row. If a process
    dies between these writes, the uncheckpointed step will be replayed on resume.
    """
    if not path.exists():
        if completed_steps:
            raise RuntimeError("Checkpoint exists but metrics.csv is missing; cannot resume cleanly")
        return
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        old_columns = reader.fieldnames
    steps = [int(r["step"]) for r in rows]
    if steps != list(range(1, len(steps) + 1)):
        raise RuntimeError("metrics.csv has missing, duplicate, or out-of-order steps")
    if len(rows) < completed_steps:
        raise RuntimeError("Checkpoint is ahead of metrics.csv; cannot resume cleanly")
    if len(rows) > completed_steps or old_columns == list(METRIC_COLUMNS[:-1]):
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
            writer.writeheader()
            writer.writerows(rows[:completed_steps])
        os.replace(tmp, path)
        if len(rows) > completed_steps:
            print(f"Discarded {len(rows)-completed_steps} uncheckpointed metric row(s) before resume")


def self_test() -> None:
    """Exercise serialization, field signs, column-major order, and recovery."""
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mod = BN254_MODULUS
        params = initial_parameters(42, 16)
        params["W1"][0, 0], params["W1"][1, 0], params["W1"][0, 1] = 11, -12, 13
        params["b3"][0, 0], params["b3"][0, 1] = -5, 7
        prover = root / "Prover.toml"
        write_prover(prover, np.zeros((BATCH_SIZE, INPUT_SIZE), dtype=np.uint8),
                     np.arange(BATCH_SIZE, dtype=np.uint8), params, mod)
        with prover.open("rb") as handle:
            doc = tomllib.load(handle)
        assert doc["W1_raw"][0][0] == str(11)
        assert doc["W1_raw"][1][0] == str(mod - 12)
        assert doc["b3_raw"][0] == str(mod - 5)
        assert len(doc["data_train"]) == BATCH_SIZE
        flat = []
        for name, _, _, _ in PARAMETERS:
            flat.extend(encode_field(int(v), mod) for v in params[name].flatten(order="F"))
        flat.append(encode_field(123456, mod))
        with prover.open("a", encoding="utf-8") as handle:
            handle.write("return = [" + ", ".join(flat) + "]\n")
        decoded, loss = read_updated_parameters(prover, mod)
        assert loss == 123456
        for name, _, _, _ in PARAMETERS:
            assert np.array_equal(decoded[name], params[name]), name
        # Simulate actual Noir-style runtime prints, including unrelated warnings.
        print_path = root / "nargo_execute.log"
        with print_path.open("w", encoding="utf-8") as handle:
            handle.write("warning: Return variable contains a constant value\n")
            handle.write("0x24d72ca10\n")
            handle.write(str([[0] * OUTPUT_SIZE for _ in range(BATCH_SIZE)]) + "\n")
            for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
                tensor = [[int(v) % mod for v in row] for row in params[name].tolist()]
                handle.write(str(tensor) + "\n")
            handle.write("Prover.toml return = [0,0,0]\n")
        printed, printed_loss = read_printed_parameters(print_path, mod)
        assert printed_loss == int("24d72ca10", 16)
        for name, _, _, _ in PARAMETERS:
            assert np.array_equal(printed[name], params[name]), name
        with patch("builtins.print"):
            metrics = root / "metrics.csv"
            for step in (1, 2):
                append_metrics(metrics, {"step": step})
            reconcile_metrics(metrics, 1)
        with metrics.open(newline="", encoding="utf-8") as handle:
            assert [int(r["step"]) for r in csv.DictReader(handle)] == [1]
        assert abs(cross_entropy(probabilities(params, np.zeros((BATCH_SIZE, INPUT_SIZE), dtype=np.uint8), 16),
                                 np.zeros(BATCH_SIZE, dtype=np.uint8)) - 2.302585093) < 0.1
    print("Self-test passed: TOML, negative Fields, both print and return decoding, metrics recovery, NumPy evaluation")




def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, help="Path to the Noir project (Nargo.toml)")
    ap.add_argument("--mnist-dir", type=Path, help="Path to the 4 MNIST .gz files")
    ap.add_argument("--train-samples", type=int, default=1600, help="Positive multiple of BATCH_SIZE (default: 1600)")
    ap.add_argument("--test-samples", type=int, default=2000, help="Positive multiple of BATCH_SIZE (default: 2000)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--w", type=int, default=32)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--field-modulus", type=int, default=BN254_MODULUS)
    ap.add_argument("--output-source", choices=("prints", "return"), default="prints",
                    help="Read updated weights from eight runtime prints (default) or flattened circuit return")
    ap.add_argument("--loss-level", type=int, default=None,
                    help="SSFP level of the printed/returned loss (default: 2 for prints, 1 for return)")
    ap.add_argument("--max-steps", type=int, default=None, help="Run at most this many NEW batches; useful for one-batch smoke tests")
    ap.add_argument("--skip-compile", action="store_true", help="Skip explicit nargo compile (execute still checks the circuit)")
    ap.add_argument("--self-test", action="store_true", help="Test the Python harness without MNIST or Nargo")
    ap.add_argument("--model-seed", type=int, default=42)
    ap.add_argument("--shuffle-seed", type=int, default=42)
    ap.add_argument("--subset-seed", type=int, default=42)
    ap.add_argument("--nargo", default="nargo")
    ap.add_argument("--workdir", type=Path, default=None, help="Outputs/checkpoints directory (default: <project>/ssfp_training)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Only write one Prover.toml, without invoking nargo")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.project is None or args.mnist_dir is None:
        ap.error("--project and --mnist-dir are required unless using --self-test")
    if args.loss_level is None:
        args.loss_level = DEFAULT_PRINT_LOSS_LEVEL if args.output_source == "prints" else 1
    if args.loss_level < 1:
        ap.error("--loss-level must be >= 1")
    if args.max_steps is not None and args.max_steps < 1:
        ap.error("--max-steps must be positive")

    project = args.project.expanduser().resolve()
    mnist_dir = args.mnist_dir.expanduser().resolve()
    workdir = (args.workdir or (project / "ssfp_training")).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if not (project / "Nargo.toml").exists():
        ap.error(f"No Nargo.toml in {project}")
    if args.train_samples <= 0 or args.train_samples % BATCH_SIZE != 0:
        ap.error(f"--train-samples must be a positive multiple of BATCH_SIZE={BATCH_SIZE}")
    if args.test_samples <= 0 or args.test_samples % BATCH_SIZE != 0:
        ap.error(f"--test-samples must be a positive multiple of BATCH_SIZE={BATCH_SIZE}")
    if args.epochs < 1 or args.d < 1 or args.w < 1:
        ap.error("Invalid epoch or precision")
    if args.field_modulus <= 2 ** (args.w + 2 * args.d + 2):
        ap.error("Field modulus is too small for chosen fixed-point configuration")

    x_train_all, y_train_all, x_test_all, y_test_all = load_mnist(mnist_dir)
    train_indices = balanced_subset_indices(y_train_all, args.train_samples, args.subset_seed)
    test_indices = balanced_subset_indices(y_test_all, args.test_samples, args.subset_seed + 1)
    x_train, y_train = x_train_all[train_indices], y_train_all[train_indices]
    x_test, y_test = x_test_all[test_indices], y_test_all[test_indices]
    digest = hashlib.sha256()
    for a in (train_indices, test_indices, x_train, y_train, x_test, y_test):
        digest.update(a.tobytes())
    subset_digest = digest.hexdigest()

    # Do not include epoch count here, to support extending training after a resume.
    config = {"w": args.w, "d": args.d, "field_modulus": str(args.field_modulus),
              "train_samples": args.train_samples, "test_samples": args.test_samples,
              "model_seed": args.model_seed, "subset_seed": args.subset_seed,
              "shuffle_seed": args.shuffle_seed, "batch_size": BATCH_SIZE,
              "loss_level": args.loss_level, "output_source": args.output_source,
              "noir_source_sha256": noir_source_fingerprint(project),
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
    if args.resume:
        reconcile_metrics(metrics_path, next_step)
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
    print(f"Updated weights: {args.output_source}; Noir loss level: {args.loss_level}")
    if not args.dry_run and not args.skip_compile:
        compile_s = run_nargo(project, args.nargo, workdir / "compile.log", "compile")
        print(f"Compilation completed in {compile_s:.2f}s", flush=True)

    end_step = min(total_steps, next_step + args.max_steps) if args.max_steps is not None else total_steps
    for step in range(next_step, end_step):
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
        if args.output_source == "prints":
            updated, loss_raw = read_printed_parameters(workdir / "last_execute.log", args.field_modulus)
        else:
            updated, loss_raw = read_updated_parameters(prover_path, args.field_modulus)
        validate_level_bounds(updated, args.w, args.d)
        noir_loss = loss_raw / float(1 << (args.d * args.loss_level))
        if not np.isfinite(noir_loss) or (pre_loss > 0.05 and not 0.1 <= noir_loss / pre_loss <= 10.0):
            raise ValueError(f"Noir loss {noir_loss:.8g} differs by over 10x from NumPy {pre_loss:.8g}; "
                             "check --loss-level, decoded output order, and circuit arithmetic")
        post_loss = cross_entropy(probabilities(updated, batch_pixels, args.d), batch_labels)
        parameters = updated

        train_scores = classification_metrics(probabilities(parameters, x_train, args.d), y_train)
        test_prob = probabilities(parameters, x_test, args.d)
        test_scores = classification_metrics(test_prob, y_test)
        validation_loss = cross_entropy(test_prob, y_test)
        row = {"step": step + 1, "epoch": epoch + 1, "batch": batch_index + 1,
               "elapsed_s": round(elapsed, 3), "noir_loss_pre": round(noir_loss, 7),
               "numpy_loss_pre": round(pre_loss, 7), "numpy_loss_post": round(post_loss, 7),
               "validation_loss": round(validation_loss, 7),
               **{f"train_{k}": round(v, 6) for k, v in train_scores.items()},
               **{f"test_{k}": round(v, 6) for k, v in test_scores.items()}}
        append_metrics(metrics_path, row)
        # Commit checkpoint last: if interrupted, an uncheckpointed row is rolled back on resume.
        write_checkpoint(checkpoint_path, parameters, step + 1, config, subset_digest)
        print(f"epoch {epoch + 1}/{args.epochs} batch {batch_index + 1}/{steps_per_epoch} "
              f"| Noir loss(pre) {noir_loss:.5f} | NumPy loss {pre_loss:.5f}->{post_loss:.5f} "
              f"| val loss {validation_loss:.5f} "
              f"| NumPy-eval train acc {train_scores['accuracy']:.1%} | test acc {test_scores['accuracy']:.1%} "
              f"| {elapsed:.2f}s", flush=True)
        if batch_index == steps_per_epoch - 1:
            print(f"  epoch {epoch + 1}: test macro precision={test_scores['precision_macro']:.3f}, "
                  f"recall={test_scores['recall_macro']:.3f}, F1={test_scores['f1_macro']:.3f}", flush=True)

    print(f"Completed {end_step} of {total_steps} planned steps. Metrics: {metrics_path}; checkpoint: {checkpoint_path}; plots: {workdir}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
