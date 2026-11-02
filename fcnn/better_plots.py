#!/usr/bin/env python3
"""Create two square loss plots and the original landscape accuracy plot."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set this to the directory that contains metrics.csv.
METRICS_DIR = Path("./ssfp_train_data")


def main():
    csv_path = METRICS_DIR / "metrics.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "step", "numpy_loss_pre", "numpy_loss_post", "noir_loss_pre", "validation_loss",
            "train_accuracy", "test_accuracy",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("metrics.csv contains no data rows")

    steps = np.array([int(row["step"]) for row in rows])
    numpy_pre = np.array([float(row["numpy_loss_pre"]) for row in rows])
    numpy_post = np.array([float(row["numpy_loss_post"]) for row in rows])
    ssfp_pre = np.array([float(row["noir_loss_pre"]) for row in rows])
    val_loss = np.array([float(row["validation_loss"]) for row in rows])
    train_acc = np.array([float(row["train_accuracy"]) for row in rows])
    test_acc = np.array([float(row["test_accuracy"]) for row in rows])

    # Graph 1: pre-update training losses.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    marker_stride = max(1, len(steps) // 24)
    ax.plot(
        steps, ssfp_pre,
        label="Noir / SSFP (pre-update)",
        color="#2ca02c",
        linewidth=1.45,
        alpha=0.88,
        linestyle="-",
        zorder=3
    )
    ax.plot(
        steps, numpy_pre,
        label="NumPy (pre-update)",
        color="#d62728",
        linewidth=1.0,
        alpha=0.9,
        linestyle=":",
        zorder=4
    )
    ax.set(title="Training loss: NumPy vs SSFP", xlabel="Training step", ylabel="Cross-entropy loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    output1 = METRICS_DIR / "training_loss_comparison.png"
    fig.savefig(output1, dpi=200)
    plt.close(fig)

    # Graph 2: post-update NumPy training-batch loss vs post-update validation loss.
    fig, ax = plt.subplots(figsize=(6.5, 6.5), layout="constrained")
    ax.set_box_aspect(1)
    ax.plot(steps, numpy_post, label="NumPy train batch (post-update)", linewidth=1.3)
    ax.plot(steps, val_loss, label="NumPy validation (post-update)", linewidth=1.3)
    ax.set(title="NumPy training vs validation loss", xlabel="Training step", ylabel="Cross-entropy loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    output2 = METRICS_DIR / "train_vs_validation_loss.png"
    fig.savefig(output2, dpi=200)
    plt.close(fig)

    # Graph 3: original accuracy graph (8 x 4.5 inches, blue/orange colors).
    fig, ax = plt.subplots(figsize=(6.5, 6.5), layout="constrained")
    ax.set_box_aspect(1)
    ax.plot(steps, train_acc, label="Train accuracy", color="tab:blue")
    ax.plot(steps, test_acc, label="Test accuracy", color="tab:orange")
    ax.set(xlabel="Training step", ylabel="Accuracy",
           title="NumPy inference on SSFP-trained FCNN", ylim=(0, 1))
    ax.legend()
    output3 = METRICS_DIR / "accuracy_curve.png"
    fig.tight_layout()
    fig.savefig(output3, dpi=150)
    plt.close(fig)

    print(f"Saved {output1}\nSaved {output2}\nSaved {output3}")


if __name__ == "__main__":
    main()
