"""Train the low-end stem-purity classifier offline.

The browser gate is intentionally conservative: it can reject uncertain audio
without needing a server. This script trains a small softmax model from labeled
feature rows exported by the audio-analysis pipeline.

CSV columns:
  rms,zcr,centroid,flatness,low_ratio,harmonicity,vocal_activity,vocal_coverage,secondary_voice,label

Labels:
  lead, backing, reject

Example:
  python3 train_classifier.py --dataset features.csv --output stem-purity-model.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Iterable

FEATURES = [
    "rms",
    "zcr",
    "centroid",
    "flatness",
    "low_ratio",
    "harmonicity",
    "vocal_activity",
    "vocal_coverage",
    "secondary_voice",
]
CLASSES = ["lead", "backing", "reject"]


def _softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    exps = [math.exp(max(-60.0, x - peak)) for x in logits]
    total = sum(exps) or 1.0
    return [x / total for x in exps]


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    missing = [name for name in FEATURES + ["label"] if not rows or name not in rows[0]]
    if missing:
        raise ValueError("dataset is missing columns: " + ", ".join(missing))
    dataset: list[dict] = []
    for row in rows:
        label = row["label"].strip().lower()
        if label not in CLASSES:
            raise ValueError(f"unknown label: {label}")
        dataset.append({
            "features": [float(row[name]) for name in FEATURES],
            "label": label,
        })
    if not dataset:
        raise ValueError("dataset is empty")
    return dataset


def _standardize(dataset: list[dict]) -> tuple[list[list[float]], list[float], list[float]]:
    matrix = [item["features"] for item in dataset]
    means = [sum(row[i] for row in matrix) / len(matrix) for i in range(len(FEATURES))]
    scales = []
    for i in range(len(FEATURES)):
        variance = sum((row[i] - means[i]) ** 2 for row in matrix) / max(1, len(matrix) - 1)
        scales.append(math.sqrt(variance) or 1.0)
    normalized = [[(row[i] - means[i]) / scales[i] for i in range(len(FEATURES))] for row in matrix]
    return normalized, means, scales


def train(dataset: list[dict], epochs: int = 1400, learning_rate: float = 0.045, l2: float = 0.003) -> dict:
    if not dataset:
        raise ValueError("dataset is empty")

    x, means, scales = _standardize(dataset)
    class_index = {label: i for i, label in enumerate(CLASSES)}
    y = [class_index[item["label"]] for item in dataset]
    counts = [max(1, y.count(i)) for i in range(len(CLASSES))]
    total = float(sum(counts))
    # Inverse-frequency weighting keeps the reject class important without
    # hard-coding one dataset-specific penalty.
    class_weight = [total/(len(CLASSES)*count) for count in counts]
    class_weight[2] *= 1.35
    weights = [[0.0 for _ in FEATURES] for _ in CLASSES]
    bias = [0.0 for _ in CLASSES]
    best_loss = float("inf")
    stale = 0

    for _epoch in range(epochs):
        grad_w = [[0.0 for _ in FEATURES] for _ in CLASSES]
        grad_b = [0.0 for _ in CLASSES]
        loss = 0.0
        for row, target in zip(x, y):
            logits = [sum(w * value for w, value in zip(weights[c], row)) + bias[c] for c in range(len(CLASSES))]
            probs = _softmax(logits)
            sample_weight = class_weight[target]
            loss -= sample_weight * math.log(max(1e-12, probs[target]))
            for c in range(len(CLASSES)):
                error = (probs[c] - (1.0 if c == target else 0.0)) * sample_weight
                grad_b[c] += error
                for f in range(len(FEATURES)):
                    grad_w[c][f] += error * row[f]

        scale = 1.0 / len(x)
        for c in range(len(CLASSES)):
            bias[c] -= learning_rate * grad_b[c] * scale
            for f in range(len(FEATURES)):
                grad = grad_w[c][f] * scale + l2 * weights[c][f]
                weights[c][f] -= learning_rate * grad

        loss /= len(x)
        if loss + 1e-5 < best_loss:
            best_loss = loss
            stale = 0
        else:
            stale += 1
            if stale >= 80:
                break

    return {
        "version": 2,
        "model": "tiny-stem-purity-softmax",
        "features": FEATURES,
        "classes": CLASSES,
        "mean": means,
        "scale": scales,
        "weights": weights,
        "bias": bias,
        "training": {
            "epochs_max": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "early_stopping_patience": 80,
            "class_weighting": "inverse_frequency_with_reject_boost",
        },
        "policy": {
            "reject_ambiguous": True,
            "accept_requires_best_class_in": ["lead", "backing"],
            "minimum_probability": 0.82,
            "minimum_margin_over_second": 0.18,
        },
    }


def accuracy(model: dict, dataset: Iterable[dict]) -> float:
    correct = 0
    total = 0
    means = model["mean"]
    scales = model["scale"]
    for item in dataset:
        row = [(item["features"][i] - means[i]) / scales[i] for i in range(len(FEATURES))]
        logits = [
            sum(model["weights"][c][i] * row[i] for i in range(len(FEATURES))) + model["bias"][c]
            for c in range(len(CLASSES))
        ]
        predicted = CLASSES[max(range(len(CLASSES)), key=lambda i: logits[i])]
        correct += predicted == item["label"]
        total += 1
    return correct / max(1, total)


def stratified_split(dataset: list[dict], validation_fraction: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    buckets = {label: [] for label in CLASSES}
    for item in dataset:
        buckets[item["label"]].append(item)
    train_rows: list[dict] = []
    validation_rows: list[dict] = []
    for label in CLASSES:
        rows = buckets[label]
        rng.shuffle(rows)
        if len(rows) < 2:
            train_rows.extend(rows)
            continue
        cut = min(len(rows) - 1, max(1, round(len(rows) * validation_fraction)))
        validation_rows.extend(rows[:cut])
        train_rows.extend(rows[cut:])
    rng.shuffle(train_rows)
    rng.shuffle(validation_rows)
    return train_rows, validation_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("stem-purity-model.json"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_csv(args.dataset)
    train_rows, validation_rows = stratified_split(dataset, args.validation_fraction, args.seed)

    # Diagnostic model is fitted only on the training split so validation
    # accuracy is a real out-of-sample measurement.
    diagnostic_model = train(train_rows)
    train_acc = accuracy(diagnostic_model, train_rows)
    validation_acc = accuracy(diagnostic_model, validation_rows) if validation_rows else None

    # Export the final model fitted on every labeled row after the holdout
    # measurement has been collected.
    model = train(dataset)
    model["evaluation"] = {
        "rows": len(dataset),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "diagnostic_train_accuracy": round(train_acc, 6),
        "diagnostic_validation_accuracy": None if validation_acc is None else round(validation_acc, 6),
        "note": "Validation is a diagnostic only; production gating still requires independent stem-level test data and threshold calibration.",
    }
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    shown = "n/a" if validation_acc is None else f"{validation_acc:.3f}"
    print(f"trained {len(dataset)} rows -> {args.output} (diagnostic_train_accuracy={train_acc:.3f}, diagnostic_validation_accuracy={shown})")


if __name__ == "__main__":
    main()
