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


def train(dataset: list[dict], epochs: int = 900, learning_rate: float = 0.07, l2: float = 0.001) -> dict:
    if not dataset:
        raise ValueError("dataset is empty")

    x, means, scales = _standardize(dataset)
    class_index = {label: i for i, label in enumerate(CLASSES)}
    y = [class_index[item["label"]] for item in dataset]

    # Extra weight on reject examples makes false acceptance more expensive.
    class_weight = [1.0, 1.0, 1.7]
    weights = [[0.0 for _ in FEATURES] for _ in CLASSES]
    bias = [0.0 for _ in CLASSES]

    for _ in range(epochs):
        grad_w = [[0.0 for _ in FEATURES] for _ in CLASSES]
        grad_b = [0.0 for _ in CLASSES]
        for row, target in zip(x, y):
            logits = [sum(w * value for w, value in zip(weights[c], row)) + bias[c] for c in range(len(CLASSES))]
            probs = _softmax(logits)
            sample_weight = class_weight[target]
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

    return {
        "version": 1,
        "model": "tiny-stem-purity-softmax",
        "features": FEATURES,
        "classes": CLASSES,
        "mean": means,
        "scale": scales,
        "weights": weights,
        "bias": bias,
        "policy": {
            "reject_ambiguous": True,
            "reject_class_weight": 1.7,
            "accept_requires_best_class_in": ["lead", "backing"],
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("stem-purity-model.json"))
    args = parser.parse_args()

    dataset = load_csv(args.dataset)
    model = train(dataset)
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(f"trained {len(dataset)} rows -> {args.output} (accuracy={accuracy(model, dataset):.3f})")


if __name__ == "__main__":
    main()
