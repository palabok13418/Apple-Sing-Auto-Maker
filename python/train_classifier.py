"""Offline training scaffold for the tiny stem-purity classifier.

Expected labels: isolated_lead, isolated_backing, mixed_song, instrumental,
noisy_or_ambiguous. The production gate should optimize precision and reject
uncertain examples rather than forcing every input into an accepted class.
"""
from __future__ import annotations
import json
from pathlib import Path

FEATURES = ["rms", "zcr", "centroid", "flatness", "low_ratio", "harmonicity"]


def train(dataset: list[dict]) -> dict:
    if not dataset:
        raise ValueError("dataset is empty")
    return {"features": FEATURES, "classes": ["lead", "backing", "reject"], "note": "train with calibrated labeled stems"}


if __name__ == "__main__":
    out = Path("model-card.json")
    out.write_text(json.dumps(train([{"features": [0] * len(FEATURES), "label": "reject"}]), indent=2) + "\n", encoding="utf-8")
    print(out)
