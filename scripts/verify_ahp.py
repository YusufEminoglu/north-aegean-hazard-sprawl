"""Recompute the published fusion weights and AHP consistency statistics."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "data" / "ahp" / "fusion_pairwise.csv"
WEIGHTS_PATH = ROOT / "data" / "ahp" / "weight_schemes.csv"
RANDOM_INDEX = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12}


def read_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    labels = rows[0][1:]
    matrix = np.asarray([[float(value) for value in row[1:]] for row in rows[1:]])
    return labels, matrix


def geometric_mean_weights(matrix: np.ndarray) -> np.ndarray:
    means = np.prod(matrix, axis=1) ** (1.0 / matrix.shape[0])
    return means / means.sum()


def consistency(matrix: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    n = matrix.shape[0]
    lambda_max = float(np.mean((matrix @ weights) / weights))
    ci = (lambda_max - n) / (n - 1)
    cr = ci / RANDOM_INDEX[n] if RANDOM_INDEX[n] else 0.0
    return lambda_max, ci, cr


def published_weights() -> np.ndarray:
    with WEIGHTS_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return np.asarray([float(row["W1_AHP"]) for row in rows])


def main() -> None:
    labels, matrix = read_matrix(MATRIX_PATH)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The AHP matrix must be square.")
    if not np.allclose(matrix * matrix.T, 1.0, atol=1e-9):
        raise ValueError("The AHP matrix is not reciprocal.")

    weights = geometric_mean_weights(matrix)
    lambda_max, ci, cr = consistency(matrix, weights)
    reported = published_weights()

    print("criterion,computed_weight,reported_weight")
    for label, computed, expected in zip(labels, weights, reported, strict=True):
        print(f"{label},{computed:.6f},{expected:.2f}")
    print(f"lambda_max,{lambda_max:.4f}")
    print(f"consistency_index,{ci:.4f}")
    print(f"consistency_ratio,{cr:.4f}")

    if cr >= 0.10:
        raise SystemExit("AHP consistency check failed: CR must be below 0.10.")
    if not np.allclose(weights, reported, atol=0.015):
        raise SystemExit("Computed fusion weights differ from the published rounded values.")


if __name__ == "__main__":
    main()
