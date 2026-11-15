"""Unchanged functions extracted to avoid historical experiment dependencies."""
from pathlib import Path
import numpy as np
from research_possession.boundary_decoder import BoundaryConfig, decode_boundary


def baseline(features, blocks, dead, boundary):
    labels = np.full(len(features), 'X', dtype='U1')
    evidence = {}
    for start, end in blocks:
        predicted, local = decode_boundary(features.sliced(start, end), dead[start:end], boundary[start:end],
                                          BoundaryConfig(status_weight=1.))
        labels[start:end] = predicted
        for name, values in local.items():
            if not isinstance(values, np.ndarray) or len(values) != end-start: continue
            if name not in evidence:
                evidence[name] = np.zeros((len(features), *values.shape[1:]), dtype=values.dtype)
            evidence[name][start:end] = values
    return labels, evidence


def write_csv(path, labels):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
