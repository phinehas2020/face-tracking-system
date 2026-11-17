"""Visible head anchors for cross-view geometry; never infer feet from cropped boxes."""

from __future__ import annotations

import numpy as np

from .vision.continuity import overlap


def head_anchor(box, poses, size):
    x, y, r, b = box
    w = max(r - x, 1)
    h = max(b - y, 1)
    candidates = []
    for pose in poses:
        points = np.asarray(pose["points"])
        if points.shape != (17, 3):
            continue
        if points[0, 2] >= 0.45:
            point = points[0, :2]
        elif points[1, 2] >= 0.4 and points[2, 2] >= 0.4:
            point = points[1:3, :2].mean(axis=0)
        else:
            continue
        if not (x - 0.1 * w <= point[0] <= r + 0.1 * w and y - 0.15 * h <= point[1] <= y + 0.2 * h):
            continue
        iou = overlap(box, pose["bbox"])
        if iou < 0.2:
            continue
        distance = float(np.linalg.norm((point - np.array([(x + r) / 2, y + 0.085 * h])) / [w, h]))
        candidates.append((0.7 * iou + 0.3 * max(0, 1 - distance), point))
    candidates.sort(key=lambda c: -c[0])
    if not candidates or (len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.04):
        return None
    return candidates[0][1] / np.asarray(size)
