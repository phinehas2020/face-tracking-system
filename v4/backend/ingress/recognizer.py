"""Which recognizer turns an accepted face into a template.

buffalo_l (InsightFace w600k_r50) is the default and needs nothing here: the face
detector returns its template. INGRESS_FACE_RECOGNIZER=kprpe_wf12m swaps in the
AdaFace ViT KP-RPE model from the CVLFace repository under INGRESS_FACE_MODELS_ROOT.
Templates from different recognizers are not comparable, so each is stored under its
own model name and a station only ever loads its own.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np

MODELS = {
    "buffalo_l": "buffalo_l/w600k_r50",
    "kprpe_wf12m": "adaface/vit_base_kprpe_webface12m",
}
REPOSITORIES = {"kprpe_wf12m": "cvlface_adaface_vit_base_kprpe_webface12m"}
SELECTED = os.environ.get("INGRESS_FACE_RECOGNIZER", "buffalo_l")
if SELECTED not in MODELS:
    raise ValueError(f"INGRESS_FACE_RECOGNIZER must be one of {sorted(MODELS)}")
FACE_MODEL = MODELS[SELECTED]

_loaded: Recognizer | None = None
_load_lock = threading.Lock()


def aligned_face(image_bgr: np.ndarray, kps) -> tuple[np.ndarray, np.ndarray]:
    """The 112 px ArcFace crop and its five landmarks as fractions of that crop."""
    from insightface.utils import face_align

    points = np.asarray(kps, dtype=np.float32)
    matrix = face_align.estimate_norm(points, 112)
    crop = face_align.norm_crop(image_bgr, points, 112)
    return crop, ((points @ matrix[:, :2].T + matrix[:, 2]) / 112.0).astype(np.float32)


class Recognizer:
    def __init__(self, name: str, root: Path, device: str) -> None:
        import torch
        import yaml
        from omegaconf import OmegaConf

        self.name, self.device = name, device
        path = (Path(root) / REPOSITORIES[name]).resolve()
        # The CVLFace repository imports its own top-level packages and reads files
        # relative to itself, so it is loaded from inside its directory, once, before
        # any camera thread starts.
        cwd = os.getcwd()
        os.chdir(path)
        sys.path.insert(0, str(path))
        try:
            text = (path / "pretrained_model/model.yaml").read_text()
            conf = OmegaConf.create(yaml.safe_load(text))
            from models import get_model

            model = get_model(conf)
            model.load_state_dict_from_path(str(path / "pretrained_model" / "model.pt"))
        finally:
            os.chdir(cwd)
            sys.path.remove(str(path))
        self.model = model.eval().to(device)
        self.torch = torch
        self.embed(np.zeros((112, 112, 3), dtype=np.uint8), np.full((5, 2), 0.5, dtype=np.float32))

    def embed(self, aligned_bgr: np.ndarray, kps_fraction: np.ndarray) -> np.ndarray:
        """One aligned crop in, one unit-length template out."""
        from .vision.accelerator import mps_guard

        torch = self.torch
        pixels = torch.from_numpy(np.ascontiguousarray(aligned_bgr[..., ::-1])).float()
        pixels = ((pixels / 255 - 0.5) / 0.5).permute(2, 0, 1)[None].contiguous()
        points = torch.from_numpy(np.asarray(kps_fraction, dtype=np.float32))[None]
        with mps_guard(self.device), torch.no_grad():
            out = self.model(pixels.to(self.device), points.to(self.device))
            if isinstance(out, (tuple, list)):
                out = out[0]
            vector = out.float().cpu().numpy()[0]
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or norm <= 0:
            raise ValueError("recognizer returned an unusable template")
        return (vector / norm).astype(np.float32)


def load() -> Recognizer | None:
    """The selected recognizer, loaded once per process; None means buffalo_l.

    A failure raises: a station must never store one recognizer's templates under
    another's name.
    """
    global _loaded
    if SELECTED == "buffalo_l":
        return None
    with _load_lock:
        if _loaded is None:
            import torch

            root = os.environ.get("INGRESS_FACE_MODELS_ROOT")
            if not root:
                raise RuntimeError("INGRESS_FACE_MODELS_ROOT is required for " + SELECTED)
            device = os.environ.get("INGRESS_FACE_RECOGNIZER_DEVICE") or (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
            _loaded = Recognizer(SELECTED, Path(root), device)
        return _loaded
