from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def choose_device() -> str | int:
    import torch

    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_yolo_model(model_name: str, model_dir: Path) -> Any:
    from ultralytics import YOLO

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_name
    if model_path.exists():
        return YOLO(str(model_path))

    old_cwd = Path.cwd()
    try:
        os.chdir(model_dir)
        return YOLO(model_name)
    finally:
        os.chdir(old_cwd)


def extract_track_outputs(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if result.boxes is None or result.boxes.id is None:
        return np.empty((0, 4), dtype=float), np.empty((0,), dtype=int), np.empty((0,), dtype=float)

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    ids = result.boxes.id.detach().cpu().numpy().astype(int)
    if result.boxes.conf is None:
        confs = np.ones(len(ids), dtype=float)
    else:
        confs = result.boxes.conf.detach().cpu().numpy()

    return boxes, ids, confs
