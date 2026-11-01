# Reads NIH ChestX-ray14 PNGs by image id.
# Run: library module, imported by the scripts

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from cxr_shortcuts.data.reserved_guard import assert_not_reserved

def default_image_loader(nih_root) -> Callable[[str], np.ndarray]:
    root = Path(nih_root)

    def load(image_id: str) -> np.ndarray:
        assert_not_reserved(image_id, "default_image_loader")
        return np.array(Image.open(root / image_id).convert("L"))

    return load

def read_image_gray(path, image_id: str | None = None) -> np.ndarray:
    p = Path(path)
    assert_not_reserved(image_id if image_id is not None else p.name, "read_image_gray")
    return np.array(Image.open(p).convert("L"))
