# Helpers for cropping a patch to its mask; the interactive annotation tool in this file is not used by the audit.
# Run: python3 -m cxr_shortcuts.edits.drain.annotate_tool --help

from __future__ import annotations

import argparse
import functools
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Protocol, Sequence, Union

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from PIL import ImageDraw

from cxr_shortcuts.data.image_io import read_image_gray
from cxr_shortcuts.data.lung_masks import overlay_mask_on_image
from cxr_shortcuts.edits.drain.donor_library import (
    REQUIRED_DONOR_COLUMNS,
    donor_laterality_from_mask,
    donor_mask_width_metrics,
    mask_principal_axis_angle_deg,
    relativize_donor_path,
    validate_donor_record,
)

Point = tuple[int, int]
Box = tuple[int, int, int, int]
Prompt = Union[Point, Box]
ReviewAction = tuple[str, Optional[np.ndarray]]

class Segmenter(Protocol):
    def predict(self, image: np.ndarray, prompt: Prompt) -> np.ndarray: ...

def annotate_donor(
    image: np.ndarray,
    prompt: Prompt,
    segmenter: Segmenter,
    add: Optional[np.ndarray] = None,
    remove: Optional[np.ndarray] = None,
) -> np.ndarray:
    mask = segmenter.predict(image, prompt)
    return apply_correction(mask, add=add, remove=remove)

def apply_correction(
    mask: np.ndarray,
    add: Optional[np.ndarray] = None,
    remove: Optional[np.ndarray] = None,
) -> np.ndarray:
    corrected = mask.copy()
    if add is not None:
        corrected = corrected | add
    if remove is not None:
        corrected = corrected & ~remove
    return corrected

def save_donor_record(records_path: Path, record: Mapping) -> pd.DataFrame:
    validate_donor_record(record)

    records_path = Path(records_path)
    row = pd.DataFrame([dict(record)], columns=list(REQUIRED_DONOR_COLUMNS))

    if records_path.exists():
        existing = pd.read_csv(records_path)
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row

    records_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(records_path, index=False)
    return combined

def crop_to_mask_bbox(
    image: np.ndarray, mask: np.ndarray, padding: int = 0
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if not mask.any():
        raise ValueError("mask has no True pixels -- nothing to crop")
    rows, cols = np.nonzero(mask)
    r0 = max(int(rows.min()) - padding, 0)
    r1 = min(int(rows.max()) + padding + 1, image.shape[0])
    c0 = max(int(cols.min()) - padding, 0)
    c1 = min(int(cols.max()) + padding + 1, image.shape[1])
    return image[r0:r1, c0:c1], mask[r0:r1, c0:c1], (r0, c0)

def annotated_source_ids(records_path: Path) -> set[str]:
    records_path = Path(records_path)
    if not records_path.exists():
        return set()
    records = pd.read_csv(records_path)
    return set(records["source_image_id"].astype(str))

def _existing_record_count(records_path: Path) -> int:
    records_path = Path(records_path)
    if not records_path.exists():
        return 0
    return len(pd.read_csv(records_path))

def dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a_sum, b_sum = int(mask_a.sum()), int(mask_b.sum())
    if a_sum == 0 and b_sum == 0:
        return 1.0
    intersection = int((mask_a & mask_b).sum())
    return 2 * intersection / (a_sum + b_sum)

def select_review_subset(ids: Sequence[str], fraction: float, rng: np.random.Generator) -> list[str]:
    if not (0 < fraction <= 1):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    sorted_ids = sorted(ids)
    n = max(1, round(len(sorted_ids) * fraction))
    chosen = rng.choice(sorted_ids, size=n, replace=False)
    return sorted(chosen.tolist())

def export_masks_for_review(
    masks: dict[str, np.ndarray],
    output_dir: Path,
    fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset = select_review_subset(list(masks.keys()), fraction, rng)
    rows = []
    for donor_id in subset:
        path = output_dir / f"{donor_id}_mask.npy"
        np.save(path, masks[donor_id])
        rows.append({"donor_id": donor_id, "mask_path": str(path)})

    manifest = pd.DataFrame(rows, columns=["donor_id", "mask_path"])
    manifest.to_csv(output_dir / "review_manifest.csv", index=False)
    return manifest

class MedSAMSegmenter:

    def __init__(self, checkpoint_path: str, model_type: str = "vit_b", device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device
        self._predictor = None
        self._last_image: Optional[np.ndarray] = None

    def _predictor_instance(self):
        if self._predictor is None:
            import torch
            from segment_anything import SamPredictor, sam_model_registry

            sam = sam_model_registry[self.model_type](checkpoint=None)
            state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            sam.load_state_dict(state_dict)
            sam.to(self.device)
            self._predictor = SamPredictor(sam)
        return self._predictor

    def predict(self, image: np.ndarray, prompt: Prompt) -> np.ndarray:
        predictor = self._predictor_instance()

        if image is not self._last_image:

            image_rgb = np.stack([image] * 3, axis=-1) if image.ndim == 2 else image
            predictor.set_image(image_rgb)
            self._last_image = image
        if len(prompt) == 2:
            row, col = prompt
            masks, _, _ = predictor.predict(
                point_coords=np.array([[col, row]]),
                point_labels=np.array([1]),
                multimask_output=False,
            )
        else:
            row0, col0, row1, col1 = prompt
            masks, _, _ = predictor.predict(
                box=np.array([col0, row0, col1, row1]),
                multimask_output=False,
            )
        return masks[0].astype(bool)

_POINT_CLICK_TOLERANCE_PX = 5

_MIN_DISPLAY_INCHES = 6.4
_ASSUMED_SCREEN_DPI = 100

def _figsize_for_image(image: np.ndarray) -> tuple[float, float]:
    height, width = image.shape[:2]
    return (
        max(_MIN_DISPLAY_INCHES, width / _ASSUMED_SCREEN_DPI),
        max(_MIN_DISPLAY_INCHES, height / _ASSUMED_SCREEN_DPI),
    )

_ZOOM_FACTOR_PER_SCROLL = 1.3

_MIN_ZOOM_WINDOW_PX = 40.0

def _positioned_axis(a: float, b: float, cursor: float, span: float, full_min: float, full_max: float) -> tuple[float, float]:
    frac = 0.5 if b == a else (cursor - a) / (b - a)
    new_a = cursor - frac * span
    new_b = new_a + span

    lo, hi = min(new_a, new_b), max(new_a, new_b)
    if lo < full_min:
        shift = full_min - lo
        new_a, new_b = new_a + shift, new_b + shift
    lo, hi = min(new_a, new_b), max(new_a, new_b)
    if hi > full_max:
        shift = hi - full_max
        new_a, new_b = new_a - shift, new_b - shift
    return new_a, new_b

def _zoomed_view_limits(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    cursor_x: float,
    cursor_y: float,
    zoom_in: bool,
    image_shape: tuple[int, int],
    zoom_factor: float = _ZOOM_FACTOR_PER_SCROLL,
    min_window_px: float = _MIN_ZOOM_WINDOW_PX,
) -> tuple[tuple[float, float], tuple[float, float]]:
    height, width = image_shape[:2]
    step_factor = (1.0 / zoom_factor) if zoom_in else zoom_factor
    span_x = (xlim[1] - xlim[0]) * step_factor
    span_y = (ylim[1] - ylim[0]) * step_factor

    if zoom_in:
        k_x = min_window_px / abs(span_x) if span_x != 0 else 1.0
        k_y = min_window_px / abs(span_y) if span_y != 0 else 1.0
        k = max(1.0, k_x, k_y)
    else:
        k_x = width / abs(span_x) if span_x != 0 else 1.0
        k_y = height / abs(span_y) if span_y != 0 else 1.0
        k = min(1.0, k_x, k_y)

    new_xlim = _positioned_axis(xlim[0], xlim[1], cursor_x, span_x * k, -0.5, width - 0.5)
    new_ylim = _positioned_axis(ylim[0], ylim[1], cursor_y, span_y * k, -0.5, height - 0.5)
    return new_xlim, new_ylim

def _wire_scroll_zoom(fig, ax, image: np.ndarray) -> None:

    def on_scroll(event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        new_xlim, new_ylim = _zoomed_view_limits(
            ax.get_xlim(), ax.get_ylim(), event.xdata, event.ydata,
            zoom_in=(event.button == "up"), image_shape=image.shape,
        )
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)

def default_prompt_capture(image: np.ndarray) -> Prompt:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    fig, ax = plt.subplots(figsize=_figsize_for_image(image))
    ax.imshow(image, cmap="gray")
    ax.set_title("Drag a box around the drain, or click a single point. Scroll to zoom.")
    _wire_scroll_zoom(fig, ax, image)

    captured: dict = {}

    def on_select(eclick, erelease) -> None:
        x0, y0, x1, y1 = eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata
        if x0 is None or y0 is None or x1 is None or y1 is None:
            return

        (dx0, dy0) = ax.transData.transform((x0, y0))
        (dx1, dy1) = ax.transData.transform((x1, y1))
        if abs(dx1 - dx0) < _POINT_CLICK_TOLERANCE_PX and abs(dy1 - dy0) < _POINT_CLICK_TOLERANCE_PX:
            captured["prompt"] = (int(round((y0 + y1) / 2)), int(round((x0 + x1) / 2)))
        else:
            captured["prompt"] = (
                int(round(min(y0, y1))),
                int(round(min(x0, x1))),
                int(round(max(y0, y1))),
                int(round(max(x0, x1))),
            )
        plt.close(fig)

    _selector = RectangleSelector(ax, on_select, useblit=True, button=[1], interactive=False)
    plt.show()

    if "prompt" not in captured:
        raise RuntimeError("no point/box prompt captured -- window closed without a selection")
    return captured["prompt"]

_STROKE_BRUSH_WIDTH_PX = 12

def _drop_unset_vertices(vertices: Sequence[tuple]) -> list[tuple[float, float]]:
    return [(x, y) for x, y in vertices if x is not None and y is not None]

def _rasterize_stroke(
    vertices: Sequence[tuple[float, float]],
    shape: tuple[int, int],
    brush_width_px: int = _STROKE_BRUSH_WIDTH_PX,
) -> np.ndarray:
    rows, cols = shape
    vertices = _drop_unset_vertices(vertices)
    canvas = PILImage.new("1", (cols, rows), 0)
    if not vertices:
        return np.array(canvas, dtype=bool)
    draw = ImageDraw.Draw(canvas)
    if len(vertices) >= 2:
        draw.line(vertices, fill=1, width=brush_width_px, joint="curve")
    radius = brush_width_px / 2
    for x, y in (vertices[0], vertices[-1]):
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=1)
    return np.array(canvas, dtype=bool)

def _enclosed_region_mask(vertices: Sequence[tuple[float, float]], shape: tuple[int, int]) -> np.ndarray:
    from skimage.draw import polygon

    rows, cols = shape
    vertices = _drop_unset_vertices(vertices)
    if len(vertices) < 3:
        return np.zeros((rows, cols), dtype=bool)
    xs, ys = zip(*vertices)
    rr, cc = polygon(np.array(ys), np.array(xs), shape=(rows, cols))
    mask = np.zeros((rows, cols), dtype=bool)
    mask[rr, cc] = True
    return mask

def default_review_and_correct(
    image: np.ndarray,
    mask: np.ndarray,
    brush_width_px: int = _STROKE_BRUSH_WIDTH_PX,
) -> ReviewAction:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import LassoSelector

    overlay = overlay_mask_on_image(image, mask)
    fig, ax = plt.subplots(figsize=_figsize_for_image(image))
    ax.imshow(overlay)
    ax.set_title("Current mask (red overlay). Scroll to zoom. Close this window, then answer in the terminal.")
    _wire_scroll_zoom(fig, ax, image)
    plt.show()

    action = ""
    while action not in ("y", "a", "r", "p"):
        action = input(
            "Accept mask? [y]es / [a]dd stroke / [r]emove region / [p]rompt again: "
        ).strip().lower()
    if action == "y":
        return "accept", None
    if action == "p":

        return "reprompt", None

    fig, ax = plt.subplots(figsize=_figsize_for_image(image))
    ax.imshow(overlay)
    if action == "a":
        ax.set_title("Paint a brush stroke to add to the mask (drag along the structure). Scroll to zoom.")
    else:
        ax.set_title("Draw a closed loop around the region to remove from the mask. Scroll to zoom.")
    _wire_scroll_zoom(fig, ax, image)

    captured: dict = {}

    def on_select(vertices) -> None:
        if action == "a":
            captured["stroke"] = _rasterize_stroke(vertices, mask.shape, brush_width_px=brush_width_px)
        else:
            captured["stroke"] = _enclosed_region_mask(vertices, mask.shape)
        plt.close(fig)

    _selector = LassoSelector(ax, on_select, button=[1])
    plt.show()

    stroke = captured.get("stroke", np.zeros_like(mask, dtype=bool))
    return ("add" if action == "a" else "remove"), stroke

def default_metadata_prompt() -> dict:
    return {}

def _load_grayscale_image(path: Path) -> np.ndarray:
    return read_image_gray(path)

ImageLoader = Callable[[Path], np.ndarray]
PromptCapture = Callable[[np.ndarray], Prompt]
ReviewAndCorrect = Callable[[np.ndarray, np.ndarray], ReviewAction]
MetadataPrompt = Callable[[], dict]

def run_annotation_session(
    pilot_ids: Sequence[str],
    image_root: Path,
    records_path: Path,
    segmenter: Segmenter,
    output_dir: Path,
    donor_id_prefix: str = "D",
    prompt_capture: PromptCapture = default_prompt_capture,
    review_and_correct: ReviewAndCorrect = default_review_and_correct,
    metadata_prompt: MetadataPrompt = default_metadata_prompt,
    image_loader: ImageLoader = _load_grayscale_image,
    print_fn: Callable[[str], None] = print,
) -> None:
    records_path = Path(records_path)
    output_dir = Path(output_dir)

    library_root = records_path.resolve().parent
    try:
        output_dir.resolve().relative_to(library_root)
    except ValueError:
        raise ValueError(
            f"--output-dir {output_dir} is not inside the donor-library root {library_root} "
            f"(the directory containing --records-path) -- crop_path/mask_path can only be "
            f"stored as portable relative paths if donor crops/masks live under the same "
            f"directory as donor_records.csv"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    done = annotated_source_ids(records_path)
    remaining = [image_id for image_id in pilot_ids if image_id not in done]
    print_fn(f"{len(done)} already annotated, {len(remaining)} remaining in this pilot")

    donor_index = _existing_record_count(records_path)
    for i, image_id in enumerate(remaining):
        started_at = datetime.now(timezone.utc).isoformat()
        image = image_loader(Path(image_root) / image_id)
        prompt = prompt_capture(image)
        mask = annotate_donor(image, prompt, segmenter)

        while True:
            action, stroke = review_and_correct(image, mask)
            if action == "accept":
                break
            elif action == "add":
                mask = apply_correction(mask, add=stroke)
            elif action == "remove":
                mask = apply_correction(mask, remove=stroke)
            elif action == "reprompt":

                prompt = prompt_capture(image)
                mask = annotate_donor(image, prompt, segmenter)
            else:
                raise ValueError(f"unknown review action: {action!r}")

        try:
            patch, local_mask, (_, crop_offset_col) = crop_to_mask_bbox(image, mask)
        except ValueError:
            print_fn(f"{image_id}: mask is empty after correction, nothing saved -- will be re-served next run")
            continue

        donor_id = f"{donor_id_prefix}{donor_index:04d}"
        crop_path = output_dir / f"{donor_id}_crop.png"
        mask_path = output_dir / f"{donor_id}_mask.npy"
        PILImage.fromarray(patch).save(crop_path)
        np.save(mask_path, local_mask)

        metadata = metadata_prompt()
        record = {
            "donor_id": donor_id,
            "source_image_id": image_id,
            "crop_path": relativize_donor_path(crop_path, library_root),
            "mask_path": relativize_donor_path(mask_path, library_root),
            "source_image_width": int(image.shape[1]),
            "source_image_height": int(image.shape[0]),
            "annotation_started_at": started_at,

            **donor_mask_width_metrics(local_mask),

            "insertion_angle_deg": mask_principal_axis_angle_deg(local_mask),
            "laterality": donor_laterality_from_mask(
                local_mask, crop_offset_col=crop_offset_col, source_image_width=image.shape[1]
            ),
            **metadata,
        }
        while True:
            record["annotation_saved_at"] = datetime.now(timezone.utc).isoformat()
            try:
                save_donor_record(records_path, record)
                break
            except ValueError as exc:
                print_fn(f"invalid donor record, re-enter metadata: {exc}")
                record.update(metadata_prompt())

        donor_index += 1
        print_fn(f"saved {donor_id} for {image_id} ({i + 1}/{len(remaining)} this session)")

_NON_INTERACTIVE_BACKENDS = frozenset({"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"})

def _ensure_interactive_backend(backend: Optional[str] = None) -> None:
    if backend is None:
        import matplotlib

        backend = matplotlib.get_backend()
    if backend.lower() in _NON_INTERACTIVE_BACKENDS:
        raise RuntimeError(
            f"matplotlib backend {backend!r} is non-interactive -- this tool needs a "
            "live display for point/box/stroke capture. Run it in an environment with "
            "a display (not headless/SSH-without-X, not a CI runner), or set MPLBACKEND "
            "to an interactive backend (e.g. TkAgg, Qt5Agg) before launching."
        )

def _read_pilot_ids(pilot_file: Path) -> list[str]:
    pilot_file = Path(pilot_file)
    if pilot_file.suffix == ".csv":
        return pd.read_csv(pilot_file)["image_id"].astype(str).tolist()
    return [line.strip() for line in pilot_file.read_text().splitlines() if line.strip()]

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Interactive drain-donor annotation tool.")
    parser.add_argument(
        "--pilot-file",
        required=True,
        type=Path,
        help="donor_pilot_selection_manifest.csv (draw order) or a plain image_id-per-line file.",
    )
    parser.add_argument("--image-root", required=True, type=Path, help="Directory containing the pilot's source images.")
    parser.add_argument("--records-path", required=True, type=Path, help="Donor records CSV to append to (created if absent).")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write donor crop/mask files.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="MedSAM (or SAM) checkpoint path.")
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--brush-width-px",
        type=int,
        default=_STROKE_BRUSH_WIDTH_PX,
        help=(
            "Width, in image pixels, of the [a]dd correction brush ([r]emove stays "
            "loop-enclosure and is unaffected). Default is unvalidated against a real "
            "drain -- see _STROKE_BRUSH_WIDTH_PX's own comment -- tune this on the "
            "first few real donors rather than editing the source."
        ),
    )
    args = parser.parse_args(argv)

    _ensure_interactive_backend()

    pilot_ids = _read_pilot_ids(args.pilot_file)
    segmenter = MedSAMSegmenter(str(args.checkpoint), model_type=args.model_type, device=args.device)
    review_and_correct = functools.partial(default_review_and_correct, brush_width_px=args.brush_width_px)
    run_annotation_session(
        pilot_ids, args.image_root, args.records_path, segmenter, args.output_dir,
        review_and_correct=review_and_correct,
    )

if __name__ == "__main__":
    main()
