# Draws the tube, composites the four arms at native resolution and scores them; also holds the generator's ranges.
# Run: python3 -m cxr_shortcuts.runners.composite_and_score --help

from __future__ import annotations

import argparse
import dataclasses
import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from sklearn.metrics import roc_auc_score

from cxr_shortcuts.data.image_io import default_image_loader
from cxr_shortcuts.data.leakage import assert_no_leakage
from cxr_shortcuts.data.lung_masks import load_chexmask_lateral_masks, load_chexmask_lung_masks
from cxr_shortcuts.data.manifests import (
    DEV_SLICE_PATIENT_COUNT,
    DEV_SLICE_SEED,
    select_dev_slice_patient_ids,
    verify_manifest_file,
)
from cxr_shortcuts.edits.drain.donor_library import mask_principal_axis_angle_deg
from cxr_shortcuts.edits.drain.geometry import NoValidPlacementError, sample_valid_placement
from cxr_shortcuts.edits.drain.insertion import hard_paste_patch, insert_drain
from cxr_shortcuts.edits.drain.frequency_discriminator import (
    predict_scores_frequency,
    train_frequency_discriminator,
)
from cxr_shortcuts.edits.drain.pretrained_realism_discriminator import ResNet18Discriminator
from cxr_shortcuts.edits.drain.realism_discriminator import (
    DeepRealismDiscriminator,
    RealismDiscriminator,
    crop_around_local_mask,
    images_to_tensor,
    images_to_tensor_3ch,
    predict_scores,
    train_discriminator,
    train_model,
)
from cxr_shortcuts.edits.drain.synthetic_tube import generate_synthetic_tube_patch
from cxr_shortcuts.eval.bootstrap import cluster_bootstrap_auc_ci
from cxr_shortcuts.eval.detector_panel import panel_gate_ci, panel_point_estimate
from cxr_shortcuts.eval.schema import require_git_sha
from cxr_shortcuts.eval.realism_gate_schema import RealismGateResultRow, append_realism_gate_result, config_hash

PANEL_TORCH_DETECTORS = {
    "realism_shallow": (RealismDiscriminator, images_to_tensor),
    "realism_deep": (lambda: DeepRealismDiscriminator(dropout=0.3), images_to_tensor),
    "resnet18": (lambda: ResNet18Discriminator(pretrained=True), images_to_tensor_3ch),
}
PANEL_ALL_DETECTORS = tuple(PANEL_TORCH_DETECTORS) + ("frequency",)

CROP_SIZE_DEFAULT = 128
BLEND_MODE_DEFAULT = "log_additive"

_CHORD_ORIENTATION_JITTER_DEG = 15.0

_ACKNOWLEDGED_UNKNOWN_SURFACES = frozenset(
    {
        "classifier_patient_split",
        "donor_train_eval_donor_id_split",
        "donor_train_eval_patient_split",
        "donor_source_vs_audit_target_patient_overlap",
        "gate_subset_locked",
        "donor_eval_outside_training_split",
        "donor_pilot_flagged",
        "screener_patient_split",
        "screener_audit_target_overlap",
        "discriminator_donor_disjointness",
        "dfr_eval_disjointness",
        "jtt_train_only",
        "checkpoint_selection_validation_only",
        "operating_point_validation_only",
        "normalization_stats_train_only",
        "siim_nih_patient_mapping",
        "control_source_screener_confirmed_drain_free",

        "dev_slice_disjoint_from_eval_arms",
    }
)

@dataclasses.dataclass(frozen=True)
class TubeGenerationRanges:

    caliber_px: tuple[float, float] = (10.0, 27.0)
    length_px: tuple[float, float] = (150.0, 250.0)
    curvature: tuple[float, float] = (0.15, 0.35)

def _draw_tube_params(ranges: TubeGenerationRanges, rng: np.random.Generator) -> tuple[float, float, float]:
    caliber_px = float(rng.uniform(*ranges.caliber_px))
    length_px = float(rng.uniform(*ranges.length_px))
    curvature = float(rng.uniform(*ranges.curvature))
    return caliber_px, length_px, curvature

def _draw_chord_orientation_deg(
    left_lung_mask: np.ndarray,
    right_lung_mask: np.ndarray,
    rng: np.random.Generator,
) -> float:
    reference_mask = left_lung_mask if rng.random() < 0.5 else right_lung_mask
    other_mask = right_lung_mask if reference_mask is left_lung_mask else left_lung_mask
    if int(reference_mask.sum()) < 2:
        reference_mask = other_mask
    sign = 1.0 if rng.random() < 0.5 else -1.0
    jitter_deg = float(rng.uniform(-_CHORD_ORIENTATION_JITTER_DEG, _CHORD_ORIENTATION_JITTER_DEG))
    if int(reference_mask.sum()) < 2:
        return sign * 90.0 + jitter_deg
    magnitude_deg = mask_principal_axis_angle_deg(reference_mask)
    return sign * magnitude_deg + jitter_deg

@dataclasses.dataclass
class CompositedImage:
    image_id: str
    patient_id: str
    real_crop: np.ndarray
    production_crop: Optional[np.ndarray]
    crude_crop: Optional[np.ndarray]
    caliber_px: float
    length_px: float
    curvature: float
    chord_orientation_deg: float
    offset: tuple[int, int]
    real_full: Optional[np.ndarray] = None
    production_full: Optional[np.ndarray] = None
    crude_full: Optional[np.ndarray] = None

def _composite_one_target(
    image_id: str,
    patient_id: str,
    target_image: np.ndarray,
    lung_mask: np.ndarray,
    left_lung_mask: np.ndarray,
    right_lung_mask: np.ndarray,
    rng: np.random.Generator,
    tube_ranges: TubeGenerationRanges,
    blend_mode: str,
    crop_size: int,
    keep_full_resolution: bool,
) -> CompositedImage:
    background_level = float(target_image[lung_mask].mean())
    chord_orientation_deg = _draw_chord_orientation_deg(left_lung_mask, right_lung_mask, rng)
    caliber_px, length_px, curvature = _draw_tube_params(tube_ranges, rng)
    patch, mask, donor_background_level = generate_synthetic_tube_patch(
        length_px=length_px, caliber_px=caliber_px, rng=rng, curvature=curvature, background_level=background_level,
        chord_orientation_deg=chord_orientation_deg,
    )
    result = insert_drain(
        target_image, lung_mask, patch, mask, rng, blend_mode=blend_mode, donor_background_level=donor_background_level,
    )
    crude_full = hard_paste_patch(target_image, patch, result.local_mask, result.offset)

    crop_kwargs = dict(offset=result.offset, local_mask_shape=result.local_mask.shape, crop_size=crop_size)
    real_crop = crop_around_local_mask(target_image, **crop_kwargs)
    production_crop = crop_around_local_mask(result.image, **crop_kwargs)
    crude_crop = crop_around_local_mask(crude_full, **crop_kwargs)

    return CompositedImage(
        image_id=image_id,
        patient_id=patient_id,
        real_crop=real_crop,
        production_crop=production_crop,
        crude_crop=crude_crop,
        caliber_px=caliber_px,
        length_px=length_px,
        curvature=curvature,
        chord_orientation_deg=chord_orientation_deg,
        offset=result.offset,
        real_full=target_image if keep_full_resolution else None,
        production_full=result.image if keep_full_resolution else None,
        crude_full=crude_full if keep_full_resolution else None,
    )

def _sample_unpaired_reference_crop(
    image_id: str,
    patient_id: str,
    target_image: np.ndarray,
    lung_mask: np.ndarray,
    left_lung_mask: np.ndarray,
    right_lung_mask: np.ndarray,
    rng: np.random.Generator,
    tube_ranges: TubeGenerationRanges,
    crop_size: int,
) -> CompositedImage:
    chord_orientation_deg = _draw_chord_orientation_deg(left_lung_mask, right_lung_mask, rng)
    caliber_px, length_px, curvature = _draw_tube_params(tube_ranges, rng)
    _patch, mask, _bg = generate_synthetic_tube_patch(
        length_px=length_px, caliber_px=caliber_px, rng=rng, curvature=curvature, background_level=0.0,
        chord_orientation_deg=chord_orientation_deg,
    )
    offset = sample_valid_placement(mask, lung_mask, rng)
    real_crop = crop_around_local_mask(target_image, offset=offset, local_mask_shape=mask.shape, crop_size=crop_size)
    return CompositedImage(
        image_id=image_id,
        patient_id=patient_id,
        real_crop=real_crop,
        production_crop=None,
        crude_crop=None,
        caliber_px=caliber_px,
        length_px=length_px,
        curvature=curvature,
        chord_orientation_deg=chord_orientation_deg,
        offset=offset,
    )

def _train_arm_patient_ids(split_manifest: pd.DataFrame, train_arm: str) -> set:
    if train_arm == "primary":
        labels = {"train_primary"}
    elif train_arm == "extended":
        labels = {"train_primary", "train_extended_only"}
    else:
        raise ValueError(f"train_arm must be 'primary' or 'extended', got {train_arm!r}")
    return set(split_manifest.loc[split_manifest["discriminator_arm"].isin(labels), "patient_id"])

def _smoke_subsample(split_manifest: pd.DataFrame, arm_patient_ids: set, n: int, rng: np.random.Generator) -> set:
    counts = split_manifest[split_manifest["patient_id"].isin(arm_patient_ids)].groupby("patient_id").size()
    if n > len(counts):
        raise ValueError(f"cannot draw a smoke sample of {n} from only {len(counts)} available patients")
    multi = sorted(counts[counts > 1].index.tolist())
    if not multi:
        raise ValueError("no multi-image patient available in this arm -- cannot guarantee the clustering code path is exercised")
    guaranteed = multi[int(rng.integers(0, len(multi)))]
    remaining_pool = sorted(p for p in counts.index if p != guaranteed)
    rest = rng.choice(remaining_pool, size=n - 1, replace=False).tolist() if n > 1 else []
    return {guaranteed, *rest}

def _export_composite(composite: CompositedImage, export_dir: Path) -> None:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(composite.real_full).save(export_dir / f"{composite.image_id}_real.png")
    PILImage.fromarray(composite.production_full).save(export_dir / f"{composite.image_id}_production.png")
    PILImage.fromarray(composite.crude_full).save(export_dir / f"{composite.image_id}_crude.png")

def _write_scores_csv(
    path: Path,
    train_composites: Sequence[CompositedImage],
    eval_paired_composites: Sequence[CompositedImage],
    unpaired_composites: Sequence[CompositedImage],
    arm_by_patient_id: dict,
    main_real_score: np.ndarray,
    main_production_score: np.ndarray,
    main_crude_score: np.ndarray,
    pc_real_score: np.ndarray,
    pc_crude_score: np.ndarray,
    unpaired_main_real_score: np.ndarray,
) -> None:
    rows = []
    for c in train_composites:
        rows.append(
            {
                "image_id": c.image_id,
                "patient_id": c.patient_id,
                "discriminator_arm": arm_by_patient_id[c.patient_id],
                "composited": True,
                "caliber_px": c.caliber_px,
                "length_px": c.length_px,
                "curvature": c.curvature,
                "chord_orientation_deg": c.chord_orientation_deg,
                "offset_row": c.offset[0],
                "offset_col": c.offset[1],
                "main_real_score": np.nan,
                "main_production_score": np.nan,
                "main_crude_score": np.nan,
                "pc_real_score": np.nan,
                "pc_crude_score": np.nan,
            }
        )
    for i, c in enumerate(eval_paired_composites):
        rows.append(
            {
                "image_id": c.image_id,
                "patient_id": c.patient_id,
                "discriminator_arm": arm_by_patient_id[c.patient_id],
                "composited": True,
                "caliber_px": c.caliber_px,
                "length_px": c.length_px,
                "curvature": c.curvature,
                "chord_orientation_deg": c.chord_orientation_deg,
                "offset_row": c.offset[0],
                "offset_col": c.offset[1],
                "main_real_score": float(main_real_score[i]),
                "main_production_score": float(main_production_score[i]),
                "main_crude_score": float(main_crude_score[i]),
                "pc_real_score": float(pc_real_score[i]),
                "pc_crude_score": float(pc_crude_score[i]),
            }
        )
    for i, c in enumerate(unpaired_composites):
        rows.append(
            {
                "image_id": c.image_id,
                "patient_id": c.patient_id,
                "discriminator_arm": arm_by_patient_id[c.patient_id],
                "composited": False,
                "caliber_px": c.caliber_px,
                "length_px": c.length_px,
                "curvature": c.curvature,
                "chord_orientation_deg": c.chord_orientation_deg,
                "offset_row": c.offset[0],
                "offset_col": c.offset[1],
                "main_real_score": float(unpaired_main_real_score[i]),
                "main_production_score": np.nan,
                "main_crude_score": np.nan,
                "pc_real_score": np.nan,
                "pc_crude_score": np.nan,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)

def _auc_with_ci(cluster_ids: Sequence, real_scores: np.ndarray, edited_scores: np.ndarray, seed: int) -> tuple:
    labels = np.concatenate([np.zeros(len(real_scores)), np.ones(len(edited_scores))])
    scores = np.concatenate([real_scores, edited_scores])
    value = float(roc_auc_score(labels, scores))
    lo, hi = cluster_bootstrap_auc_ci(cluster_ids, real_scores, edited_scores, seed=seed)
    return value, lo, hi

def run_composite_and_score(
    split_manifest: pd.DataFrame,
    rca_df: pd.DataFrame,
    image_loader: Callable[[str], np.ndarray],
    train_arm: str,
    seed: int,
    output_dir,
    run_id: str,
    results_path=None,
    smoke: bool = False,
    smoke_train_n: int = 12,
    smoke_eval_paired_n: int = 0,
    smoke_eval_unpaired_n: int = 0,
    smoke_export_n: int = 5,
    smoke_export_dir=None,
    blend_mode: str = BLEND_MODE_DEFAULT,
    crop_size: int = CROP_SIZE_DEFAULT,
    tube_ranges: TubeGenerationRanges = TubeGenerationRanges(),
    dev_slice: bool = False,
    dev_slice_n: int = DEV_SLICE_PATIENT_COUNT,
    dev_slice_seed: int = DEV_SLICE_SEED,
    crop_cache_dir=None,
    device: str = "cpu",
    allow_sealed_arm_draw: bool = False,
    panel_detectors: tuple[str, ...] = (),
    panel_epochs: int = 60,
    panel_seeds: int = 0,
    panel_lr: float = 0.005,
    panel_batch_size: int = 8,
) -> dict:
    if smoke and results_path is not None:
        raise ValueError(
            "--results-path cannot be combined with --smoke -- a wiring run must never reach the real results schema"
        )
    if smoke and smoke_export_dir is None:
        raise ValueError("--smoke requires --smoke-export-dir -- a wiring run must export composites for visual inspection")
    if dev_slice and results_path is not None:
        raise ValueError(
            "--results-path cannot be combined with --dev-slice -- a dev-slice number must never reach the real results schema"
        )
    if dev_slice and smoke:
        raise ValueError("--dev-slice cannot be combined with --smoke -- two different, mutually exclusive run modes")
    if not smoke and not dev_slice and results_path is None:
        raise ValueError("a non-smoke, non-dev-slice run requires --results-path")
    unknown_panel_detectors = set(panel_detectors) - set(PANEL_ALL_DETECTORS)
    if unknown_panel_detectors:
        raise ValueError(f"unknown panel_detectors {sorted(unknown_panel_detectors)}, expected a subset of {PANEL_ALL_DETECTORS}")

    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_patient_ids = _train_arm_patient_ids(split_manifest, train_arm)
    eval_paired_patient_ids = set(
        split_manifest.loc[split_manifest["discriminator_arm"] == "eval_paired", "patient_id"]
    )
    eval_unpaired_patient_ids = set(
        split_manifest.loc[split_manifest["discriminator_arm"] == "eval_unpaired_reference", "patient_id"]
    )

    if smoke:

        if (smoke_eval_paired_n or smoke_eval_unpaired_n) and not allow_sealed_arm_draw:
            raise ValueError(
                "smoke mode would draw patients from eval_paired / eval_unpaired_reference, which are sealed against further "
                "exposure; use --dev-slice to exercise the evaluation path on train_extended_only instead"
            )
        if not (smoke_eval_paired_n or smoke_eval_unpaired_n):
            raise ValueError("smoke mode has no non-reserved evaluation population; use --dev-slice")
        train_patient_ids = _smoke_subsample(split_manifest, train_patient_ids, smoke_train_n, rng)
        eval_paired_patient_ids = _smoke_subsample(split_manifest, eval_paired_patient_ids, smoke_eval_paired_n, rng)
        eval_unpaired_patient_ids = _smoke_subsample(
            split_manifest, eval_unpaired_patient_ids, smoke_eval_unpaired_n, rng
        )

    dev_slice_patient_ids: Optional[set] = None
    if dev_slice:
        train_extended_only_ids = sorted(
            split_manifest.loc[split_manifest["discriminator_arm"] == "train_extended_only", "patient_id"].unique()
        )
        dev_slice_patient_ids = set(
            select_dev_slice_patient_ids(train_extended_only_ids, n=dev_slice_n, seed=dev_slice_seed)
        )

        train_patient_ids = train_patient_ids - dev_slice_patient_ids
        eval_paired_patient_ids = dev_slice_patient_ids
        eval_unpaired_patient_ids = set()

    from cxr_shortcuts.eval.population_log import log_population

    log_population(output_dir, "train_patients", train_patient_ids, "patient")
    log_population(output_dir, "eval_paired_patients", eval_paired_patient_ids, "patient")
    log_population(output_dir, "eval_unpaired_patients", eval_unpaired_patient_ids, "patient")
    if dev_slice_patient_ids is not None:
        log_population(output_dir, "dev_slice_patients", dev_slice_patient_ids, "patient")

    relevant_patient_ids = train_patient_ids | eval_paired_patient_ids | eval_unpaired_patient_ids

    run_manifest = split_manifest[split_manifest["patient_id"].isin(relevant_patient_ids)].copy()

    if dev_slice:
        real_eval_arm_patient_ids = set(
            split_manifest.loc[
                split_manifest["discriminator_arm"].isin(["eval_paired", "eval_unpaired_reference"]), "patient_id"
            ]
        )
        assert_no_leakage(
            discriminator_manifest=run_manifest,
            dev_slice_patient_ids=dev_slice_patient_ids,
            eval_arm_patient_ids=real_eval_arm_patient_ids,
            acknowledged_unknown=_ACKNOWLEDGED_UNKNOWN_SURFACES - {"dev_slice_disjoint_from_eval_arms"},
        )
    else:
        assert_no_leakage(discriminator_manifest=run_manifest, acknowledged_unknown=_ACKNOWLEDGED_UNKNOWN_SURFACES)

    arm_by_patient_id = run_manifest.drop_duplicates("patient_id").set_index("patient_id")["discriminator_arm"].to_dict()

    train_rows = run_manifest[run_manifest["patient_id"].isin(train_patient_ids)]
    eval_paired_rows = run_manifest[run_manifest["patient_id"].isin(eval_paired_patient_ids)]
    eval_unpaired_rows = run_manifest[run_manifest["patient_id"].isin(eval_unpaired_patient_ids)]

    needed_image_ids = pd.concat(
        [train_rows["image_id"], eval_paired_rows["image_id"], eval_unpaired_rows["image_id"]]
    ).unique()
    relevant_rca_rows = rca_df[rca_df["Image Index"].isin(needed_image_ids)]
    lung_masks = load_chexmask_lung_masks(relevant_rca_rows)
    lateral_masks = load_chexmask_lateral_masks(relevant_rca_rows)

    n_export_remaining = smoke_export_n if smoke else 0
    exported_image_ids: list = []
    n_dropped_no_placement = 0

    def _composite_rows(rows: pd.DataFrame, arm_name: str) -> list:
        nonlocal n_export_remaining, n_dropped_no_placement
        composites = []
        for _, row in rows.sort_values("image_id").iterrows():
            image_id, patient_id = row["image_id"], row["patient_id"]
            target_image = image_loader(image_id)
            keep_full = n_export_remaining > 0
            left_lung_mask, right_lung_mask = lateral_masks[image_id]
            try:
                composite = _composite_one_target(
                    image_id,
                    patient_id,
                    target_image,
                    lung_masks[image_id],
                    left_lung_mask,
                    right_lung_mask,
                    rng,
                    tube_ranges,
                    blend_mode,
                    crop_size,
                    keep_full,
                )
            except NoValidPlacementError:

                n_dropped_no_placement += 1
                continue
            if keep_full:
                _export_composite(composite, smoke_export_dir)
                exported_image_ids.append(image_id)
                n_export_remaining -= 1
            composites.append(composite)
        if not composites:
            raise RuntimeError(
                f"all {len(rows)} images in the {arm_name!r} arm were dropped for "
                f"NoValidPlacementError -- nothing left to composite"
            )
        return composites

    train_composites = _composite_rows(train_rows, "train")
    eval_paired_composites = _composite_rows(eval_paired_rows, "eval_paired")

    unpaired_composites = []
    for _, row in eval_unpaired_rows.sort_values("image_id").iterrows():
        image_id, patient_id = row["image_id"], row["patient_id"]
        target_image = image_loader(image_id)
        left_lung_mask, right_lung_mask = lateral_masks[image_id]
        try:
            unpaired_composites.append(
                _sample_unpaired_reference_crop(
                    image_id, patient_id, target_image, lung_masks[image_id],
                    left_lung_mask, right_lung_mask, rng, tube_ranges, crop_size
                )
            )
        except NoValidPlacementError:
            n_dropped_no_placement += 1
            continue
    if eval_unpaired_rows.shape[0] and not unpaired_composites:
        raise RuntimeError(
            f"all {eval_unpaired_rows.shape[0]} images in the 'eval_unpaired_reference' arm were "
            f"dropped for NoValidPlacementError -- nothing left to sample"
        )

    if crop_cache_dir is not None:
        crop_cache_dir = Path(crop_cache_dir)
        crop_cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            crop_cache_dir / "crops.npz",
            train_real=np.stack([c.real_crop for c in train_composites]),
            train_production=np.stack([c.production_crop for c in train_composites]),
            train_crude=np.stack([c.crude_crop for c in train_composites]),
            train_patient_ids=np.array([c.patient_id for c in train_composites]),
            train_image_ids=np.array([c.image_id for c in train_composites]),
            eval_real=np.stack([c.real_crop for c in eval_paired_composites]),
            eval_production=np.stack([c.production_crop for c in eval_paired_composites]),
            eval_crude=np.stack([c.crude_crop for c in eval_paired_composites]),
            eval_patient_ids=np.array([c.patient_id for c in eval_paired_composites]),
            eval_image_ids=np.array([c.image_id for c in eval_paired_composites]),
        )

    train_real = [c.real_crop for c in train_composites]
    train_production = [c.production_crop for c in train_composites]
    train_crude = [c.crude_crop for c in train_composites]
    main_model = train_discriminator(train_real, train_production, seed=seed, device=device)
    positive_control_model = train_discriminator(train_real, train_crude, seed=seed, device=device)

    eval_real_crops = [c.real_crop for c in eval_paired_composites]
    eval_production_crops = [c.production_crop for c in eval_paired_composites]
    eval_crude_crops = [c.crude_crop for c in eval_paired_composites]
    eval_patient_ids_arr = [c.patient_id for c in eval_paired_composites]

    main_real_score = predict_scores(main_model, eval_real_crops)
    main_production_score = predict_scores(main_model, eval_production_crops)

    main_crude_score = predict_scores(main_model, eval_crude_crops)
    pc_real_score = predict_scores(positive_control_model, eval_real_crops)
    pc_crude_score = predict_scores(positive_control_model, eval_crude_crops)

    unpaired_real_crops = [c.real_crop for c in unpaired_composites]
    unpaired_main_real_score = (
        predict_scores(main_model, unpaired_real_crops) if unpaired_real_crops else np.array([])
    )

    _write_scores_csv(
        output_dir / "scores.csv",
        train_composites,
        eval_paired_composites,
        unpaired_composites,
        arm_by_patient_id,
        main_real_score,
        main_production_score,
        main_crude_score,
        pc_real_score,
        pc_crude_score,
        unpaired_main_real_score,
    )

    main_production_auc, main_production_lo, main_production_hi = _auc_with_ci(
        eval_patient_ids_arr, main_real_score, main_production_score, seed=seed
    )
    main_crude_auc, main_crude_lo, main_crude_hi = _auc_with_ci(
        eval_patient_ids_arr, main_real_score, main_crude_score, seed=seed + 1
    )
    pc_crude_auc, pc_crude_lo, pc_crude_hi = _auc_with_ci(
        eval_patient_ids_arr, pc_real_score, pc_crude_score, seed=seed + 2
    )

    unpaired_auc = None
    if unpaired_composites:
        unpaired_labels = np.concatenate([np.zeros(len(unpaired_main_real_score)), np.ones(len(main_production_score))])
        unpaired_scores = np.concatenate([unpaired_main_real_score, main_production_score])
        unpaired_auc = float(roc_auc_score(unpaired_labels, unpaired_scores))

    panel_statistic = None
    panel_lo = panel_hi = None
    panel_per_detector: dict = {}
    if panel_detectors and panel_seeds > 0:
        per_detector_seed_scores: dict = {}
        for name in panel_detectors:
            seed_scores = []
            for s in range(panel_seeds):
                if name == "frequency":
                    freq_model = train_frequency_discriminator(train_real, train_production, seed=s)
                    r = predict_scores_frequency(freq_model, eval_real_crops)
                    p = predict_scores_frequency(freq_model, eval_production_crops)
                else:
                    factory, to_tensor_fn = PANEL_TORCH_DETECTORS[name]
                    torch_model = train_model(
                        factory, train_real, train_production, seed=s, epochs=panel_epochs, lr=panel_lr,
                        to_tensor_fn=to_tensor_fn, device=device, batch_size=panel_batch_size,
                    )
                    r = predict_scores(torch_model, eval_real_crops, to_tensor_fn=to_tensor_fn)
                    p = predict_scores(torch_model, eval_production_crops, to_tensor_fn=to_tensor_fn)
                seed_scores.append((r, p))
            per_detector_seed_scores[name] = seed_scores
        panel_statistic, panel_per_detector = panel_point_estimate(per_detector_seed_scores)
        panel_lo, panel_hi = panel_gate_ci(eval_patient_ids_arr, per_detector_seed_scores, seed=seed)

    if results_path is not None:
        resolved_config = {
            "train_arm": train_arm,
            "blend_mode": blend_mode,
            "crop_size": crop_size,
            "seed": seed,
            "caliber_px_range": tube_ranges.caliber_px,
            "length_px_range": tube_ranges.length_px,
            "curvature_range": tube_ranges.curvature,
        }
        common = dict(
            run_id=run_id,
            seed=seed,
            train_arm=train_arm,
            blend_mode=blend_mode,
            crop_size=crop_size,
            config_hash=config_hash(resolved_config),
            git_sha=require_git_sha(),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        n_eval_paired_patients = len(eval_paired_patient_ids)
        n_eval_paired_images = len(eval_paired_composites)
        append_realism_gate_result(
            RealismGateResultRow(
                discriminator="main", comparison="paired_production", eval_arm="eval_paired",
                n_patients=n_eval_paired_patients, n_images=n_eval_paired_images,
                value=main_production_auc, ci_low=main_production_lo, ci_high=main_production_hi, **common,
            ),
            results_path,
        )
        append_realism_gate_result(
            RealismGateResultRow(
                discriminator="main", comparison="paired_crude", eval_arm="eval_paired",
                n_patients=n_eval_paired_patients, n_images=n_eval_paired_images,
                value=main_crude_auc, ci_low=main_crude_lo, ci_high=main_crude_hi, **common,
            ),
            results_path,
        )
        append_realism_gate_result(
            RealismGateResultRow(
                discriminator="positive_control", comparison="paired_crude", eval_arm="eval_paired",
                n_patients=n_eval_paired_patients, n_images=n_eval_paired_images,
                value=pc_crude_auc, ci_low=pc_crude_lo, ci_high=pc_crude_hi, **common,
            ),
            results_path,
        )
        if unpaired_auc is not None:
            append_realism_gate_result(
                RealismGateResultRow(
                    discriminator="main", comparison="unpaired_production", eval_arm="eval_unpaired_reference",
                    n_patients=len(eval_unpaired_patient_ids), n_images=len(unpaired_composites),
                    value=unpaired_auc, ci_low=None, ci_high=None, **common,
                ),
                results_path,
            )
        for name, mean_auc in panel_per_detector.items():
            append_realism_gate_result(
                RealismGateResultRow(
                    discriminator=f"panel_{name}", comparison="paired_production", eval_arm="eval_paired",
                    n_patients=n_eval_paired_patients, n_images=n_eval_paired_images,
                    value=mean_auc, ci_low=None, ci_high=None, **common,
                ),
                results_path,
            )
        if panel_statistic is not None:
            append_realism_gate_result(
                RealismGateResultRow(
                    discriminator="panel_statistic", comparison="paired_production", eval_arm="eval_paired",
                    n_patients=n_eval_paired_patients, n_images=n_eval_paired_images,
                    value=panel_statistic, ci_low=panel_lo, ci_high=panel_hi, **common,
                ),
                results_path,
            )

    return {
        "run_id": run_id,
        "smoke": smoke,
        "dev_slice": dev_slice,
        "train_arm": train_arm,
        "n_train_patients": len(train_patient_ids),
        "n_train_images": len(train_composites),
        "n_eval_paired_patients": len(eval_paired_patient_ids),
        "n_eval_paired_images": len(eval_paired_composites),
        "n_eval_unpaired_patients": len(eval_unpaired_patient_ids),
        "n_eval_unpaired_images": len(unpaired_composites),
        "main_paired_production_auc": main_production_auc,
        "main_paired_production_ci": (main_production_lo, main_production_hi),
        "main_paired_crude_auc": main_crude_auc,
        "main_paired_crude_ci": (main_crude_lo, main_crude_hi),
        "positive_control_paired_crude_auc": pc_crude_auc,
        "positive_control_paired_crude_ci": (pc_crude_lo, pc_crude_hi),
        "unpaired_production_auc": unpaired_auc,
        "panel_statistic": panel_statistic,
        "panel_ci": (panel_lo, panel_hi),
        "panel_per_detector": panel_per_detector,
        "n_dropped_no_placement": n_dropped_no_placement,
        "exported_image_ids": exported_image_ids,
        "scores_csv_path": str(output_dir / "scores.csv"),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Composite-and-score runner for the realism-discriminator Stage 0 gate")
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--rca-csv", required=True, type=Path)
    parser.add_argument("--nih-root", required=True, type=Path)
    parser.add_argument("--train-arm", required=True, choices=["primary", "extended"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--results-path", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="wiring-check mode: small real subsample, never writes --results-path, exports composites",
    )
    parser.add_argument("--smoke-train-n", type=int, default=12)
    parser.add_argument("--smoke-eval-paired-n", type=int, default=0)
    parser.add_argument("--smoke-eval-unpaired-n", type=int, default=0)
    parser.add_argument("--smoke-export-n", type=int, default=5)
    parser.add_argument("--smoke-export-dir", type=Path, default=None)
    parser.add_argument("--blend-mode", default=BLEND_MODE_DEFAULT, choices=["poisson", "log_additive"])
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE_DEFAULT)
    parser.add_argument(
        "--dev-slice",
        action="store_true",
        help="composite the rendering-iteration dev slice (train_extended_only patients) in place of eval_paired; never writes --results-path",
    )
    parser.add_argument("--dev-slice-n", type=int, default=DEV_SLICE_PATIENT_COUNT)
    parser.add_argument("--dev-slice-seed", type=int, default=DEV_SLICE_SEED)
    parser.add_argument(
        "--crop-cache-dir",
        type=Path,
        default=None,
        help="save every composited crop (train + eval, real/production/crude) to <dir>/crops.npz",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="'cpu' or 'cuda', for the two train_discriminator() calls -- a pure execution-environment "
        "choice, defaults to 'cpu'.",
    )
    parser.add_argument(
        "--panel-detectors", default="",
        help="comma-separated subset of "
        f"{PANEL_ALL_DETECTORS} to also train/report on the paired_production task -- opt-in, "
        "default empty (panel wiring off, existing main/positive_control rows unaffected).",
    )
    parser.add_argument("--panel-epochs", type=int, default=60, help="epochs for the three torch panel detectors (frequency has none to sweep)")
    parser.add_argument("--panel-seeds", type=int, default=0, help="seeds per panel detector; 0 means panel wiring is off regardless of --panel-detectors")
    parser.add_argument("--panel-lr", type=float, default=0.005)
    parser.add_argument("--panel-batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.smoke and args.results_path is not None:
        parser.error("--results-path cannot be combined with --smoke -- a wiring run must never reach the real results schema")
    if args.smoke and args.smoke_export_dir is None:
        parser.error("--smoke requires --smoke-export-dir")
    if args.dev_slice and args.results_path is not None:
        parser.error("--results-path cannot be combined with --dev-slice -- a dev-slice number must never reach the real results schema")
    if args.dev_slice and args.smoke:
        parser.error("--dev-slice cannot be combined with --smoke")
    if not args.smoke and not args.dev_slice and args.results_path is None:
        parser.error("a non-smoke, non-dev-slice run requires --results-path")

    verify_manifest_file(args.split_manifest)
    split_manifest = pd.read_csv(args.split_manifest)
    rca_df = pd.read_csv(args.rca_csv)
    image_loader = default_image_loader(args.nih_root)

    summary = run_composite_and_score(
        split_manifest,
        rca_df,
        image_loader,
        args.train_arm,
        args.seed,
        args.output_dir,
        args.run_id,
        results_path=args.results_path,
        smoke=args.smoke,
        smoke_train_n=args.smoke_train_n,
        smoke_eval_paired_n=args.smoke_eval_paired_n,
        smoke_eval_unpaired_n=args.smoke_eval_unpaired_n,
        smoke_export_n=args.smoke_export_n,
        smoke_export_dir=args.smoke_export_dir,
        blend_mode=args.blend_mode,
        crop_size=args.crop_size,
        dev_slice=args.dev_slice,
        dev_slice_n=args.dev_slice_n,
        dev_slice_seed=args.dev_slice_seed,
        crop_cache_dir=args.crop_cache_dir,
        device=args.device,
        panel_detectors=tuple(d for d in args.panel_detectors.split(",") if d),
        panel_epochs=args.panel_epochs,
        panel_seeds=args.panel_seeds,
        panel_lr=args.panel_lr,
        panel_batch_size=args.panel_batch_size,
    )
    print(summary)

if __name__ == "__main__":
    main()
