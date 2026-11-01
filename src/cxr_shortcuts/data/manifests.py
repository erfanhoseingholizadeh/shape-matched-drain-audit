# Patient-level splits, the classification manifest and its hash sidecar.
# Run: library module, imported by the scripts

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from cxr_shortcuts.data.lung_masks import DEFAULT_RCA_COLUMN, rca_filter_report

Ratios = tuple[float, float, float]

def assign_patient_splits(
    patient_ids: Sequence[str],
    ratios: Ratios = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, str]:
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    unique_ids = sorted(set(patient_ids))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)

    n = len(shuffled)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))

    assignment: dict[str, str] = {}
    for pid in shuffled[:n_train]:
        assignment[pid] = "train"
    for pid in shuffled[n_train : n_train + n_val]:
        assignment[pid] = "val"
    for pid in shuffled[n_train + n_val :]:
        assignment[pid] = "test"
    return assignment

def build_manifest(
    records: pd.DataFrame,
    patient_id_col: str = "patient_id",
    ratios: Ratios = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> pd.DataFrame:
    assignment = assign_patient_splits(records[patient_id_col].unique(), ratios=ratios, seed=seed)
    manifest = records.copy()
    manifest["split"] = manifest[patient_id_col].map(assignment)
    return manifest

def assign_gate_subset(
    patient_ids: Sequence[str],
    fraction: float = 0.5,
    seed: int = 0,
) -> dict[str, str]:
    unique_ids = sorted(set(patient_ids))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)

    n_gate = int(round(len(shuffled) * fraction))
    assignment: dict[str, str] = {}
    for pid in shuffled[:n_gate]:
        assignment[pid] = "gate"
    for pid in shuffled[n_gate:]:
        assignment[pid] = "dev"
    return assignment

def attach_gate_subset(
    manifest: pd.DataFrame,
    split_col: str = "split",
    target_split: str = "test",
    patient_id_col: str = "patient_id",
    fraction: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    target_patients = manifest.loc[manifest[split_col] == target_split, patient_id_col].unique()
    assignment = assign_gate_subset(target_patients, fraction=fraction, seed=seed)
    result = manifest.copy()
    result["gate_subset"] = result[patient_id_col].map(assignment).fillna("")
    return result

def _canonical_id_order(patient_ids: Iterable, numeric: bool) -> list:
    unique = list(set(patient_ids))
    keys = [str(p) for p in unique]
    if len(set(keys)) != len(keys):
        raise ValueError("patient ids that collide when compared as strings (e.g. 1 and '1') cannot be ordered canonically")
    if numeric and all(k.lstrip("-").isdigit() for k in keys):
        return sorted(unique, key=lambda p: int(str(p)))
    return sorted(unique, key=str)

VAL_SUBSET_SEED = 2000
VAL_SUBSET_FRACTION = 0.5

def assign_val_subset(
    patient_ids: Sequence[str],
    fraction: float = VAL_SUBSET_FRACTION,
    seed: int = VAL_SUBSET_SEED,
) -> dict[str, str]:
    unique_ids = _canonical_id_order(patient_ids, numeric=False)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique_ids))

    n_select = int(round(len(unique_ids) * fraction))
    assignment: dict = {}
    for rank, i in enumerate(order):
        assignment[unique_ids[i]] = "select" if rank < n_select else "report"
    return assignment

def attach_val_subset(
    manifest: pd.DataFrame,
    split_col: str = "split",
    target_split: str = "val",
    patient_id_col: str = "patient_id",
    fraction: float = VAL_SUBSET_FRACTION,
    seed: int = VAL_SUBSET_SEED,
) -> pd.DataFrame:
    target_patients = manifest.loc[manifest[split_col] == target_split, patient_id_col].unique()
    assignment = {str(k): v for k, v in assign_val_subset(target_patients, fraction=fraction, seed=seed).items()}
    result = manifest.copy()
    result["val_subset"] = result[patient_id_col].astype(str).map(assignment).fillna("")
    return result

DISCRIMINATOR_ARM_COUNTS: dict[str, int] = {
    "train_extended": 1346,
    "eval_paired": 500,
    "eval_unpaired_reference": 450,
}
DISCRIMINATOR_TRAIN_PRIMARY_COUNT = 700

_IMAGES_PER_PATIENT_BIN_ORDER = ["1", "2", "3-4", "5-9", "10+"]

def _images_per_patient_bin(n_images: int) -> str:
    if n_images == 1:
        return "1"
    if n_images == 2:
        return "2"
    if n_images <= 4:
        return "3-4"
    if n_images <= 9:
        return "5-9"
    return "10+"

def _largest_remainder_allocate(total: int, fractions: Mapping[str, float]) -> dict[str, int]:
    ideal = {key: total * frac for key, frac in fractions.items()}
    floors = {key: int(np.floor(v)) for key, v in ideal.items()}
    remainder = total - sum(floors.values())
    order = sorted(ideal.keys(), key=lambda key: ideal[key] - floors[key], reverse=True)
    for key in order[:remainder]:
        floors[key] += 1
    return floors

def assign_discriminator_arms(
    patient_ids: Sequence[str],
    images_per_patient: Mapping[str, int],
    arm_counts: Mapping[str, int] = DISCRIMINATOR_ARM_COUNTS,
    primary_count: int = DISCRIMINATOR_TRAIN_PRIMARY_COUNT,
    seed: int = 0,
) -> tuple[dict[str, str], frozenset[str]]:
    unique_ids = sorted(set(patient_ids))
    total = len(unique_ids)
    if sum(arm_counts.values()) != total:
        raise ValueError(
            f"arm_counts sums to {sum(arm_counts.values())}, expected exactly {total} "
            f"(the number of unique patient_ids given) -- allocation must partition the pool exactly"
        )
    arm_order = list(arm_counts.keys())

    bins: dict[str, list[str]] = {label: [] for label in _IMAGES_PER_PATIENT_BIN_ORDER}
    for pid in unique_ids:
        bins[_images_per_patient_bin(images_per_patient[pid])].append(pid)

    rng = np.random.default_rng(seed)
    shuffled_bins = {
        label: list(rng.permutation(pids)) if pids else []
        for label, pids in bins.items()
    }

    arm_fracs = {arm: count / total for arm, count in arm_counts.items()}
    per_bin_counts: dict[str, dict[str, int]] = {}
    per_bin_remainders: dict[str, dict[str, float]] = {}
    for label in _IMAGES_PER_PATIENT_BIN_ORDER:
        bin_size = len(shuffled_bins[label])
        if bin_size == 0:
            per_bin_counts[label] = {arm: 0 for arm in arm_order}
            per_bin_remainders[label] = {arm: 0.0 for arm in arm_order}
            continue
        ideal = {arm: bin_size * arm_fracs[arm] for arm in arm_order}
        counts = _largest_remainder_allocate(bin_size, arm_fracs)
        per_bin_counts[label] = counts
        per_bin_remainders[label] = {arm: ideal[arm] - int(np.floor(ideal[arm])) for arm in arm_order}

    def _global_totals() -> dict[str, int]:
        totals = {arm: 0 for arm in arm_order}
        for label in _IMAGES_PER_PATIENT_BIN_ORDER:
            for arm in arm_order:
                totals[arm] += per_bin_counts[label][arm]
        return totals

    drift = {arm: arm_counts[arm] - count for arm, count in _global_totals().items()}
    while any(d != 0 for d in drift.values()):
        arm_deficit = max(drift, key=lambda a: drift[a])
        arm_excess = min(drift, key=lambda a: drift[a])
        assert drift[arm_deficit] > 0 and drift[arm_excess] < 0
        best_label, best_score = None, None
        for label in _IMAGES_PER_PATIENT_BIN_ORDER:
            if per_bin_counts[label][arm_excess] <= 0:
                continue
            score = per_bin_remainders[label][arm_deficit] - per_bin_remainders[label][arm_excess]
            if best_score is None or score > best_score:
                best_label, best_score = label, score
        if best_label is None:
            raise AssertionError(
                "controlled-rounding correction stuck -- no bin has a movable patient "
                f"from {arm_excess!r} to satisfy {arm_deficit!r}'s deficit; this indicates "
                "an arm_counts/pool-size mismatch that should have been caught above"
            )
        per_bin_counts[best_label][arm_excess] -= 1
        per_bin_counts[best_label][arm_deficit] += 1
        drift[arm_excess] += 1
        drift[arm_deficit] -= 1

    arm_assignment: dict[str, str] = {}
    extended_bin_patients: dict[str, list[str]] = {}
    for label in _IMAGES_PER_PATIENT_BIN_ORDER:
        pids = shuffled_bins[label]
        cursor = 0
        for arm in arm_order:
            n = per_bin_counts[label][arm]
            for pid in pids[cursor : cursor + n]:
                arm_assignment[pid] = arm
            if arm == "train_extended":
                extended_bin_patients[label] = pids[cursor : cursor + n]
            cursor += n

    primary_frac = primary_count / arm_counts["train_extended"]
    bin_extended_sizes = {label: len(pids) for label, pids in extended_bin_patients.items()}
    primary_counts_per_bin = _largest_remainder_allocate(
        primary_count,
        {label: size / arm_counts["train_extended"] for label, size in bin_extended_sizes.items()},
    ) if arm_counts["train_extended"] > 0 else {label: 0 for label in extended_bin_patients}

    primary_patient_ids: set[str] = set()
    for label, pids in extended_bin_patients.items():
        n_primary = primary_counts_per_bin.get(label, 0)
        primary_patient_ids.update(pids[:n_primary])

    return arm_assignment, frozenset(primary_patient_ids)

DEV_SLICE_PATIENT_COUNT = 60
DEV_SLICE_SEED = 1000

def select_dev_slice_patient_ids(
    train_extended_only_patient_ids: Sequence[str],
    n: int = DEV_SLICE_PATIENT_COUNT,
    seed: int = DEV_SLICE_SEED,
) -> frozenset[str]:
    unique_ids = _canonical_id_order(train_extended_only_patient_ids, numeric=True)
    if n > len(unique_ids):
        raise ValueError(
            f"cannot draw a {n}-patient dev slice from only {len(unique_ids)} "
            f"train_extended_only patients"
        )
    rng = np.random.default_rng(seed)
    return frozenset(unique_ids[i] for i in rng.choice(len(unique_ids), size=n, replace=False))

def discriminator_arm_stratum_balance(
    arm_assignment: Mapping[str, str],
    values: Mapping[str, float],
    n_bins: int = 4,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "patient_id": list(arm_assignment.keys()),
            "arm": list(arm_assignment.values()),
            "value": [values[pid] for pid in arm_assignment.keys()],
        }
    )
    df["value_bin"] = pd.qcut(df["value"], q=n_bins, duplicates="drop")
    return (
        df.groupby(["value_bin", "arm"], observed=True)["value"]
        .agg(["count", "mean"])
        .reset_index()
    )

def assert_discriminator_arms_disjoint_and_nested(
    arm_assignment: Mapping[str, str],
    primary_patient_ids: Sequence[str],
) -> None:
    known_arms = {"train_extended", "eval_paired", "eval_unpaired_reference"}
    unexpected_arms = set(arm_assignment.values()) - known_arms
    if unexpected_arms:
        raise AssertionError(f"unexpected discriminator arm label(s): {sorted(unexpected_arms)}")

    extended_patients = {pid for pid, arm in arm_assignment.items() if arm == "train_extended"}
    primary_set = set(primary_patient_ids)
    not_extended = primary_set - extended_patients
    if not_extended:
        raise AssertionError(
            f"primary_patient_ids contains {len(not_extended)} patient(s) not in the "
            f"'train_extended' arm: {sorted(not_extended)[:10]} -- extended must be a "
            "superset of primary"
        )

def attach_discriminator_arm(
    manifest: pd.DataFrame,
    arm_assignment: Mapping[str, str],
    primary_patient_ids: Iterable[str],
    patient_id_col: str = "patient_id",
) -> pd.DataFrame:
    primary_set = set(primary_patient_ids)

    def _label(patient_id: str) -> str:
        arm = arm_assignment.get(patient_id)
        if arm is None:
            return ""
        if arm == "train_extended":
            return "train_primary" if patient_id in primary_set else "train_extended_only"
        return arm

    result = manifest.copy()
    result["discriminator_arm"] = result[patient_id_col].map(_label)
    return result

def assert_no_discriminator_arm_leakage(
    manifest: pd.DataFrame,
    patient_id_col: str = "patient_id",
    arm_col: str = "discriminator_arm",
) -> None:
    in_pool = manifest[manifest[arm_col] != ""]
    counts = in_pool.groupby(patient_id_col)[arm_col].nunique()
    leaking = counts[counts > 1]
    if not leaking.empty:
        raise AssertionError(f"discriminator-arm leakage across patients: {sorted(leaking.index)}")

def attach_donor_eligibility(
    manifest: pd.DataFrame,
    neatx_labels: pd.DataFrame,
    image_id_col: str = "image_id",
) -> pd.DataFrame:
    merged = manifest.merge(neatx_labels[[image_id_col, "drain_present"]], on=image_id_col, how="left")
    merged["donor_eligible"] = merged["drain_present"].fillna(False).astype(bool)
    return merged.drop(columns=["drain_present"])

def assert_no_patient_leakage(
    manifest: pd.DataFrame,
    patient_id_col: str = "patient_id",
    split_col: str = "split",
) -> None:
    counts = manifest.groupby(patient_id_col)[split_col].nunique()
    leaking = counts[counts > 1]
    if not leaking.empty:
        raise AssertionError(f"patient leakage across splits: {sorted(leaking.index)}")

def manifest_hash(manifest: pd.DataFrame) -> str:
    canonical = manifest.sort_values(list(manifest.columns)).to_csv(index=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def save_manifest(manifest: pd.DataFrame, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = manifest.sort_values(list(manifest.columns))
    canonical.to_csv(path, index=False)
    digest = manifest_hash(manifest)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
    return digest

def assert_sufficient_pool(
    eligible_count: int,
    required_count: int,
    headroom_factor: float = 1.5,
    context: str = "eligible pool",
) -> None:
    minimum = math.ceil(required_count * headroom_factor)
    if eligible_count < minimum:
        raise ValueError(
            f"{context}: {eligible_count} eligible, need >= {minimum} "
            f"({required_count} required x {headroom_factor} headroom for "
            "drain-screen attrition) -- resolve before freezing the split manifest."
        )

def select_rca_eligible_negatives(
    manifest: pd.DataFrame,
    rca_df: pd.DataFrame,
    min_rca: float,
    target_split: str,
    required_count: int,
    headroom_factor: float = 1.5,
    pneumothorax_col: str = "pneumothorax",
    split_col: str = "split",
    image_id_col: str = "image_id",
    rca_image_id_col: str = "Image Index",
    rca_column: str = DEFAULT_RCA_COLUMN,
) -> tuple[pd.DataFrame, dict]:
    candidates = manifest[(~manifest[pneumothorax_col].astype(bool)) & (manifest[split_col] == target_split)]
    report = rca_filter_report(
        candidates[image_id_col], rca_df, min_rca, image_id_col=rca_image_id_col, column=rca_column
    )
    assert_sufficient_pool(
        report["eligible"],
        required_count,
        headroom_factor=headroom_factor,
        context=f"split={target_split!r} pneumothorax-negative pool after RCA>={min_rca} filtering",
    )
    eligible = candidates[candidates[image_id_col].isin(report["eligible_ids"])].copy()
    return eligible, report

def verify_manifest_file(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256(path.read_text().encode("utf-8")).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected = sidecar.read_text().strip()
    if digest != expected:
        raise AssertionError(f"manifest {path} does not match its committed hash in {sidecar}")
    return digest
