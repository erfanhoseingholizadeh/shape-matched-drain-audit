# Loaders for the NIH ChestX-ray14 metadata and the NEATX drain labels.
# Run: library module, imported by the scripts

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from cxr_shortcuts.data.image_io import read_image_gray
from cxr_shortcuts.data.reserved_guard import assert_not_reserved
from cxr_shortcuts.data.transforms import to_model_tensor

def load_nih_chestxray14_metadata(metadata_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(metadata_csv)
    return pd.DataFrame(
        {
            "image_id": raw["Image Index"],
            "patient_id": raw["Patient ID"].astype(str),
            "view_position": raw["View Position"],
            "pneumothorax": raw["Finding Labels"].str.contains("Pneumothorax", na=False),
        }
    )

NIH_14_LABELS: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Pneumothorax",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Hernia",
    "Infiltration",
)
PNEUMOTHORAX_LABEL_INDEX = NIH_14_LABELS.index("Pneumothorax")

def load_nih_chestxray14_multilabel_metadata(metadata_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(metadata_csv)
    result = pd.DataFrame({"image_id": raw["Image Index"]})
    for label in NIH_14_LABELS:
        result[label] = raw["Finding Labels"].str.contains(label, na=False)
    return result

def attach_multilabel_targets(manifest: pd.DataFrame, multilabel_df: pd.DataFrame) -> pd.DataFrame:
    return manifest.merge(multilabel_df, on="image_id", how="left")

def load_neatx_drain_labels(neatx_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(neatx_csv)
    return pd.DataFrame(
        {
            "image_id": raw["Image Index"],
            "drain_present": raw["Drain"].astype(bool),
        }
    )

def load_neatx_drain_locations(neatx_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(neatx_csv)
    return pd.DataFrame(
        {
            "image_id": raw["Image Index"],
            "Drain_Location_a1": raw["Drain_Location_a1"],
            "Drain_Location_a2": raw["Drain_Location_a2"],
        }
    )

def select_neatx_drain_positive_image_ids(neatx_labels: pd.DataFrame) -> list[str]:
    return neatx_labels.loc[neatx_labels["drain_present"], "image_id"].tolist()

def attach_drain_labels(records: pd.DataFrame, neatx_labels: pd.DataFrame) -> pd.DataFrame:
    return records.merge(neatx_labels[["image_id", "drain_present"]], on="image_id", how="left")

class NIHPneumothoraxDataset(Dataset):

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_root: Path,
        split: str,
        image_size: int = 224,
        label_columns: Sequence[str] = ("pneumothorax",),
        image_path_column: str | None = None,
    ):
        self.image_root = Path(image_root)
        self.image_size = image_size
        self.label_columns = list(label_columns)
        self.image_path_column = image_path_column
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[idx]
        assert_not_reserved(row["image_id"], "NIHPneumothoraxDataset.__getitem__")
        image_path = Path(row[self.image_path_column]) if self.image_path_column else self.image_root / row["image_id"]
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")

        image = read_image_gray(image_path, image_id=row["image_id"])
        image_tensor = to_model_tensor(image, size=self.image_size)
        if len(self.label_columns) == 1:
            label = torch.tensor(float(row[self.label_columns[0]]))
        else:
            label = torch.tensor([float(row[col]) for col in self.label_columns])
        return image_tensor, label
