# Builds the reserved-id list used for the confirmatory run (the arms-only list for the held-out composite).
# Run: library module, imported by the scripts

import hashlib
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "src"))
from cxr_shortcuts.data.gate_guard import load_manifest, load_table

M = R / "data/processed/manifests"
nih = load_manifest(M / "nih_pneumothorax.csv", dtype={"patient_id": str})
disc = load_table(M / "discriminator_split.csv", id_col="patient_id", id_kind="patient", manifest_path=M / "nih_pneumothorax.csv", dtype={"patient_id": str})
pats = set(disc.loc[disc.discriminator_arm.isin(("eval_paired", "eval_unpaired_reference")), "patient_id"].astype(str))
ids = sorted(set(nih.loc[nih.patient_id.astype(str).isin(pats), "image_id"]) | set(disc.loc[disc.patient_id.astype(str).isin(pats), "image_id"]))
out = R / "review/heldout/confirmatory/reserved_image_ids_arms_only.txt"
out.parent.mkdir(parents=True, exist_ok=True)
text = "\n".join(ids) + "\n"
out.write_text(text)
print(f"{len(ids)} reserved arm image ids, {len(pats)} patients, sha256 {hashlib.sha256(text.encode()).hexdigest()}")
