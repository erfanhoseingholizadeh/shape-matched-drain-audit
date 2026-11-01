# Shape-matched audit of shortcut reliance in pneumothorax classifiers

> Do Synthetic Chest-Drain Edits Reveal Shortcut Reliance? A Shape-Matched Audit of Pneumothorax Classifiers.
> Erfan Hoseingholizadeh, Mahdieh Khanmohammadi, Mina Farmanbar. Submitted to ISBI 2027.

This is the code behind the paper. It trains pneumothorax classifiers on NIH ChestX-ray14, draws a synthetic chest-drain tube into films that show a pneumothorax but no drain, and measures how far the classifier's logit moves.

Each edit is paired with a control that copies another patient's pixels into the same footprint, so the main contrast is the rendered tube minus a tube-shaped patch of tissue. The audit is checked on classifiers trained with a planted tube, and a held-out half of the test split is read once under a protocol fixed beforehand. The paper has the results and their limits; this README only says how to run the code.

## Quick start

```
git clone <url of this repository>
cd shape-matched-drain-audit
python3.12 -m venv .venv
source .venv/bin/activate
pip install --require-hashes -r requirements/cpu.lock.txt
export PYTHONPATH=src
python3 -m pytest tests -q
```

The tests need no data. For a machine with a GPU, install `requirements/cuda.lock.txt` instead (PyTorch 2.6.0, CUDA 12.4 build). That lock was resolved from the versions recorded for the cluster environment and has not been install-tested.

## Getting the data

No data is included. Download these yourself and keep to their terms.

| Data | Where | Terms |
|---|---|---|
| NIH ChestX-ray14 images and `Data_Entry_2017.csv` | NIH Clinical Center release, mirrored on Kaggle: https://www.kaggle.com/datasets/nih-chest-xrays/data | see the NIH release page and the Kaggle dataset page |
| NEATX non-expert tube annotations (`NIH-CX14_TubeAnnotations_NonExperts_aggregated.csv`) | https://zenodo.org/records/14944064 (DOI 10.5281/zenodo.14944064) | CC BY-NC-SA 2.0 |
| CheXmask lung masks (`OriginalResolution/ChestX-Ray8.csv`, about 2.2 GB) | https://physionet.org/content/chexmask-cxr-segmentation-data/1.0.0/ | CC BY 4.0 (version 1.0.0) |

Put the files in this layout, inside the repository folder:

```
data/
  raw/
    nih_chestxray14/
      Data_Entry_2017.csv
      images/
        00000001_000.png
        00000001_001.png
        ...
    neatx/
      neatx.csv
    chexmask/
      chestxray8.csv
```

`neatx.csv` is the NEATX file above, renamed. `chestxray8.csv` is the CheXmask file above, renamed. All the PNG files go directly into `images/`; the Kaggle download keeps them in twelve `images_00N/images/` folders, so move them into one. The images are 1024 by 1024 pixels. The edits are made at that size and resized only at the classifier's input.

The scripts create `data/processed/` (manifests, planted training sets) and `review/` (the recorded outputs that the analyses read) themselves. The derived files (split manifests, checkpoints, per-image scores, composite metadata) will be deposited on acceptance under the licence of the NEATX labels.

## Configuration

Site settings are in `settings.toml`:

```
cluster_root = "${HOME}/cxr_shortcuts"
nodelist = ""
exclude = ""
partition_gpu = "gpu"
partition_cpu_small = "cpu"
partition_cpu_mid = "cpu"
partition_cpu_large = "cpu"
```

`cluster_root` is a folder that holds `data/`, the virtual environments and one folder per stage. `nodelist` pins the jobs that carry a node list to one node; `exclude` names nodes to avoid; an empty value drops that line from the job. The partition settings name the Slurm partitions.

An environment variable `CXR_<KEY>` (for example `CXR_CLUSTER_ROOT="$HOME/other_root"`) overrides the file, and `CXR_SETTINGS=/path/to/other.toml` points to another file. `${HOME}` is expanded.

The training configs and the Slurm scripts address everything through `cluster_root`, on a single machine as well as on a cluster. Set it up once:

```
ROOT="$(python3 -c 'import sys; sys.path.insert(0, "."); from slurm import cxr_settings; print(cxr_settings.get("cluster_root"))')"
mkdir -p data logs outputs "$ROOT"
ln -s "$PWD/data" "$ROOT/data"
slurm/link_stage_dirs.sh
for f in configs/*.yaml; do python3 slurm/render.py "$f" --out "$f"; done
```

The first line reads `cluster_root` from the settings. The third makes the downloaded data visible under it (the second creates `logs/`, where the jobs write their logs, and `outputs/`). `link_stage_dirs.sh` links `<cluster_root>/<stage>` to this repository for every stage, so all stages share one tree. The last line fills the `cluster_root` placeholders in the configs; it edits the files in `configs/`.

The Slurm jobs use three Python environments under `cluster_root`: `venv312_gpu` for GPU jobs and `venv312_panel` and `venv312` for CPU jobs. Create them like the one in the quick start, with the CUDA lock for `venv312_gpu` and the CPU lock for the other two:

```
python3.12 -m venv "$ROOT/venv312_gpu"
"$ROOT/venv312_gpu/bin/pip" install --require-hashes -r requirements/cuda.lock.txt
```

The Slurm jobs also read small marker files from the repository root and write them into each job's manifest: `COMMIT` (the git commit hash) and `SRC_TREE_DIGEST` (a digest of the Python files under `src/`). A few jobs also read `SRC_TREE_DIGEST_RECIPE` (one line saying how the digest was made) and `INPUTS_DIGEST` (a digest of `inputs/`).

```
git rev-parse HEAD > COMMIT
(cd src && LC_ALL=C find . -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1) > SRC_TREE_DIGEST
echo "find . -name '*.py' in src, sorted, sha256sum, sha256sum (LC_ALL=C)" > SRC_TREE_DIGEST_RECIPE
```

## Running the pipeline

The steps follow the paper. Each shows the command to run on one machine and, for the GPU steps, the Slurm script that did the same job on the cluster. Steps marked GPU need a GPU in practice; the scoring scripts also run on CPU with `--device cpu`. A script marked "no options" in its header starts working as soon as it is run.

Most scripts with a smoke-test option take `--smoke-n N` (a tiny subset); `composite_planted_training_set.py` takes `--limit N` and `manipulation_check.py` takes `--limit-images N`. The Slurm scripts pass the count on as `SMOKE` or `SMOKE_N`. Run the smoke test before every full run.

**1. Manifests (CPU).**

```
python3 -m cxr_shortcuts.runners.build_manifest
python3 data_prep/build_multilabel_manifest.py
cp data/processed/manifests/nih_pneumothorax_multilabel.csv data/
python3 data_prep/build_reserved_image_ids.py
```

`build_manifest` reads `Data_Entry_2017.csv` and `neatx.csv` and writes the patient-level split manifest `data/processed/manifests/nih_pneumothorax.csv`. `build_multilabel_manifest.py` adds the 14 NIH labels; the copy puts it where the configs and Slurm scripts look. `build_reserved_image_ids.py` writes the list of films that the image loaders refuse to read: the films of the held-out half and of the patients in the evaluation arms of the realism check (`eval_paired` and `eval_unpaired_reference`). It reads `data/processed/manifests/discriminator_split.csv`, the split of patients for the realism check (a split manifest, among the derived files deposited on acceptance). `data_prep/dataset_split_counts.py` counts the images and patients of each split.

**2. Train the original checkpoints (GPU).** Five seeds per architecture.

```
python3 -m cxr_shortcuts.runners.train --config configs/densenet121_original.yaml --seed 0 --image-size 224 --device cuda
python3 -m cxr_shortcuts.runners.train --config configs/resnet50_original.yaml --seed 0 --image-size 224 --device cuda
```

Repeat for seeds 1 to 4. A checkpoint is written to `outputs/<name>_seed<N>_res224/best_checkpoint.pt`. On Slurm:

```
slurm/submit.sh slurm/original_checkpoints/train_seed0.sbatch
slurm/submit.sh slurm/resnet50_vit/train_resnet50_seed0.sbatch
```

with `train_seed1` to `train_seed4` and `train_resnet50_seed1` to `seed4` for the other seeds. The ViT-B/16 attempts that failed the validity threshold are `slurm/resnet50_vit/train_vit_b16_seed0.sbatch` to `seed4` and the retry in `slurm/vit_retry/`.

**3. Validity of the checkpoints (GPU or CPU).** The two validity clauses (AUC and median negative probability) on the validation report half, and the per-image predictions on every validation film that the later recomputations read.

```
python3 scoring/val_report_eval.py --manifest data/nih_pneumothorax_multilabel.csv --nih-root data/raw/nih_chestxray14/images --checkpoint outputs/stage0_densenet121_nih_phaseA_seed0_res224/best_checkpoint.pt --image-size 224 --output outputs/val_report_eval_seed0.json
python3 scoring/val_full_rescan_densenet121.py --manifest data/nih_pneumothorax_multilabel.csv --nih-root data/raw/nih_chestxray14/images --checkpoint <checkpoint of seed 0> --checkpoint <checkpoint of seed 1> --output-csv outputs/val_full_per_image.csv --output-json outputs/val_full_rescan.json
python3 data_prep/clean_population.py
```

Add `--backbone resnet50` to `val_report_eval.py` for ResNet-50, and use `val_full_rescan_resnet50.py` for the ResNet-50 per-image predictions. The rescan scripts take one `--checkpoint` per seed (five). On Slurm:

```
slurm/submit.sh slurm/original_checkpoints/eval_seed0.sbatch
slurm/submit.sh slurm/original_checkpoints/val_full_rescan.sbatch
slurm/submit.sh slurm/vit_retry/val_full_rescan_resnet50.sbatch
```

with `eval_seed1` to `eval_seed4` for the other seeds. `clean_population.py` derives the validation patients that were in neither selection half.

**4. Planting plan and planted checkpoints (training: GPU).** Classifiers trained with a tube drawn into some training films, at several strengths (`zero`, `weak`, `mid30`, `mid50`, `strong`), as positive controls.

```
python3 editing/planting_plan.py
python3 editing/composite_planted_training_set.py --strength strong --manifest-path data/processed/manifests/nih_pneumothorax_multilabel.csv --nih-root data/raw/nih_chestxray14/images --chexmask-csv data/raw/chexmask/chestxray8.csv --output-image-dir data/processed/planted_insertion/strong --resolved-manifest-out outputs/resolved_manifest_strong.csv --out-dir review/planted_control/outputs --n-workers 8
python3 -m cxr_shortcuts.runners.train --config configs/densenet121_planted_strong.yaml --seed 0 --image-size 224 --device cuda
```

Add `--limit N` to `composite_planted_training_set.py` for a smoke test. On Slurm:

```
slurm/submit.sh slurm/planted_control/planting_plan.sbatch
slurm/submit.sh slurm/planted_control/composite_strong.sbatch
slurm/submit.sh slurm/planted_control/train_strong_seed0.sbatch
slurm/submit.sh slurm/resnet50_planted/train.sbatch --export=ALL,CONFIG=resnet50_planted_strong,SEED=0
```

The other strengths are `composite_zero`, `composite_weak` and `train_<strength>_seed0` to `seed2` in the same folder; `slurm/planted_intermediate/` holds the same jobs for `mid30` and `mid50`. `CONFIG` is the config's name without `configs/` and `.yaml`.

**5. Audits and manipulation check (GPU).** The manipulation check confirms that a planted checkpoint reacts to the drawn tube; the audits score negative targets with the four arms (rendered tube, tube-shaped tissue, rendered disk, disk-shaped tissue); the two-by-two audit separates shape from content.

```
python3 scoring/manipulation_check.py --conflict-set <conflict_set>.csv --nih-root data/raw/nih_chestxray14/images --checkpoint-manifest <checkpoint_manifest>.json --out-dir outputs/manipulation_check --backbone densenet121 --strengths strong,weak,zero --limit-images 20
python3 scoring/fresh_composite_audit.py composite --pool-ids <pool_ids>.txt --split-manifest data/nih_pneumothorax_multilabel.csv --rca-csv <chexmask_rows_of_the_pool>.csv --nih-root data/raw/nih_chestxray14/images --composite-dir composites/fresh --checkpoint-manifest <checkpoint_manifest>.json --out-dir outputs/fresh_composite --smoke-n 5
python3 scoring/fresh_composite_audit.py score --composite-dir composites/fresh --anchor-json <anchor>.json --checkpoint-manifest <checkpoint_manifest>.json --out-dir outputs/fresh_score --smoke-n 5
python3 scoring/twobytwo_audit.py score --composite-dir composites/fresh --anchor-json <anchor>.json --checkpoints <checkpoints>.json --nih-root data/raw/nih_chestxray14/images --arch resnet50 --out-dir outputs/twobytwo --smoke-n 5
```

The manipulation check takes `--limit-images N` as its smoke option; the other three take `--smoke-n N`. Drop them for the full run. `<checkpoint_manifest>.json` lists the checkpoints being audited, `<anchor>.json` is the anchor file of the audit, `<conflict_set>.csv` the validation positives and negatives of the manipulation check, and `<pool_ids>.txt` the film ids of the evaluation pool; these inputs come from earlier steps of the cluster runs and are not produced by the scripts in this repository. `fresh_composite_audit.py` and `twobytwo_audit.py` also take `--backbone` or `--arch` for ResNet-50. On Slurm:

```
slurm/submit.sh slurm/manipulation_check/manipulation_check_real.sbatch
slurm/submit.sh slurm/fresh_audit/composite_originals.sbatch
slurm/submit.sh slurm/fresh_audit/planted.sbatch
slurm/submit.sh slurm/planted_intermediate_real/manip_real.sbatch
slurm/submit.sh slurm/planted_intermediate_real/audit_real.sbatch
slurm/submit.sh slurm/resnet50_planted/twobytwo.sbatch --export=ALL,ARCH=resnet50,SMOKE=0
```

The Slurm jobs read these inputs from `inputs/` in the repository root.

**6. Exploratory audit on positive targets (composite: CPU, score: GPU or CPU).**

```
python3 scoring/positive_audit.py prepare --rca-csv data/raw/chexmask/chestxray8.csv --out-json inputs/exploratory_inputs.json --out-masks inputs/chexmask_targets.csv
python3 scoring/positive_audit.py composite --input-json inputs/exploratory_inputs.json --rca-csv inputs/chexmask_targets.csv --nih-root data/raw/nih_chestxray14/images --composite-dir composites/real --out-dir outputs/positive_composite --smoke-n 5
python3 scoring/positive_audit.py score --composite-dir composites/real --checkpoints inputs/<checkpoints>.json --arch densenet121 --out-dir outputs/positive_score_densenet121 --smoke-n 5
python3 analysis/positive_analysis.py --densenet <scores_densenet121.csv> --resnet <scores_resnet50.csv> --inputs inputs/exploratory_inputs.json --out review/positive_audit/positive_analysis.json
```

`--checkpoints` is a JSON list of objects with `label`, `checkpoint`, `family` and `strata`. Drop `--smoke-n` for the full run. `prepare` stops if a population check fails (see the safeguards below). On Slurm: `slurm/positive_audit/composite.sbatch` (variables `COMP`, `OUT`, `SMOKE`), `score.sbatch` or `score_cpu.sbatch` (variables `ARCH`, `CKPTS`, `COMP`, `OUT`, `SMOKE`) and `twobytwo_planted.sbatch`.

**7. Held-out read (composite: CPU, score: CPU).** The paper's read of the held-out half was made once, under a protocol committed beforehand. A second read is not a confirmatory result. The entry point that opens the held-out half (`--half gate --protocol-commit <40-character hash>` on `prepare`) accepts only the commit that first added the frozen protocol file and checks that the file is unchanged; the paper's protocol commit is not in this repository's history, so a read made here would follow a protocol of your own.

```
python3 scoring/heldout_run.py prepare --half dev --out-dir dev_inputs
python3 scoring/heldout_run.py composite --input-json dev_inputs/inputs.json --expected-inputs-sha256 <inputs_sha256 in dev_inputs/prepare_record.json> --rca-csv dev_inputs/chexmask_targets.csv --nih-root data/raw/nih_chestxray14/images --composite-dir composites/dev --out-dir outputs/dev_composite --smoke-n 5
python3 scoring/heldout_run.py score --composite-dir composites/dev --checkpoints <checkpoints>.json --arch densenet121 --out-dir outputs/dev_score --device cpu --smoke-n 5
python3 analysis/heldout_analysis.py --arch densenet121 --scores <scores_densenet121.csv> --inputs dev_inputs/inputs.json --out outputs/dev_analysis.json
```

These commands run the same code on the development half of the test split, which is how the pipeline can be checked without opening the held-out half. On Slurm, `slurm/heldout/heldout.sbatch` runs the `composite` and `score` jobs (variables `JOB`, `HALF`, `COMP`, `ARCH`, `CKPTS`, `OUT`, `SMOKE`) and `part4.sbatch` the ResNet-50 checks that go with it (`JOB` is `manip`, `audit_planted`, `audit_originals`, `twobytwo` or `positives`).

**8. Further seeds and near-margin plantings (GPU).** Ten more seeds per architecture (the `e4_*` configs), two plantings near the equivalence margin per architecture and a weak ResNet-50 planting (the `e3_*` configs).

```
python3 training/make_further_run_configs.py
python3 training/make_further_run_checkpoint_lists.py --out-dir inputs
python3 -m cxr_shortcuts.runners.train --config configs/e4_densenet121.yaml --seed 5 --image-size 224 --device cuda
python3 scoring/further_run_validity.py score --manifest data/nih_pneumothorax_multilabel.csv --nih-root data/raw/nih_chestxray14/images --checkpoints inputs/ckpts_validity.json --out-dir outputs/further_validity --smoke-n 5
python3 analysis/further_run_readout.py --arch densenet121 --scores <scores_densenet121.csv> --out outputs/further_readout_densenet121.json
```

The `e3_*` configs use seeds 0 to 2 and the `e4_*` configs seeds 5 to 14. On Slurm:

```
slurm/submit.sh slurm/further_runs/train.sbatch --export=ALL,CONFIG=e4_densenet121,SEED=5
slurm/submit.sh slurm/further_runs/val_validity.sbatch --export=ALL,CKPTS=ckpts_validity,OUT=further_validity,SMOKE_N=0
slurm/submit.sh slurm/further_runs/audit_score.sbatch --export=ALL,ARCH=densenet121,CKPTS=ckpts_e4_densenet121,OUT=e4_densenet121,SMOKE_N=0
```

`CKPTS` is the name of a list in `inputs/`, without `.json`. `e3_plan.sbatch` and `e3_composite.sbatch` (variable `STRENGTH`) make the planting plan and the training sets of the `e3_*` plantings.

**9. 512-pixel replication and 224-pixel control (GPU).** The planted and original checkpoints retrained at 512 pixels (the `e5_*` configs), and a 224-pixel control (the `e5ctl224_*` configs).

```
python3 training/make_px512_configs.py
python3 training/make_px512_checkpoint_lists.py --out-dir inputs
python3 -m cxr_shortcuts.runners.train --config configs/e5_densenet121_strong.yaml --seed 0 --image-size 512 --device cuda
python3 scoring/px512_validity.py score --size 512 --manifest data/nih_pneumothorax_multilabel.csv --nih-root data/raw/nih_chestxray14/images --checkpoints inputs/<checkpoints>.json --out-dir outputs/px512_validity --smoke-n 5
python3 analysis/px512_readout.py --scores-dn <scores_densenet121.csv> --scores-rn <scores_resnet50.csv> --validity <validity.json> --margins <margins.json> --out outputs/px512_readout.json
```

The 512-pixel runs need micro-batches because of memory: `training/px512_memory_probe.py --backbone densenet121 --micro-batch 8 --out outputs/probe/densenet121_8_plain.json` measures one setting, and `training/record_px512_microbatch.py outputs/probe` applies the memory rule to the probe outputs and writes `configs/e5_microbatch.json`. On Slurm:

```
slurm/submit.sh slurm/replication_512/train.sbatch --export=ALL,CONFIG=e5_densenet121_strong,SEED=0
slurm/submit.sh slurm/replication_512/train_ctl224.sbatch --export=ALL,CONFIG=e5ctl224_densenet121,SEED=0
```

`probe.sbatch`, `e5_validity.sbatch`, `e5_audit_score.sbatch` and `e5_manip.sbatch` in the same folder run the memory probe, the validity, the audit and the manipulation check.

**10. Artifact statistics and probability-scale summaries (CPU).**

```
python3 scoring/artifact_stats.py stats --composite-dir composites/real --masks-csv inputs/chexmask_targets.csv --out-dir review/edit_artifacts/artifact_stats --smoke-n 5
python3 analysis/artifact_summary.py
python3 analysis/probability_scale.py
```

Drop `--smoke-n` for the full run. On Slurm: `slurm/submit.sh slurm/edit_artifacts/artifact_stats.sbatch --export=ALL,OUT=artifact_stats,SMOKE=0`, which writes `outputs/artifact_stats`; then give that folder to the summary with `python3 analysis/artifact_summary.py --stats-dir outputs/artifact_stats`.

## Running on Slurm

Every Slurm script is submitted through `slurm/submit.sh`, which fills the `cluster_root`, node and partition placeholders from `settings.toml` and then calls `sbatch`:

```
slurm/submit.sh slurm/original_checkpoints/train_seed0.sbatch
slurm/submit.sh slurm/positive_audit/score.sbatch --export=ALL,ARCH=densenet121,CKPTS=<checkpoints>,COMP=real,OUT=positive_score_densenet121,SMOKE=0
```

Do the setup under "Configuration" once first (`link_stage_dirs.sh`, `COMMIT` and `SRC_TREE_DIGEST`, the three environments). Extra `sbatch` options go after the script name.

A script that needs variables reads them from the environment and stops if one is unset. The names are the upper-case words in the script; the ones used above are `ARCH`, `CKPTS`, `COMP`, `OUT`, `CONFIG`, `SEED`, `STRENGTH`, `JOB`, `HALF` and `SMOKE` (or `SMOKE_N`, a count of films; `0` runs everything). Jobs write under `outputs/`, `composites/`, `manifests/` and `logs/` in the repository root, because every stage folder is a link to it.

The GPU scripts check the allocated card before they start and again after a failure. Exit codes:

- 0: success.
- 4: a finished training whose logged selection-half digest is not the registered one.
- 5: the allocated GPU was occupied by another process.
- 6: the occupancy check could not read the card.
- any other non-zero code: a failure of the job itself.

## Reproducing the numbers and figures from the derived files

The paper's numbers are read from recorded outputs, the derived files deposited on acceptance. Put them under `review/`, at the paths in the column `recorded_output` of `docs/number_sources.csv`. Then:

```
python3 data_prep/dataset_split_counts.py
python3 analysis/clean_recompute.py
python3 analysis/assigned_recompute.py
python3 analysis/gap_recompute.py
python3 analysis/interval_widths.py
python3 analysis/quarter_table.py
python3 paper_values/validity_values.py
python3 paper_values/heldout_values.py
python3 paper_values/further_run_values.py
python3 paper_values/donor_clustering_values.py
python3 paper_values/donor_clustering_se.py
python3 paper_values/margins_and_spans.py
python3 analysis/artifact_summary.py
python3 analysis/probability_scale.py
python3 paper_values/check_number_sources.py
```

The value scripts give each cited number from the recorded outputs. `check_number_sources.py` rereads every recorded number and compares it with the value printed in the paper (the column `value_as_printed`).

The verdict table and the results figure (Fig. 2):

```
python3 paper_values/isbi_tables.py
python3 paper_values/isbi_figures.py
```

They write `paper_isbi/tables/` and `paper_isbi/figures/`. Add `--check` to see whether the files there are current.

Fig. 1 is the authors' drawing of the framework with real film crops on its cubes. This script fills them:

```
python3 paper_values/isbi_fig1_real_crops.py render
```

It reads the authors' drawing, saved as `figures_source/framework_drawing.png` (the drawing is not part of this repository); the inspection composite of the example target, `review/positive_audit/inspect/00001331_000.npz`; and the donor film, a copy of the NIH ChestX-ray14 image `00008590_000.png`, saved as `review/fig1_real_crops/donor/00008590_000.png`. The example target is chosen by the rule in `paper_values/edit_example.py` (the first development-half target by image identifier among the persisted composites), and every file is checked against its recorded sha256. The output is `paper_isbi/figures/combined.png`. The crops show NIH images and are illustrations only.

## Reproducibility

Every number in the paper is traced in `docs/number_sources.csv` (360 rows): the script that gives it and the recorded output it is read from. `paper_values/check_number_sources.py` checks them all. The numbers that the released code computes rerun from the recorded outputs and match the values printed in the paper.

The dependencies are pinned with SHA-256 hashes in `requirements/` (Python 3.12). If you retrain or rescore, the only difference to expect is GPU floating-point arithmetic. A device check scored the original and zero-strength planted checkpoints of both architectures on a GPU and on CPU: the primary contrast of every checkpoint moved by at most 0.0009 logits, and the pooled contrasts by at most 0.00008.

## Safeguards against data leakage

Each safeguard is in code; the module is named.

- **Patient-level splits.** Patients are assigned to train, validation and test by a seeded split, and the code stops if one patient appears in two splits: `assign_patient_splits` and `assert_no_patient_leakage` in `src/cxr_shortcuts/data/manifests.py`, called by `runners/build_manifest.py` and again by `runners/train.py`.
- **Held-out half.** The test split is halved by patient (`assign_gate_subset` in `data/manifests.py`). The loaders in `src/cxr_shortcuts/data/gate_guard.py` drop the held-out half's rows, and the one entry point that reads them, `confirmatory_entry`, needs the protocol commit and logs every read. `data/reserved_guard.py` refuses to load any film on the reserved-image list.
- **Selection and report halves.** The validation patients are halved (`assign_val_subset` in `data/manifests.py`). `runners/train.py` picks the epoch on the selection half only and logs the patient lists of both halves with their digests. The further-seed and 512-pixel trainings (`slurm/further_runs/train.sbatch`, `slurm/replication_512/train.sbatch` and `train_ctl224.sbatch`) compare the logged selection-half digest with the registered one and exit with code 4 if it differs.
- **Donors.** `data_prep/exploratory_populations.py` and `data_prep/heldout_populations.py` build the targets and donors and check them, and `scoring/positive_audit.py prepare` and `scoring/heldout_run.py prepare` stop if a check fails. The checks: donor films come from patients other than the target's, from the same half or stratum as the target, with no NEATX drain label on any film of that patient, and the placements of one target use different donor patients. In the held-out run the donor film also has the target's view position.
- **Targets.** The same checks exclude the patients of the realism check's evaluation arms (`eval_paired` and `eval_unpaired_reference`) from the targets and the donors.
- **Strata and the selection halves.** `data_prep/exploratory_populations.py` also checks the validation strata of the exploratory audit against the two validation selection halves (`S_int` and `S_str` in `data_prep/clean_population.py`): every validation stratum must be disjoint from `S_int`, and the clean stratum from `S_str` as well. One stratum (`R_int_Q`) lies inside `S_str`, the selection half of the planted checkpoints, on purpose; the audit reports it separately.

## Repository layout

| Path | What is in it |
|---|---|
| `src/cxr_shortcuts/` | the package: data and split code, models, training, the tube generator and compositing, scoring, bootstrap |
| `data_prep/` | manifests, reserved-image lists, population and target derivations |
| `training/` | config generators, the checkpoint lists the scoring jobs read, the 512-pixel memory probe |
| `editing/` | the planting plan and the planted training sets |
| `scoring/` | jobs that score films with a trained classifier: validity, audits, manipulation checks, the held-out run, artifact statistics |
| `analysis/` | readouts, intervals, the registered verdict rule, the bootstrap, recomputations on the validation populations |
| `paper_values/` | the scripts that give the paper's numbers, table and figures, and the number checker |
| `slurm/` | the Slurm scripts of the cluster runs, one folder per stage; the settings loader, the renderer, `submit.sh`, `link_stage_dirs.sh` |
| `configs/` | training configs |
| `docs/number_sources.csv` | every number in the paper, the script that gives it and the recorded output it is read from |
| `tests/` | a small test set |
| `requirements/` | pinned requirements with hashes, CPU and CUDA |
| `settings.toml` | the few site settings (cluster paths, node names), with defaults |

Each stage folder is a Python package, and the scripts put the repository root on `sys.path` themselves, so `python3 analysis/heldout_analysis.py` works from the repository root.

## Tests

```
python3 -m pytest tests -q
```

The tests cover imports, the tube generator and the blend on a synthetic image, the registered verdict rule on fixed numbers, the bootstrap on a toy array, the settings and template rendering, and the number checker. They need no data.

## Citation

Please cite the paper. `CITATION.cff` has the metadata of this software.

```
@unpublished{hoseingholizadeh2027drain,
  title  = {Do Synthetic Chest-Drain Edits Reveal Shortcut Reliance? A Shape-Matched Audit of Pneumothorax Classifiers},
  author = {Hoseingholizadeh, Erfan and Khanmohammadi, Mahdieh and Farmanbar, Mina},
  note   = {Submitted to ISBI 2027}
}
```

## Licence

The code is under the Apache License, Version 2.0 (`LICENSE`). The NIH images, the NEATX labels and the CheXmask masks keep their own terms. The derived files (split manifests, checkpoints, per-image scores, composite metadata) will be deposited on acceptance and follow the licence of the NEATX labels, CC BY-NC-SA; they are not in this repository.
