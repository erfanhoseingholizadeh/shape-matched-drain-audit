# Shape-matched audit of shortcut reliance in pneumothorax classifiers

This repository holds the code behind the paper

> Do Synthetic Chest-Drain Edits Reveal Shortcut Reliance? A Shape-Matched Audit of Pneumothorax Classifiers.
> Erfan Hoseingholizadeh, Mahdieh Khanmohammadi, Mina Farmanbar. Submitted to ISBI 2027.

The paper asks whether pneumothorax classifiers trained on NIH ChestX-ray14 rely on chest drains. A synthetic tube is drawn into films that show a pneumothorax without a drain (the targets), and the change in the classifier's logit is measured. Because a tube-shaped patch of tissue can move the score by itself, each edit is paired with a control that copies another patient's pixels into the same footprint. The main contrast is the rendered tube minus that tissue patch. We audit five DenseNet-121 and five ResNet-50 checkpoints, check the audit on checkpoints that were trained with a planted tube, and read a held-out half of the test split once under a protocol fixed beforehand. The paper has the results and their limits; this README does not repeat them.

Code is at https://github.com/erfanhoseingholizadeh/shape-matched-drain-audit. The data, the trained checkpoints, the per-image scores and the composites are not in this repository. The derived files (split manifests, checkpoints, per-image scores, composite metadata) will be deposited on acceptance and follow the licence of the NEATX labels (CC BY-NC-SA).

## Layout

| Path | What is in it |
|---|---|
| `src/cxr_shortcuts/` | the package: data and split code, models, training, the tube generator and compositing, scoring, bootstrap |
| `data_prep/` | manifests, reserved-image lists, population and target derivations |
| `training/` | config generators, the checkpoint lists the scoring jobs read, the 512-pixel memory probe |
| `editing/` | the planting plan and the planted training sets |
| `scoring/` | jobs that score films with a trained classifier: validity, audits, manipulation checks, the held-out run, artifact statistics |
| `analysis/` | readouts, intervals, the registered verdict rule, the bootstrap, recomputations on the validation populations |
| `paper_values/` | the scripts that give the paper's numbers, table and figures, and the number checker (the paper itself is not part of this repository) |
| `slurm/` | the Slurm scripts of the cluster runs, one folder per stage; the settings loader, the renderer, `submit.sh`, `link_stage_dirs.sh` |
| `configs/` | training configs (see "Config names") |
| `docs/number_sources.csv` | every number in the paper, the script that gives it and the recorded output it is read from |
| `tests/` | a small test set that runs in under a minute |
| `requirements/` | pinned requirements with hashes, CPU and CUDA |
| `settings.toml` | the few site settings (cluster paths, node names), with defaults |

Each stage folder is a Python package; the scripts put the repository root on `sys.path` themselves, so `python3 analysis/heldout_analysis.py` works from the repository root. The classifier code is used from `src/` (see Setup).

What is in the repository and why, by group (371 numbers of the paper are traced in `docs/number_sources.csv`):

| Group | Files | Reason |
|---|---|---|
| package: audit and composite-and-score runners | 2 | the runners that build the four arms for each target and score them; the held-out run, the exploratory runs and the planted-checkpoint runs all call them |
| package: data, models, training, evaluation | 27 | ImageNet-pretrained DenseNet-121 and ResNet-50 training (runners/train.py), the patient-level split and manifest code, the guards on the held-out half, and the bootstrap and evaluation primitives every analysis imports |
| package: synthetic tube, compositing and the tissue control | 12 | the parametric tube generator, the log-domain blend, the tissue and disk controls and the lung-mask geometry (everything runners/composite_and_score.py imports, including the realism-panel modules it loads) |
| scripts: analyses, decision rule and bootstrap | 18 | the pooled contrasts, the two-stage patient bootstrap, the registered verdict rule, the planted-checkpoint readouts, the further-seed and 512-pixel readouts, the probability-scale summaries |
| scripts: classifier validity, populations and margins | 25 | the validity clauses and their intervals for the ten original checkpoints, the cleaned and never-selected validation populations, the target populations of each half, and the margins |
| scripts: composite, score and manipulation-check jobs | 7 | the jobs that composite the arms, score them with the checkpoints, run the manipulation check, and compute the artifact statistics of the four arms |
| scripts: earlier audit, imported | 6 | the earlier negative-target audit and its operating-point definitions, imported by the later composite and analysis jobs |
| scripts: inputs to training and scoring | 10 | the 14-label manifest, the reserved-image list, the planting plans and planted training sets, the training configs of the further-seed and 512-pixel runs, and the checkpoint lists the scoring jobs read |
| scripts: the paper's numbers, table and figures | 12 | the value scripts that read recorded outputs and give each cited number, the generators of the verdict table and the results figure, and the scripts that draw the real-example strip and the design overview |

What was left out of this release and why:

| Left out | Reason |
|---|---|
| Drain-detector and generative-edit code | no longer reported in the ISBI paper |
| Data, checkpoints, per-image scores, composites, checkpoint lists | data and derived files are not released with the code; derived files follow the NEATX licence and are deposited on acceptance |
| Earlier pilots and Phase B probes | not behind any cited number (early pilots, the probes of a closed phase, the realism-panel simulations) |
| Governance, consistency and claim-guard tooling | checks of the study record and of the manuscript text; they need files that are not released |
| Cluster transport and queue helpers | move files over the owner's login connection or poll the queue; they compute nothing |
| Tests of the working repository | they test the study tooling; the release has its own small test set |

## Requirements

Python 3.12. The runs used PyTorch 2.6.0 (CUDA 12.4 build on the cluster) with the package versions pinned in `requirements/`. The CPU lock was install-tested (below); the CUDA lock was resolved from the versions recorded for the cluster environment and has not been install-tested. The runs used one A100 GPU per job (the jobs pinned to one node ran on an 80 GB card); the 224-pixel runs used batch size 128 and the 512-pixel runs used micro-batches because of memory. Scoring runs on a GPU or on CPU. Everything downstream of the recorded outputs (analyses, table, figures, number checks) runs on a laptop in seconds to minutes.

## Data

No data is included. Get these yourself and respect their terms:

| Data | Where | Terms |
|---|---|---|
| NIH ChestX-ray14 images and `Data_Entry_2017.csv` | NIH Clinical Center release, mirrored on Kaggle: https://www.kaggle.com/datasets/nih-chest-xrays/data | see the NIH release page and the Kaggle dataset page |
| NEATX non-expert tube annotations (`NIH-CX14_TubeAnnotations_NonExperts_aggregated.csv`) | https://zenodo.org/records/14944064 (DOI 10.5281/zenodo.14944064) | CC BY-NC-SA 2.0 |
| CheXmask lung masks (`OriginalResolution/ChestX-Ray8.csv`, about 2.2 GB) | https://physionet.org/content/chexmask-cxr-segmentation-data/1.0.0/ | CC BY 4.0 (version 1.0.0) |

Expected layout under the data root (the config files and Slurm scripts use these relative names): `data/raw/nih_chestxray14/images/`, `data/raw/neatx/neatx.csv` (the NEATX file above, renamed), `data/raw/chexmask/chestxray8.csv` (the CheXmask file above, renamed). The images are 1024 by 1024 PNG files; the edits are made at native resolution and resized only at the classifier's input.

## Setup

```
python3.12 -m venv .venv
.venv/bin/pip install --require-hashes -r requirements/cpu.lock.txt
```

On the cluster use `requirements/cuda.lock.txt` instead. The CPU lock installed in a fresh environment in 73 seconds. The package is used from `src/` (`PYTHONPATH=src` or `pip install -e .`).

Site settings live in `settings.toml`: `cluster_root` (a directory holding `data/`, the virtual environment `venv312_gpu/` and one folder per stage), `nodelist` and `exclude` (node names; empty by default). Environment variables `CXR_CLUSTER_ROOT`, `CXR_NODELIST` and `CXR_EXCLUDE` override the file, and `CXR_SETTINGS` points to another file.

Slurm scripts hold `@CXR_...@` placeholders. Submit them through `slurm/submit.sh`, which fills the placeholders and calls `sbatch`:

```
slurm/link_stage_dirs.sh                               # once: links <cluster_root>/<stage> to this repository
slurm/submit.sh slurm/heldout/heldout.sbatch
```

The original runs deployed a copy of the code into one folder per stage. This release uses one tree for all stages, so run `link_stage_dirs.sh` once.

## Stages, in order

Hardware and time are the limits requested in the Slurm files; actual times were shorter. A script without a command line (marked "no options" in its header) starts working as soon as it is run. Slurm folders are named for the stage they ran. Names that start with `e3`, `e4` and `e5` mark three later experiments: planted controls near the equivalence margin (E3), ten more seeds per architecture (E4) and the 512-pixel replication with its 224-pixel control (E5).

| # | Stage | Command | Needs |
|---|---|---|---|
| 1 | Split manifest from the NIH metadata and the NEATX labels | `python3 -m cxr_shortcuts.runners.build_manifest --nih-metadata-csv ... --neatx-csv ...` | CPU, minutes |
| 2 | 14-label manifest, reserved-image list | `python3 data_prep/build_multilabel_manifest.py`, `python3 data_prep/build_reserved_image_ids.py` | CPU, minutes |
| 3 | Ten original checkpoints (5 DenseNet-121, 5 ResNet-50), then ViT-B/16 attempts | `slurm/original_checkpoints/train_seed0.sbatch` to `train_seed4`, `slurm/resnet50_vit/train_resnet50_seed0.sbatch` to `seed4`, `slurm/resnet50_vit/` and `slurm/vit_retry/` for ViT-B/16 | 1 GPU each, 4 h limit |
| 4 | Validity of the checkpoints (AUC, median negative probability), per-image validation predictions | `slurm/original_checkpoints/eval_seed0.sbatch` to `eval_seed4`, `val_full_rescan.sbatch`, `slurm/vit_retry/val_full_rescan_resnet50.sbatch`, then `python3 data_prep/clean_population.py` and the scripts of `analysis/` named `clean_recompute`, `assigned_recompute`, `gap_recompute`, `interval_widths`, `quarter_table` | 1 GPU, 30 min; CPU for the recomputations |
| 5 | Planted checkpoints: planting plan, planted training sets, training | `python3 editing/planting_plan.py`, `slurm/planted_control/` (DenseNet-121 zero, weak, strong), `slurm/planted_intermediate/` (intermediate strengths), `slurm/resnet50_planted/` (ResNet-50) | CPU 6 h per set; 1 GPU 4 h per run |
| 6 | Manipulation check, audits on the negative targets and the two-by-two | the Slurm files of `slurm/manipulation_check/`, `slurm/fresh_audit/`, `slurm/planted_intermediate/`, `slurm/planted_intermediate_real/` and `slurm/resnet50_planted/twobytwo.sbatch` | 1 GPU, 1 h each |
| 7 | Exploratory audit on positive targets | `slurm/positive_audit/composite.sbatch`, `score.sbatch` (or `score_cpu.sbatch`), then `python3 analysis/positive_analysis.py` | CPU 3 h; GPU 1.5 h |
| 8 | Held-out read, once, under the frozen protocol | `slurm/heldout/heldout.sbatch` (prepare, composite, score), then `python3 analysis/heldout_analysis.py` | CPU 5 h; see the note below |
| 9 | Further seeds, near-margin plantings, ResNet-50 weak planting | `python3 training/make_further_run_configs.py`, `slurm/further_runs/`, `python3 analysis/further_run_readout.py` | 1 GPU 3 h per run |
| 10 | 512-pixel replication and its 224-pixel control | `python3 training/px512_memory_probe.py`, `python3 training/record_px512_microbatch.py`, `python3 training/make_px512_configs.py`, `slurm/replication_512/`, `python3 analysis/px512_readout.py` | 1 GPU, 2 to 3 h per run |
| 11 | Artifact statistics of the four arms, probability-scale summaries | `slurm/edit_artifacts/artifact_stats.sbatch`, `python3 analysis/artifact_summary.py`, `python3 analysis/probability_scale.py` | CPU 1 h, then minutes |

Smoke test before every real submission: run the same script on a tiny subset first (`--smoke-n` where a script has it).

The held-out half is guarded by `src/cxr_shortcuts/data/gate_guard.py`: loaders drop its rows, and the one entry point that reads them logs every read. The paper's read was made once; running stage 8 again is a second read and is not a confirmatory result.

Exit codes of the GPU Slurm scripts: 0 success; 4 a finished training whose logged selection-half digest is not the registered one; 5 the allocated GPU was occupied by another process; 6 the occupancy check could not read the card; any other non-zero code is a failure of the job itself.

## Regenerating the numbers and figures

The scripts below regenerate the paper's numbers and figures from the data and the recorded outputs. The paper is not part of this repository.

The numbers are read from recorded outputs, which are the derived files deposited on acceptance. Put them under `review/` with the paths in `docs/number_sources.csv` (the column `recorded_output`).

1. Numbers: run the value scripts in `paper_values/` (`heldout_values.py`, `further_run_values.py`, `donor_clustering_values.py`, `validity_values.py`, `margins_and_spans.py`, `donor_clustering_se.py`) and `data_prep/dataset_split_counts.py`, `analysis/artifact_summary.py`, `analysis/probability_scale.py`, then `python3 paper_values/check_number_sources.py`, which re-reads each recorded number and compares it with the value printed in the paper (the column `value_as_printed` of `docs/number_sources.csv`). A few numbers are design constants or come from the study record and are listed as such.
2. The verdict table and the results figure: `python3 paper_values/isbi_tables.py` and `python3 paper_values/isbi_figures.py`.
3. The strip of the edit on one target needs the persisted composite of one development-half target: `python3 paper_values/isbi_edit_strip.py render`.
4. The audit-design overview is a schematic that uses no image: `python3 paper_values/isbi_design_figure.py`.

One number pair cannot be regenerated by the shipped code: the paper's count of registered population properties (25, of which 3 failed) comes from an earlier revision of a checker that has since grown. It is marked in `docs/number_sources.csv`. `paper_values/heldout_values.py` and `paper_values/further_run_values.py` are shipped without the blocks that compute values the paper does not cite (they read outputs of analyses that are not part of this release); the values the paper does cite are unchanged.

## Tests

```
python3 -m pytest tests -q
```

The tests cover imports, the tube generator and the blend on a synthetic image, the registered verdict rule on fixed numbers, the bootstrap on a toy array, the settings and template rendering, and the number checker. They need no data and run in under a minute on CPU.

## Config names

Training configs are in `configs/`: `densenet121_original.yaml` and `resnet50_original.yaml` (the original checkpoints; seeds are set on the command line by the Slurm scripts), `vit_b16_original.yaml` and `vit_b16_retry.yaml` (the ViT-B/16 attempts), `densenet121_planted_{zero,weak,mid30,mid50,strong}.yaml` and `resnet50_planted_{zero,mid30,strong}.yaml` (planted checkpoints), `planted_control_densenet121.yaml`, `e3_*`, `e4_*`, `e5_*` and `e5ctl224_*` (written by the generators in `training/`), and `stage0_*.yaml` (the earlier base configs). The files copied from the cluster keep the paths of the cluster run, with the same placeholders as the Slurm scripts.

## Known gaps

- `scoring/val_report_eval.py` is the evaluation script of every validity JSON. The copies deployed for the ResNet-50 and ViT-B/16 runs were byte-identical to it (sha256 equal), so this release ships it once.
- Some analysis scripts read further recorded files (operating-point definitions, calibration fits) whose producers are not shipped because no number of the paper depends on them.
- Slurm scripts are kept as they ran, apart from the placeholders and the renamed paths; they assume the folder layout under `cluster_root` described above.

## Citation

Please cite the paper. `CITATION.cff` has the metadata of this repository.

```
@unpublished{hoseingholizadeh2027drain,
  title  = {Do Synthetic Chest-Drain Edits Reveal Shortcut Reliance? A Shape-Matched Audit of Pneumothorax Classifiers},
  author = {Hoseingholizadeh, Erfan and Khanmohammadi, Mahdieh and Farmanbar, Mina},
  note   = {Submitted to ISBI 2027; code at https://github.com/erfanhoseingholizadeh/shape-matched-drain-audit}
}
```

## Licence

The code is under the Apache License, Version 2.0 (`LICENSE`). The derived files (split manifests, checkpoints, per-image scores, composite metadata) will follow the licence of the NEATX labels, CC BY-NC-SA, and will be deposited separately on acceptance; they are not included in this repository. The NIH images, the NEATX labels and the CheXmask masks keep their own terms.
