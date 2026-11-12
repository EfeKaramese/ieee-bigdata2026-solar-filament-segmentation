# Segment, Then Select: PQ-Aware Component Scoring for Solar Filament Segmentation

Solution to the **IEEE BigData Cup 2026, Challenge 02 — Solar Filament Segmentation Challenge 2026**
([Kaggle *filament-segmentation-2026*](https://www.kaggle.com/competitions/filament-segmentation-2026)): instance
segmentation of solar filaments in full-disk GONG H-α images (MAGFiLO), scored with Panoptic Quality (PQ) against
every annotator record.

The 4-page report is in [`report/report.pdf`](report/report.pdf).

## Results

All numbers are out of fold (OOF) on the 707 training images, computed with the organisers' self-evaluation
functions (included unchanged in [`filament/official_eval.py`](filament/official_eval.py)).

| | PQ | SQ | RQ | TP / FP / FN |
|---|---|---|---|---|
| All components (no selection) | 0.409 | 0.675 | 0.607 | 5661 / 4807 / 2538 |
| **With the PQ-aware scorer (submitted)** | **0.442** | 0.679 | 0.651 | 5157 / 2482 / 3042 |

- Scorer gain: +0.033 PQ, 95% paired-bootstrap CI [+0.028, +0.037]. PQ per fold: 0.437 / 0.443 / 0.443 / 0.445.
- On the 296 images with several annotators the model scores 0.443, while one annotator scored against another
  reaches 0.340.
- Overlapping prediction–annotation pairs: mean IoU 0.594 (76% above 0.5), mean Dice 0.726.
- Public leaderboard: 0.38.

Every number in the report is traceable to a file in [`results/`](results/).

## Method

Built on the structure of the public notebook
[*Solar Filament Segmentation 2026 (0.37)*](https://www.kaggle.com/code/tushhaaarrrr/solar-filament-segmentation-2026-0-37)
by tushhaaarrrr (annotator-mean soft target with a spine head, connected components as instances, a gradient-boosted
component scorer with the PQ/2 keep rule). Our changes and additions:

1. **Preprocessing** — Hough disk detection with limb refinement, division by the radial median profile (clipped to
   [0.75, 1.25]), light Gaussian blur and CLAHE; a single-channel input.
2. **Network** — U-Net with an ImageNet-pretrained ConvNeXt-tiny encoder (`convnext_tiny.fb_in22k_ft_in1k`, timm),
   LayerNorm on the encoder outputs, ASPP without global pooling, scSE with a local 15×15 channel gate and full/half
   resolution stem branches, so that training on 1024 px crops and inference on the full 2048 px disk behave the same.
3. **Training** — 1024 px crops of the native 2048 px images (half centred on filaments), BCE + Dice + clDice + spine
   BCE, AdamW with a one-cycle schedule, 40 epochs, 4 folds.
4. **Instances** — 4-flip TTA (test: mean of the 4 fold models), hysteresis 0.4 / 0.7, closing (r = 6), hole filling,
   minimum area 80 px; instances are pixel-disjoint by construction.
5. **PQ-aware selection** — a HistGradientBoosting regressor predicts each component's expected matched IoU *u*
   from 31 features; a component raises pooled PQ exactly when *u* > PQ/2. The scorer is calibrated (ECE 0.018),
   so this theoretical threshold (0.221) performs as well as the tuned one (0.225).

Every hyper-parameter is in [`configs/final.yaml`](configs/final.yaml).

## Repository

```
configs/final.yaml        every hyper-parameter (single source of truth)
filament/                 the method as a Python package
  preprocess.py           disk detection, limb flattening, CLAHE
  data.py                 annotations, multi-annotator targets, folds, cache, training crops
  model.py                ConvNeXt-tiny U-Net
  losses.py, train.py     losses and the resumable training loop
  infer.py                TTA / fold-ensemble prediction
  postprocess.py          components, features, scorer, selection
  evaluate.py             PQ, paired bootstrap
  official_eval.py        the organisers' self-evaluation functions (unchanged)
  pipeline.py             glue used by the notebooks
  submission.py           submission writing and validation
  analysis.py, temporal.py  analyses and the temporal-consistency experiment
notebooks/pipeline.ipynb  end to end: data -> preprocessing -> (training) -> OOF evaluation -> submission.csv
notebooks/analysis.ipynb  analyses of the OOF predictions behind Sections 3.2-3.4 of the report
experiments/              ablations and the temporal-consistency experiment (see experiments/README.md)
splits/folds.csv          the 4 folds (image level, stratified by station)
weights/                  convnext_fold{0..3}.pt (half precision), scorer.pkl
results/                  outputs of the runs reported in the paper
report/report.pdf         the 4-page report
```

## Reproduce

```bash
git clone https://github.com/EfeKaramese/ieee-bigdata2026-solar-filament-segmentation.git
cd ieee-bigdata2026-solar-filament-segmentation
pip install -r requirements.txt
```

Download the competition data, set `DATA_DIR` in the first code cell of `notebooks/pipeline.ipynb` (or the
environment variable `DATA_DIR`) and run the notebook from the `notebooks/` folder.

- `TRAIN = False` (default) loads the shipped weights, predicts the OOF and test sets, evaluates with the official
  functions and writes `submission.csv` (about 1.5 h on a GPU).
- `TRAIN = True` retrains everything from scratch (about 4.7 h on one NVIDIA A100; longer on smaller GPUs). Training
  is resumable: run the notebook again after an interruption.

**Re-running gives the same result up to GPU non-determinism.** The executed `notebooks/pipeline.ipynb` in this
repository is a clean-environment run with `TRAIN = False` on an NVIDIA L4 (the reported numbers come from the
original run on an A100). It reaches OOF PQ 0.443 (SQ 0.680, RQ 0.652) instead of 0.442. The probability maps differ
slightly between GPUs, the cross-fitted scorer is refitted on them and the OOF-tuned threshold moves from 0.225 to
0.263; because PQ is flat around its optimum, this changes which marginal components are kept but hardly changes PQ.
The test submission always uses the shipped scorer and threshold (`weights/scorer.pkl`, τ = 0.225).

Mixed precision is chosen automatically (bfloat16 on GPUs with compute capability ≥ 8, float16 otherwise). The pinned
versions in `requirements.txt` are those of the run that produced `weights/`; `scikit-learn` must match because
`weights/scorer.pkl` is a pickled model.

## Data use

- Networks and scorers are trained only on the competition's training split. Public MAGFiLO annotations and the test
  images are never used for training; inference uses only the test H-α images.
- The station letter in the file name is used only to stratify the folds, never as a model input.
- The temporal-consistency experiment (report Section 3.6) uses unlabeled GONG H-α bursts from 2023–2026, after the
  most recent competition image; it is not part of the submitted model.

## License and acknowledgements

Code: MIT (see [`LICENSE`](LICENSE)). The MAGFiLO data (CC BY-NC 4.0) and the pre-trained ImageNet weights are
subject to their own licenses and are not redistributed here.

This work utilizes GONG data obtained by the NSO Integrated Synoptic Program, managed by the National Solar
Observatory, which is operated by the Association of Universities for Research in Astronomy (AURA), Inc. under a
cooperative agreement with the National Science Foundation and with contribution from the National Oceanic and
Atmospheric Administration.

The author used Claude (Anthropic) extensively to write and debug code and analysis scripts and to draft and edit the
report. The author directed the work, ran all experiments and checked all reported numbers against the experiment
outputs.
