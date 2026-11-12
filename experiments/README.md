# Experiments

Notebooks in this folder use the same `filament` package as `notebooks/pipeline.ipynb`. Results are in `results/`.
The temporal notebooks also need the unlabeled GONG bursts (burst tables and frames), which are not part of this repository.

| Notebook | Report | What it measures | Hardware |
|---|---|---|---|
| `ablation_fold0.ipynb` | Table 3, upper block | Fold 0 re-trained with the final code, one setting changed at a time: union target, whole disk at 1024 px, no clDice, +20 epochs (paired bootstrap against the reference) | Kaggle T4, float16 |
| `temporal_stability.ipynb` | Section 3.6 | Self-PQ of the final models on 80 held-out GONG bursts, compared with the earlier ResNet-34 models (their checkpoints are not shipped) | Colab |
| `temporal_distillation_convnext.ipynb` | Section 3.6 | Burst-target + consistency student vs. single-frame control vs. final model (stability and fold-0 accuracy) | Colab A100 |

The scorer ablations (Table 3, middle block), closing radius and the analyses of Sections 3.2–3.4 are produced by
`notebooks/analysis.ipynb` from the OOF predictions of the final run.

## Earlier experiments (marked † in Table 3)

These were run with earlier versions of the code during development (fold 0, paired bootstrap against the
corresponding baseline of that version) and are reported for completeness; their notebooks are not part of this
repository.

| Change | ΔPQ [95% CI] |
|---|---|
| ConvNeXt-tiny vs. ResNet-34 encoder | +0.013 [+0.002, +0.023] |
| DINOv3 ConvNeXt-B encoder | −0.023 [−0.033, −0.012] |
| Self-attention bottleneck | −0.008 [−0.016, +0.001] |
| Skeleton Recall loss | −0.040 |
| Three input channels (raw, flattened, enhanced) | +0.001 |
| Closing along the local filament axis (L = 21 / 41) | −0.008 / −0.029 |
| + ResNet-34 ensemble | +0.002 [−0.002, +0.006] |
| Size-routed ResNet / ConvNeXt | +0.000 [−0.002, +0.003] |
