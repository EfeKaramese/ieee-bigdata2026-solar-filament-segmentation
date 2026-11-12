# Weights

Produced by `notebooks/pipeline.ipynb` with `TRAIN = True` (Google Colab, NVIDIA A100, bfloat16).

| File | Content |
|---|---|
| `convnext_fold{0..3}.pt` | ConvNeXt-tiny U-Net of each fold, half-precision state dict (~73 MB each), loaded in float32 by `filament.model.load_model` |
| `scorer.pkl` | component scorer (HistGradientBoostingRegressor) and keep threshold; requires scikit-learn 1.6.1 |
| `demet_tc_fold0.pt` | optional: burst-distilled student of report Section 3.6 (fold 0), not used for the submission |
