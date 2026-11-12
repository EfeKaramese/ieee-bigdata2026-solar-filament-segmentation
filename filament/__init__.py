"""Solar filament instance segmentation for H-alpha full-disk images (IEEE BigData Cup 2026, Challenge 02).

Modules
-------
config       YAML configuration with dotted overrides
preprocess   disk detection, radial limb flattening, CLAHE
data         annotations, multi-annotator targets, folds, preprocessing cache, training dataset
model        ConvNeXt-tiny U-Net (encoder-output LayerNorm, ASPP, local scSE, full/half-resolution stem branches)
losses       BCE + Dice + clDice + spine auxiliary loss
train        training loop (bf16/fp16, non-finite guard, resumable)
infer        test-time augmentation, fold averaging
postprocess  hysteresis components, 31 component features, PQ-aware component scorer
evaluate     per-annotator-record PQ, paired bootstrap
official_eval  organisers' self-evaluation functions (unchanged)
submission   RLE encoding and submission checks
pipeline     high-level steps used by the notebooks
"""
__version__ = "1.0.0"
