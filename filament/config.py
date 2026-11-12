"""Configuration: one YAML file plus optional dotted overrides, e.g. {"data.target": "union"}."""
import copy
from pathlib import Path

import yaml


def load_config(path, overrides=None):
    cfg = yaml.safe_load(Path(path).read_text())
    for key, value in (overrides or {}).items():
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"unknown config key: {key}")
        node[leaf] = value
    return cfg


def scaled(cfg, size):
    """Copy of the config with pixel quantities scaled to a smaller working size (used for quick dry runs)."""
    cfg = copy.deepcopy(cfg)
    s = size / cfg["data"]["full"]
    if s == 1:
        return cfg
    cfg["data"]["full"] = size
    cfg["data"]["input_res"] = max(16, int(round(cfg["data"]["input_res"] * s)))
    cfg["data"]["crop"] = max(16, int(round(cfg["data"]["crop"] * s)))
    cfg["data"]["spine_px"] = max(1, int(round(cfg["data"]["spine_px"] * s)))
    cfg["postprocess"]["close"] = max(1, int(round(cfg["postprocess"]["close"] * s)))
    cfg["postprocess"]["min_area"] = max(2, int(round(cfg["postprocess"]["min_area"] * s * s)))
    return cfg
