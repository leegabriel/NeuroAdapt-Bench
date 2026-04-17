import os
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModel
from transformers import dynamic_module_utils as transformers_dynamic_module_utils


@lru_cache(maxsize=None)
def load_position_bank(model_dir):
    hf_root = Path("/tmp/huggingface")
    hf_root.mkdir(parents=True, exist_ok=True)
    (hf_root / "modules").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_root)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_root / "transformers")
    os.environ["HF_MODULES_CACHE"] = str(hf_root / "modules")
    transformers_dynamic_module_utils.HF_MODULES_CACHE = str(hf_root / "modules")
    return AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
    )


def _lookup_single_position(model, name):
    with torch.no_grad():
        position = model([name]).float().cpu()
    if position.ndim != 2 or position.size(0) != 1 or position.size(1) != 3:
        raise ValueError(f"Unexpected position-bank output for {name}: {tuple(position.shape)}")
    return position[0]


def resolve_positions(electrodes, model_dir, aliases=None):
    aliases = aliases or {}
    model = load_position_bank(model_dir)
    positions = []

    for electrode in electrodes:
        resolved = aliases.get(electrode, electrode)
        if "-" in resolved:
            left, right = resolved.split("-", 1)
            left_position = _lookup_single_position(model, left)
            right_position = _lookup_single_position(model, right)
            positions.append((left_position + right_position) / 2.0)
        else:
            positions.append(_lookup_single_position(model, resolved))

    return torch.stack(positions, dim=0)


def expand_positions(positions, batch_size):
    return positions.unsqueeze(0).expand(batch_size, -1, -1)
