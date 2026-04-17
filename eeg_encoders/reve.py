import os
from pathlib import Path

import torch.nn as nn
from transformers import AutoModel
from transformers import dynamic_module_utils as transformers_dynamic_module_utils


def _configure_hf_cache():
    hf_root = Path("/tmp/huggingface")
    hf_root.mkdir(parents=True, exist_ok=True)
    (hf_root / "modules").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_root)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_root / "transformers")
    os.environ["HF_MODULES_CACHE"] = str(hf_root / "modules")
    transformers_dynamic_module_utils.HF_MODULES_CACHE = str(hf_root / "modules")


class REVEBackboneEncoder(nn.Module):
    def __init__(self, model_dir, device):
        super().__init__()
        _configure_hf_cache()
        self.model = AutoModel.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.output_dim = self.model.embed_dim
        self.expects_pos = True

    def forward(self, signal, pos=None):
        if pos is None:
            raise ValueError("REVEBackboneEncoder requires batch positions via `pos`.")
        return self.model(signal, pos)


class REVEEncoder(nn.Module):
    def __init__(self, model_dir, device, pooled=True):
        super().__init__()
        self.backbone = REVEBackboneEncoder(model_dir=model_dir, device=device)
        self.output_dim = self.backbone.output_dim
        self.pooled = pooled
        self.expects_pos = True

    def forward(self, signal, pos=None):
        encoded = self.backbone(signal, pos)
        if self.pooled:
            return self.backbone.model.attention_pooling(encoded)
        return encoded
