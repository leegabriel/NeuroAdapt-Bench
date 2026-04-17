import os
import random

import numpy as np
import torch

from eeg_encoders.positions import expand_positions
from eeg_encoders.reve_positions import data_positions


def seed_everything(seed=5):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _unwrap_model(model):
    current = model
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "model", None)


def _model_data_name(model):
    for current in _unwrap_model(model):
        data_name = getattr(current, "data_name", None)
        if data_name is not None:
            return data_name
    return None


def _model_expects_pos(model):
    for current in _unwrap_model(model):
        if getattr(current, "expects_pos", False):
            return True
        encoder = getattr(current, "encoder", None)
        if getattr(encoder, "expects_pos", False):
            return True
    return False


def forward_kwargs_from_batch(batch, model):
    # Test-time adaptation only sees model inputs, so hold out ground-truth labels
    # here and use them later only for evaluation metrics/reporting
    kwargs = {key: value for key, value in batch.items() if key not in {"label", "pos"}}
    if _model_expects_pos(model):
        # REVE-style encoders need a positional tensor built from the dataset's
        # electrode layout, synthesize and batch-expand it here at runtime
        data_name = _model_data_name(model)
        if data_name is None:
            raise ValueError("Model requires positions, but no data_name is attached to the model.")
        positions = data_positions(data_name).to(kwargs["signal"].device)
        kwargs["pos"] = expand_positions(positions, kwargs["signal"].shape[0])
    return kwargs


def labels_from_batch(batch):
    return batch["label"]
