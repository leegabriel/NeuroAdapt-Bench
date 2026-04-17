import torch
import torch.nn as nn

from methods.shot import (
    SHOT,
    collect_params,
    configure_model,
    freeze_module
)
from methods.tent import (
    Tent,
    collect_params as collect_tent_params,
    configure_model as configure_tent_model
)
from methods.t3a import T3A

ADAPTER_ONLY = False

def feature_tent(model, lr=1e-3, momentum=0.9):
    if ADAPTER_ONLY:
        # Adapter-only Tent: keep the encoder and classifier frozen, and
        # update only normalization parameters inside the feature adapter
        model.eval()
        model.requires_grad_(False)
        model.feature_adapter = configure_tent_model(model.feature_adapter)
        for module in model.feature_adapter.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        params, _ = collect_tent_params(model.feature_adapter)
    else:
        # Full-model Tent: update normalization parameters wherever they
        # appear in the model while keeping dropout disabled
        model = configure_tent_model(model)
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        params, _ = collect_tent_params(model)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
    return Tent(model, optimizer, steps=1, episodic=False)


def feature_shot(model, lr=1e-4, weight_decay=1e-4):
    model.eval()
    if ADAPTER_ONLY:
        # Adapter-only SHOT: keep the pretrained encoder and final classifier
        # fixed, and adapt only the task-facing feature adapter
        configure_model(model, trainable_modules=[model.feature_adapter])
    else:
        # Broader SHOT path: adapt encoder + feature adapter while keeping the
        # final classifier fixed
        configure_model(model, trainable_modules=[model.encoder, model.feature_adapter])
    freeze_module(model.classification_head)
    params, _ = collect_params(model)
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        weight_decay=weight_decay,
    )
    return SHOT(model, optimizer)


def feature_t3a(model, filter_k=20, episodic=False):
    # T3A adapts only its internal support/prototype state, not model weights,
    # so keeping the wrapped model frozen in eval mode 
    model.eval()
    freeze_module(model)
    return T3A(
        model,
        classifier=model.classification_head,
        filter_k=filter_k,
        episodic=episodic,
    )
