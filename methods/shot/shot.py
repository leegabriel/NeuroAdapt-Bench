from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.common import logits_to_class_logits
from utils.runtime import forward_kwargs_from_batch, move_batch_to_device


class SHOT(nn.Module):
    """Generic SHOT wrapper for models that expose features and logits."""

    def __init__(
        self,
        model,
        optimizer,
        steps=1,
        episodic=False,
        mi_lambda=1.0,
        pl_weight=1.0,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.mi_lambda = mi_lambda
        self.pl_weight = pl_weight
        self.centroids = None

        assert steps > 0, "SHOT requires >= 1 step(s) to forward and update"

        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, *args, **kwargs):
        if self.centroids is None:
            raise RuntimeError("Call refresh_centroids() before SHOT adaptation.")
        if self.episodic:
            self.reset()

        outputs = None
        for _ in range(self.steps):
            outputs = forward_and_adapt(
                self.model,
                self.optimizer,
                self.centroids,
                mi_lambda=self.mi_lambda,
                pl_weight=self.pl_weight,
                *args,
                **kwargs,
            )
        return outputs

    @torch.no_grad()
    def refresh_centroids(self, dataloader, device):
        all_features = []
        all_probs = []

        for raw_batch in dataloader:
            batch = move_batch_to_device(raw_batch, device)
            outputs = self.model(**forward_kwargs_from_batch(batch, self.model))
            features = flatten_features(extract_features(outputs))
            logits = logits_to_class_logits(extract_centroid_logits(outputs))
            all_features.append(features.detach().cpu())
            all_probs.append(torch.softmax(logits, dim=1).detach().cpu())

        features = torch.cat(all_features, dim=0)
        probs = torch.cat(all_probs, dim=0)
        self.centroids = build_refined_centroids(features, probs).to(device)
        return self.centroids

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(
            self.model,
            self.optimizer,
            self.model_state,
            self.optimizer_state,
        )

def extract_features(outputs):
    if not isinstance(outputs, dict) or "features" not in outputs:
        raise ValueError("SHOT model outputs must contain a 'features' tensor")
    return outputs["features"]


def extract_eval_logits(outputs):
    if not isinstance(outputs, dict) or "logit" not in outputs:
        raise ValueError("SHOT model outputs must contain a 'logit' tensor")
    return outputs["logit"]


def extract_adapt_logits(outputs):
    if not isinstance(outputs, dict):
        raise ValueError("SHOT expects dict outputs from the wrapped model")
    return outputs.get("adapt_logit", outputs["logit"])


def extract_centroid_logits(outputs):
    if not isinstance(outputs, dict):
        raise ValueError("SHOT expects dict outputs from the wrapped model")
    return outputs.get("centroid_logit", outputs["logit"])


def flatten_features(features):
    return features.flatten(1)


def freeze_module(module):
    module.requires_grad_(False)
    module.eval()
    return module


def configure_model(model, trainable_modules):
    """Freeze the full model and re-enable gradients only for selected modules."""
    model.requires_grad_(False)
    for module in trainable_modules:
        for param in module.parameters():
            if param.is_floating_point() or param.is_complex():
                param.requires_grad_(True)
    return model


def collect_params(model):
    """Collect trainable parameters and their names."""
    params = []
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(param)
            names.append(name)
    return params, names


def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def conditional_entropy(probs, eps=1e-8):
    return -(probs * torch.log(probs + eps)).sum(dim=1).mean()


def marginal_entropy(probs, eps=1e-8):
    mean_probs = probs.mean(dim=0)
    return -(mean_probs * torch.log(mean_probs + eps)).sum()


def predict_centroid_labels(features, centroids):
    normalized_features = F.normalize(features, dim=1)
    normalized_centroids = F.normalize(centroids, dim=1)
    return (normalized_features @ normalized_centroids.transpose(0, 1)).argmax(dim=1)


def build_refined_centroids(features, probs):
    centroids = (probs.transpose(0, 1) @ features) / (probs.sum(dim=0).unsqueeze(1) + 1e-6)
    pseudo_labels = predict_centroid_labels(features, centroids)

    refined = torch.zeros_like(centroids)
    refined.index_add_(0, pseudo_labels, features)
    counts = torch.bincount(pseudo_labels, minlength=probs.size(1)).float().unsqueeze(1)
    return refined / counts.clamp_min(1e-6)


@torch.enable_grad()
def forward_and_adapt(model, optimizer, centroids, mi_lambda, pl_weight, *args, **kwargs):
    outputs = model(*args, **kwargs)
    features = flatten_features(extract_features(outputs))
    adapt_logits = logits_to_class_logits(extract_adapt_logits(outputs))
    probs = torch.softmax(adapt_logits, dim=1)

    with torch.no_grad():
        pseudo_labels = predict_centroid_labels(features, centroids)

    mi_loss = conditional_entropy(probs) - mi_lambda * marginal_entropy(probs)
    pl_loss = F.cross_entropy(adapt_logits, pseudo_labels)
    total_loss = mi_loss + pl_weight * pl_loss

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()

    return {
        "features": extract_features(outputs).detach(),
        "logit": extract_eval_logits(outputs).detach(),
        "adapt_logit": adapt_logits.detach(),
        "tta_loss": total_loss.detach(),
    }
