"""Reusable T3A wrapper adapted from the official T3A implementation.

Upstream references:
- paper: Test-Time Classifier Adjustment Module for Model-Agnostic Domain
  Generalization (NeurIPS 2021)
- repo: https://github.com/matsuolab/Domainbed_contrib
- source adapted here: domainbed/adapt_algorithms.py, class ``T3A``

What is preserved from the upstream implementation:
- initialize warm supports from ``classifier.weight``
- compute warmup pseudo-labels and entropy from classifier outputs
- append target features, pseudo-labels, and entropy online
- retain the lowest-entropy supports per class
- rebuild classifier weights from normalized supports

What is changed here:
- the DomainBed ``Algorithm`` wrapper is removed
- this version expects a local adapter that returns ``{\"features\", \"logit\"}``
- forward always performs the online T3A update used for evaluation in this repo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.common import logits_to_class_logits


class T3A(nn.Module):
    """Generic T3A wrapper for models that expose penultimate features and logits."""

    def __init__(self, model, classifier, filter_k=20, episodic=False):
        super().__init__()
        classifier = t3a_compatible_classifier(classifier)

        self.model = model
        self.filter_k = filter_k
        self.episodic = episodic
        self.num_classes = classifier.out_features

        self.model.requires_grad_(False)
        self.model.eval()

        warmup_supports = classifier.weight.detach().clone()
        warmup_logits = classifier(warmup_supports)
        warmup_labels = F.one_hot(
            warmup_logits.argmax(dim=1),
            num_classes=self.num_classes,
        ).float()
        warmup_ent = softmax_entropy(warmup_logits)

        self.register_buffer("warmup_supports", warmup_supports)
        self.register_buffer("warmup_labels", warmup_labels)
        self.register_buffer("warmup_ent", warmup_ent)

        self.supports = warmup_supports.clone()
        self.labels = warmup_labels.clone()
        self.ent = warmup_ent.clone()

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        if self.episodic:
            self.reset()

        outputs = self.model(*args, **kwargs)
        features = flatten_features(extract_features(outputs))
        logits = logits_to_class_logits(extract_logits(outputs))

        pseudo_labels = F.one_hot(
            logits.argmax(dim=1),
            num_classes=self.num_classes,
        ).float()
        entropies = softmax_entropy(logits)

        self.supports = self.supports.to(features.device)
        self.labels = self.labels.to(features.device)
        self.ent = self.ent.to(features.device)

        self.supports = torch.cat([self.supports, features.detach()], dim=0)
        self.labels = torch.cat([self.labels, pseudo_labels.detach()], dim=0)
        self.ent = torch.cat([self.ent, entropies.detach()], dim=0)

        supports, labels, entropies = select_supports(
            self.supports,
            self.labels,
            self.ent,
            self.num_classes,
            self.filter_k,
        )
        self.supports = supports
        self.labels = labels
        self.ent = entropies

        normalized_supports = F.normalize(self.supports, dim=1)
        weights = normalized_supports.t() @ self.labels
        adapted_logits = features @ F.normalize(weights, dim=0)

        result = {"logit": adapted_logits.detach()}
        if isinstance(outputs, dict) and outputs.get("features") is not None:
            result["features"] = outputs["features"].detach()
        return result

    def reset(self):
        self.supports = self.warmup_supports.clone()
        self.labels = self.warmup_labels.clone()
        self.ent = self.warmup_ent.clone()


def extract_features(outputs):
    if not isinstance(outputs, dict) or "features" not in outputs:
        raise ValueError("T3A model outputs must contain a 'features' tensor")
    return outputs["features"]


def extract_logits(outputs):
    if not isinstance(outputs, dict) or "logit" not in outputs:
        raise ValueError("T3A model outputs must contain a 'logit' tensor")
    return outputs["logit"]


def flatten_features(features):
    return features.flatten(1)


def t3a_compatible_classifier(classifier):
    if not isinstance(classifier, torch.nn.Linear):
        raise TypeError(f"T3A expects a torch.nn.Linear classifier head, got {type(classifier)}")

    if classifier.out_features != 1:
        return classifier

    virtual_classifier = torch.nn.Linear(
        classifier.in_features,
        2,
        bias=classifier.bias is not None,
    ).to(device=classifier.weight.device, dtype=classifier.weight.dtype)

    with torch.no_grad():
        weight = classifier.weight.detach()
        virtual_classifier.weight[0].copy_(-0.5 * weight[0])
        virtual_classifier.weight[1].copy_(0.5 * weight[0])
        if classifier.bias is not None:
            bias = classifier.bias.detach()
            virtual_classifier.bias[0].copy_(-0.5 * bias[0])
            virtual_classifier.bias[1].copy_(0.5 * bias[0])

    virtual_classifier.requires_grad_(False)
    virtual_classifier.eval()
    return virtual_classifier


def select_supports(supports, labels, entropies, num_classes, filter_k):
    if filter_k == -1:
        return supports, labels, entropies

    predicted_classes = labels.argmax(dim=1).long()
    kept_indices = []
    all_indices = torch.arange(entropies.size(0), device=entropies.device)

    for class_idx in range(num_classes):
        class_mask = predicted_classes == class_idx
        class_indices = all_indices[class_mask]
        if class_indices.numel() == 0:
            continue
        class_entropies = entropies[class_mask]
        order = torch.argsort(class_entropies)
        kept_indices.append(class_indices[order][:filter_k])

    if kept_indices:
        kept_indices = torch.cat(kept_indices, dim=0)
    else:
        kept_indices = all_indices[:0]

    return supports[kept_indices], labels[kept_indices], entropies[kept_indices]


@torch.jit.script
def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
