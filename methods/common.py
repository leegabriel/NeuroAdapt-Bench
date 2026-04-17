import torch


def logits_to_class_logits(logits):
    if logits.ndim == 1:
        return torch.stack((-0.5 * logits, 0.5 * logits), dim=1)
    if logits.ndim == 2 and logits.size(1) == 1:
        logits = logits.squeeze(1)
        return torch.stack((-0.5 * logits, 0.5 * logits), dim=1)
    if logits.ndim == 2 and logits.size(1) >= 2:
        return logits
    raise ValueError(f"Unsupported logits shape for class-logit conversion: {tuple(logits.shape)}")
