import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)


def extract_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logit"].detach()
    if torch.is_tensor(outputs):
        return outputs.detach()
    raise ValueError(f"Unsupported model output type: {type(outputs)}")


def predict_labels_from_logits(logits):
    if logits.ndim == 1:
        return (torch.sigmoid(logits) >= 0.5).long()
    if logits.ndim == 2 and logits.size(1) == 1:
        return (torch.sigmoid(logits.squeeze(1)) >= 0.5).long()
    return logits.argmax(dim=1)


def binary_positive_scores_from_logits(logits):
    if logits.ndim == 1:
        return torch.sigmoid(logits)
    if logits.ndim == 2 and logits.size(1) == 1:
        return torch.sigmoid(logits.squeeze(1))
    if logits.ndim == 2 and logits.size(1) == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    raise ValueError(
        "Binary evaluation expects logits shaped as (N,), (N, 1), or (N, 2)."
    )


def binary_metrics(y_true, y_pred, y_score):
    if len(set(y_true)) < 2:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "cohen_kappa": None,
            "weighted_f1": None,
        }
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "cohen_kappa": None,
        "weighted_f1": None,
    }


def multiclass_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": None,
        "pr_auc": None,
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }
