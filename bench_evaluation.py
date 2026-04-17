import torch

from config import config
from utils.evaluation import (
    binary_metrics,
    binary_positive_scores_from_logits,
    extract_logits,
    multiclass_metrics,
    predict_labels_from_logits,
)
from utils.runtime import forward_kwargs_from_batch, labels_from_batch, move_batch_to_device


def evaluate_model(model, dataloader, device, data_name):
    data_config = getattr(config.datasets, data_name.upper())
    all_y_true = []
    all_y_pred = []
    all_y_score = []
    losses = []

    for raw_batch in dataloader:
        batch = move_batch_to_device(raw_batch, device)
        # Benchmark evaluation defaults to inference-only, while gradient-based TTA
        # wrappers such as Tent/SHOT re-enable gradients inside their own
        # adaptation step when needed
        with torch.no_grad():
            outputs = model(**forward_kwargs_from_batch(batch, model))

        logits = extract_logits(outputs)
        # Read held-out ground-truth labels only after the model/TTA step so
        # they are used for metrics, not adaptation
        labels = labels_from_batch(batch)
        if labels.ndim > 1:
            labels = labels.squeeze(-1)

        all_y_true.extend(labels.detach().cpu().tolist())
        all_y_pred.extend(predict_labels_from_logits(logits).cpu().tolist())

        if data_config["task"] == "binary":
            all_y_score.extend(binary_positive_scores_from_logits(logits).detach().cpu().tolist())

        if isinstance(outputs, dict) and outputs.get("tta_loss") is not None:
            losses.append(float(outputs["tta_loss"].item()))

    if data_config["task"] == "binary":
        metrics = binary_metrics(all_y_true, all_y_pred, all_y_score)
    else:
        metrics = multiclass_metrics(all_y_true, all_y_pred)

    metrics["tta_loss"] = sum(losses) / len(losses) if losses else None
    return metrics
