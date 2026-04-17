import torch
import torch.nn.functional as F

from bench_evaluation import evaluate_model
from utils.runtime import (
    forward_kwargs_from_batch,
    labels_from_batch,
    move_batch_to_device,
)


def classification_loss(logits, labels):
    if labels.ndim > 1:
        labels = labels.squeeze(-1)
    if logits.ndim == 1 or (logits.ndim == 2 and logits.size(1) == 1):
        return F.binary_cross_entropy_with_logits(logits.reshape(-1), labels.float())
    return F.cross_entropy(logits, labels.long())


def set_training_mode(model):
    model.train()
    # Classifier fine-tuning in tta-eeg-bench always keeps the shared encoder
    # frozen and trains only the dataset-specific task head
    model.freeze_encoder()
    model.feature_adapter.train()
    model.classification_head.train()


def task_head_parameters(model):
    # In this benchmark, the task head is the dataset-specific adapter plus the
    # final classifier layer on top of the shared encoder
    return list(model.feature_adapter.parameters()) + list(model.classification_head.parameters())


def train_one_epoch(model, dataloader, optimizer, device):
    set_training_mode(model)
    total_loss = 0.0
    total_examples = 0

    for raw_batch in dataloader:
        batch = move_batch_to_device(raw_batch, device)
        labels = labels_from_batch(batch)
        if labels.ndim > 1:
            labels = labels.squeeze(-1)

        outputs = model(**forward_kwargs_from_batch(batch, model))
        loss = classification_loss(outputs["logit"], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate_loss(model, dataloader, device):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_examples = 0

    for raw_batch in dataloader:
        batch = move_batch_to_device(raw_batch, device)
        labels = labels_from_batch(batch)
        if labels.ndim > 1:
            labels = labels.squeeze(-1)
        outputs = model(**forward_kwargs_from_batch(batch, model))
        loss = classification_loss(outputs["logit"], labels)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    if was_training:
        model.train()
    return total_loss / max(total_examples, 1)


def evaluate_metrics(model, dataloader, device, data_name):
    """Run metric evaluation in eval mode, then restore training mode if needed."""
    was_training = model.training
    model.eval()
    metrics = evaluate_model(model, dataloader, device, data_name=data_name)
    if was_training:
        model.train()
    return metrics
