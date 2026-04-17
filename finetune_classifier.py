import argparse
import json
from pathlib import Path

import torch

from config import config
from data.torch_loader import create_dataloader
from tee_logging import setup_tee_logging
from utils.runtime import seed_everything
from utils.training import (
    evaluate_loss,
    evaluate_metrics,
    task_head_parameters,
    train_one_epoch,
)
from tta_models.builders import build_model

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune EEG task heads across all registered datasets.")
    parser.add_argument("--encoder", choices=config.models.NAMES, required=True)
    parser.add_argument("--dataset", choices=config.datasets.NAMES, required=True)
    parser.add_argument("--experiment", choices=config.experiments.NAMES, default="common")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=config.training.BATCH_SIZE)
    parser.add_argument("--seed", type=int, choices=config.training.SEEDS, default=config.training.DEFAULT_SEED)
    return parser.parse_args()


def build_dataloaders(data_name, seed, batch_size):
    return {
        split: create_dataloader(
            data_name=data_name,
            train_val_test=split,
            seed=seed,
            batch_size=batch_size,
        )
        for split in ("train", "val", "test")
    }


def classifier_paths(args):
    classifier_dir = config.experiments.DIRS[args.experiment] / f"seed_{args.seed}" / "classifier"
    artifact_stem = f"{args.encoder}__{args.dataset}"
    checkpoints_dir = classifier_dir / "checkpoints"
    return {
        "log_path": classifier_dir / "logs" / f"{artifact_stem}.log",
        "best_model_path": checkpoints_dir / f"{artifact_stem}__best_model.pth",
        "last_model_path": checkpoints_dir / f"{artifact_stem}__last_model.pth",
        "metrics_path": checkpoints_dir / f"{artifact_stem}__metrics.json",
    }


def save_summary(metrics_path, summary):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = classifier_paths(args)
    log_path = setup_tee_logging(paths["log_path"])

    print(f"Logging to {log_path}")
    print(
        "Selected device: "
        f"{device} "
        f"(cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()})"
    )
    print(f"experiment: {args.experiment}")
    print(f"seed: {args.seed}")
    print(f"encoder: {args.encoder}")
    print(f"dataset: {args.dataset}")
    print(f"batch_size: {args.batch_size}")

    model = build_model(
        args.experiment,
        args.encoder,
        args.dataset,
        device=device,
        checkpoint_path=None,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
    )

    model.freeze_encoder()
    params = task_head_parameters(model)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    Path(paths["best_model_path"]).parent.mkdir(parents=True, exist_ok=True)

    best_score = float("-inf")
    history = []

    dataloaders = build_dataloaders(args.dataset, args.seed, args.batch_size)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]
    print(
        f"{args.dataset} split sizes: "
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, device)
        # Validation metrics run in eval mode and then restore training mode so
        # the next epoch continues with the task head in train mode
        val_metrics = evaluate_metrics(model, val_loader, device, args.dataset)
        dataset_config = getattr(config.datasets, args.dataset.upper())
        score = (
            val_metrics["roc_auc"]
            if dataset_config["task"] == "binary"
            else val_metrics["cohen_kappa"]
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            }
        )

        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"score={(score if score is not None else 0.0):.4f}"
        )

        torch.save(model.state_dict(), paths["last_model_path"])
        if score is not None and score > best_score:
            best_score = score
            torch.save(model.state_dict(), paths["best_model_path"])

    model.load_state_dict(torch.load(paths["best_model_path"], map_location=device))
    val_metrics = evaluate_metrics(model, val_loader, device, args.dataset)
    eval_metrics = evaluate_metrics(model, test_loader, device, args.dataset)

    summary = {
        "encoder": args.encoder,
        "dataset": args.dataset,
        "experiment": args.experiment,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_score": best_score,
        "val_metrics": val_metrics,
        "eval_metrics": eval_metrics,
        "history": history,
        "checkpoint": str(paths["best_model_path"]),
    }

    save_summary(paths["metrics_path"], summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
