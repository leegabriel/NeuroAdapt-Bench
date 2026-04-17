import argparse
import csv
import fcntl
import gc

import torch

from config import config
from data.torch_loader import create_dataloader
from tee_logging import setup_tee_logging
from bench_evaluation import evaluate_model
from tta_models.adaptation import (
    feature_shot,
    feature_t3a,
    feature_tent,
)
from tta_models.builders import build_model
from utils.runtime import seed_everything


def print_metrics(title, metrics):
    print(title)
    print(f"  accuracy:          {metrics['accuracy']:.4f}")
    print(f"  balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"  roc_auc:           {metrics['roc_auc']:.4f}")
        print(f"  pr_auc:            {metrics['pr_auc']:.4f}")
    if metrics["cohen_kappa"] is not None:
        print(f"  cohen_kappa:       {metrics['cohen_kappa']:.4f}")
        print(f"  weighted_f1:       {metrics['weighted_f1']:.4f}")
    if metrics["tta_loss"] is not None:
        print(f"  tta_loss:          {metrics['tta_loss']:.6f}")


def print_dataset_summary(data_name, dataset, dataloader):
    first_sample = dataset[0]
    label = first_sample["label"]
    if torch.is_tensor(label) and label.numel() == 1:
        label = label.item()
    print(f"{data_name} test dataset")
    print(f"  samples:     {len(dataset)}")
    print(f"  input shape: {tuple(first_sample['signal'].shape)}")
    print(f"  label:       {label}")
    print(f"  batches:     {len(dataloader)}")


def _no_tta(model, _dataloader, _device):
    # No-TTA keeps the loaded classifier fully frozen in eval mode for plain
    # inference, and the outer benchmark loop also runs the forward pass under
    # torch.no_grad()
    model.requires_grad_(False)
    model.eval()
    return model


# These helpers only build the method wrapper, and Tent/SHOT override the outer
# torch.no_grad() context inside their own gradient-enabled adaptation step,
# while T3A stays frozen and updates only its support/prototype state
def _tent(model, _dataloader, _device):
    return feature_tent(model)


def _t3a(model, _dataloader, _device):
    return feature_t3a(model)


def _shot(model, dataloader, device):
    shot_model = feature_shot(model)
    shot_model.refresh_centroids(dataloader, device)
    return shot_model


METHODS = {
    "no_tta": _no_tta,
    "Tent": _tent,
    "T3A": _t3a,
    "SHOT": _shot,
}


FIELDNAMES = [
    "experiment",
    "seed",
    "foundation_model",
    "dataset",
    "batch_size",
    "method",
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "cohen_kappa",
    "weighted_f1",
    "tta_loss",
    "source_log",
]


def write_results_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def merge_results_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.touch(exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as handle:
                existing_rows = list(csv.DictReader(handle))
        else:
            existing_rows = []

        replacement_key = {
            (
                row["experiment"],
                row["seed"],
                row["foundation_model"],
                row["dataset"],
                row.get("batch_size", ""),
            )
            for row in rows
        }
        merged_rows = [
            row
            for row in existing_rows
            if (
                row["experiment"],
                row["seed"],
                row["foundation_model"],
                row["dataset"],
                row.get("batch_size", ""),
            )
            not in replacement_key
        ]
        merged_rows.extend(rows)
        merged_rows.sort(
            key=lambda row: (
                row["experiment"],
                int(row["seed"]),
                row["foundation_model"],
                row["dataset"],
                int(row.get("batch_size") or -1),
                row["method"],
            )
        )
        write_results_csv(path, merged_rows)


def benchmark_log_path(experiment, seed, model_name, data_name, batch_size):
    seed_dir = config.experiments.DIRS[experiment] / f"seed_{seed}"
    return seed_dir / "logs" / f"{model_name}__{data_name}__bs{batch_size}.log"


def benchmark_fragment_path(experiment, seed, model_name, data_name, batch_size):
    seed_dir = config.experiments.DIRS[experiment] / f"seed_{seed}"
    return seed_dir / "benchmark" / "results" / f"{model_name}__{data_name}__bs{batch_size}.csv"


def evaluate_model_family(experiment, model_name, data_name, device, seed, source_log, batch_size):
    checkpoint = (
        config.experiments.DIRS[experiment]
        / f"seed_{seed}"
        / "classifier"
        / "checkpoints"
        / f"{model_name}__{data_name}__best_model.pth"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    test_loader = create_dataloader(
        data_name=data_name,
        train_val_test="test",
        seed=seed,
        batch_size=batch_size,
    )
    print_dataset_summary(data_name, test_loader.dataset, test_loader)

    rows = []
    for method_name, method_fn in METHODS.items():
        model = build_model(
            experiment,
            model_name,
            data_name,
            device=device,
            checkpoint_path=checkpoint,
        )
        adapted_model = method_fn(model, test_loader, device)
        metrics = evaluate_model(adapted_model, test_loader, device, data_name)
        print_metrics(f"{model_name} {data_name} {experiment} {method_name}", metrics)
        rows.append(
            {
                "experiment": experiment,
                "seed": seed,
                "foundation_model": model_name,
                "dataset": data_name,
                "batch_size": batch_size,
                "method": method_name,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "cohen_kappa": metrics["cohen_kappa"],
                "weighted_f1": metrics["weighted_f1"],
                "tta_loss": metrics["tta_loss"],
                "source_log": str(source_log),
            }
        )
        # Free GPU memory between methods to avoid accumulation across model families
        del adapted_model
        del model
        torch.cuda.empty_cache()
        gc.collect()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run TTA experiments for a single registered dataset/model pair.")
    parser.add_argument("--experiment", choices=config.experiments.NAMES, required=True)
    parser.add_argument("--seed", type=int, choices=config.training.SEEDS, default=config.training.DEFAULT_SEED)
    parser.add_argument("--model", choices=config.models.NAMES, required=True)
    parser.add_argument("--dataset", choices=config.datasets.NAMES, required=True)
    parser.add_argument("--batch-size", type=int, default=config.training.BATCH_SIZE)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_dir = config.experiments.DIRS[args.experiment] / f"seed_{args.seed}"
    results_path = seed_dir / "results.csv"
    fragment_path = benchmark_fragment_path(
        args.experiment,
        args.seed,
        args.model,
        args.dataset,
        args.batch_size,
    )
    log_path = setup_tee_logging(
        benchmark_log_path(
            args.experiment,
            args.seed,
            args.model,
            args.dataset,
            args.batch_size,
        )
    )

    print(f"Logging benchmark activity to {log_path}")
    print(f"experiment: {args.experiment}")
    print(f"seed: {args.seed}")
    print(f"device: {device}")
    print(f"batch_size: {args.batch_size}")
    print(f"model: {args.model}")
    print(f"dataset: {args.dataset}")

    rows = []
    print(f"\n=== {args.model} {args.dataset} {args.experiment} ===")
    try:
        rows.extend(
            evaluate_model_family(
                args.experiment,
                args.model,
                args.dataset,
                device,
                args.seed,
                source_log=log_path,
                batch_size=args.batch_size,
            )
        )
    except FileNotFoundError as error:
        print(f"SKIP {args.model} {args.dataset}: {error}")

    write_results_csv(fragment_path, rows)
    print(f"Wrote benchmark fragment to {fragment_path}")
    if rows:
        merge_results_csv(results_path, rows)
        print(f"Updated aggregate results at {results_path}")


if __name__ == "__main__":
    main()
