from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import load_config, parse_cli_overrides, deep_update
from src.data import OxfordFlowers102Dataset, get_split_sizes
from src.engine import create_optimizer, create_scheduler, evaluate, fit
from src.metrics import save_confusion_heatmap, summarize_class_errors
from src.models import build_model, make_parameter_groups
from src.utils import ensure_dir, now_timestamp, save_json, set_seed


def create_transforms(config: dict):
    from torchvision import transforms

    image_size = int(config.get("image_size", 224))
    normalize = transforms.Normalize(
        mean=config.get("normalize_mean", [0.485, 0.456, 0.406]),
        std=config.get("normalize_std", [0.229, 0.224, 0.225]),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def main():
    parser = argparse.ArgumentParser(description="Train Oxford 102 Flowers classifier.")
    parser.add_argument("--config", required=True)
    args, unknown = parser.parse_known_args()

    config = load_config(args.config)
    overrides = parse_cli_overrides(unknown)
    config = deep_update(config, overrides)

    experiment_name = config.get("experiment_name") or f"{config['model_name']}_{now_timestamp()}"
    config["experiment_name"] = experiment_name

    output_dir = ensure_dir(Path(config.get("output_root", "outputs")) / experiment_name)
    ensure_dir(output_dir / "checkpoints")
    save_json(config, output_dir / "resolved_config.json")

    set_seed(int(config.get("seed", 42)))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA but torch.cuda.is_available() is False.")

    train_transform, eval_transform = create_transforms(config)
    data_root = Path(config.get("data_root", "data"))
    split_sizes = get_split_sizes(data_root)

    train_dataset = OxfordFlowers102Dataset(data_root, "train", transform=train_transform)
    val_dataset = OxfordFlowers102Dataset(data_root, "val", transform=eval_transform)
    test_dataset = OxfordFlowers102Dataset(data_root, "test", transform=eval_transform)

    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 4))
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(config).to(device)
    parameter_groups = make_parameter_groups(model, config)
    optimizer = create_optimizer(config, parameter_groups)
    scheduler = create_scheduler(config, optimizer, steps_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.0)))

    print(f"[Config] experiment={experiment_name}")
    print(f"[Config] device={device} batch_size={batch_size} epochs={config['epochs']}")
    print(f"[Config] data_root={data_root}")
    print(f"[Data] train={split_sizes['train']} val={split_sizes['val']} test={split_sizes['test']}")

    save_json(
        {
            "device": str(device),
            "split_sizes": split_sizes,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        output_dir / "run_info.json",
    )

    training_result = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        output_dir=output_dir,
    )

    checkpoint = torch.load(training_result["best_checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Test] evaluating best checkpoint from epoch {training_result['best_epoch']}")
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        amp_enabled=bool(config.get("amp", True)) and device.type == "cuda",
    )
    test_summary = {
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "best_epoch": training_result["best_epoch"],
        "best_val_accuracy": training_result["best_val_accuracy"],
    }
    save_json(test_summary, output_dir / "test_summary.json")
    save_json(
        summarize_class_errors(test_metrics["confusion_matrix"], top_k=20),
        output_dir / "top_confusions.json",
    )
    save_json(
        {
            "per_class_accuracy": test_metrics["per_class_accuracy"],
        },
        output_dir / "per_class_metrics.json",
    )
    save_confusion_heatmap(
        test_metrics["confusion_matrix"],
        output_dir / "confusion_matrix.png",
        title=f"Test Confusion Matrix - {experiment_name}",
    )

    print(
        f"[Test] loss={test_summary['test_loss']:.4f} "
        f"acc={test_summary['test_accuracy']:.4f} "
        f"macro_f1={test_summary['test_macro_f1']:.4f}"
    )
    print(f"Training complete. Results saved to: {output_dir}")
    print(test_summary)


if __name__ == "__main__":
    main()
