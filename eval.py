from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import deep_update, load_config, parse_cli_overrides
from src.data import OxfordFlowers102Dataset
from src.engine import evaluate
from src.metrics import save_confusion_heatmap, summarize_class_errors
from src.models import build_model
from src.utils import ensure_dir, save_json, set_seed
from train import create_transforms


def save_prediction_grid(metrics: dict, output_dir: Path, max_images: int = 16) -> None:
    samples = []
    for idx, (path, target, pred, probs) in enumerate(
        zip(metrics["image_paths"], metrics["targets"], metrics["preds"], metrics["probs"])
    ):
        confidence = float(probs[pred])
        samples.append(
            {
                "path": path,
                "target": int(target),
                "pred": int(pred),
                "confidence": confidence,
                "correct": int(target) == int(pred),
            }
        )
    samples.sort(key=lambda item: (item["correct"], -item["confidence"]))
    samples = samples[:max_images]
    if not samples:
        return

    plt.figure(figsize=(12, 12))
    for idx, sample in enumerate(samples, start=1):
        image = plt.imread(sample["path"])
        plt.subplot(4, 4, idx)
        plt.imshow(image)
        plt.axis("off")
        title = f"T:{sample['target']} P:{sample['pred']}\n{sample['confidence']:.2f}"
        plt.title(title, fontsize=8, color="green" if sample["correct"] else "red")
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_grid.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Oxford 102 Flowers classifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args, unknown = parser.parse_known_args()

    config = deep_update(load_config(args.config), parse_cli_overrides(unknown))
    set_seed(int(config.get("seed", 42)))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    _, eval_transform = create_transforms(config)
    print(f"[Config] split={args.split} device={device} checkpoint={args.checkpoint}")

    dataset = OxfordFlowers102Dataset(config.get("data_root", "data"), args.split, transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.0)))
    metrics = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        amp_enabled=bool(config.get("amp", True)) and device.type == "cuda",
    )

    checkpoint_path = Path(args.checkpoint)
    output_dir = ensure_dir(checkpoint_path.parent.parent / f"eval_{args.split}")
    summary = {
        "split": args.split,
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }
    save_json(summary, output_dir / "summary.json")
    save_json(summarize_class_errors(metrics["confusion_matrix"], top_k=20), output_dir / "top_confusions.json")
    save_confusion_heatmap(metrics["confusion_matrix"], output_dir / "confusion_matrix.png")
    save_prediction_grid(metrics, output_dir)

    print(f"[Eval] loss={summary['loss']:.4f} acc={summary['accuracy']:.4f} macro_f1={summary['macro_f1']:.4f}")
    print(f"[Eval] artifacts saved to {output_dir}")
    print(summary)


if __name__ == "__main__":
    main()
