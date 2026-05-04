from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.metrics import confusion_matrix_from_arrays, macro_f1_from_confusion, per_class_accuracy
from src.utils import plot_training_curves, save_history_csv, save_json


def create_optimizer(config: dict[str, Any], parameter_groups):
    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameter_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameter_groups,
            momentum=float(config.get("momentum", 0.9)),
            nesterov=bool(config.get("nesterov", True)),
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(config: dict[str, Any], optimizer, steps_per_epoch: int):
    scheduler_name = str(config.get("scheduler", "cosine")).lower()
    epochs = int(config["epochs"])
    warmup_epochs = int(config.get("warmup_epochs", 0))
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(0, steps_per_epoch * warmup_epochs)

    if scheduler_name == "none":
        return None

    if scheduler_name == "cosine":
        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if scheduler_name == "step":
        step_size = int(config.get("step_size", 10))
        gamma = float(config.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def create_amp_scaler(device: torch.device, enabled: bool):
    if device.type != "cuda":
        return None
    return torch.amp.GradScaler("cuda", enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    amp_enabled: bool,
    scaler,
    epoch: int,
    total_epochs: int,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=f"Train {epoch}/{total_epochs}", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
            scheduler.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{total_correct / max(1, total_samples):.4f}",
        )

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device, amp_enabled: bool):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_paths: list[str] = []
    all_image_ids: list[int] = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        all_targets.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_paths.extend(batch["image_path"])
        all_image_ids.extend([int(v) for v in batch["image_id"]])

    targets = np.concatenate(all_targets, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    num_classes = probs.shape[1]
    matrix = confusion_matrix_from_arrays(targets, preds, num_classes=num_classes)
    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
        "macro_f1": macro_f1_from_confusion(matrix),
        "per_class_accuracy": per_class_accuracy(matrix),
        "targets": targets,
        "preds": preds,
        "probs": probs,
        "confusion_matrix": matrix,
        "image_paths": all_paths,
        "image_ids": all_image_ids,
    }


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    config: dict[str, Any],
    output_dir: str | Path,
):
    output_dir = Path(output_dir)
    history: list[dict[str, Any]] = []
    best_val_acc = -1.0
    best_epoch = -1
    best_checkpoint_path = output_dir / "checkpoints" / "best.pt"
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = create_amp_scaler(device=device, enabled=amp_enabled)

    total_epochs = int(config["epochs"])
    print(f"[Start] training {total_epochs} epochs")

    for epoch in range(1, total_epochs + 1):
        print(f"[Epoch {epoch}/{total_epochs}] train")
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            amp_enabled=amp_enabled,
            scaler=scaler,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        print(f"[Epoch {epoch}/{total_epochs}] validate")
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )

        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr_group_0": optimizer.param_groups[0]["lr"],
            "lr_group_1": optimizer.param_groups[-1]["lr"],
        }
        history.append(row)

        print(
            f"[Epoch {epoch}/{total_epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_val_accuracy": best_val_acc,
                },
                best_checkpoint_path,
            )
            save_json(
                {
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_acc,
                    "val_macro_f1": val_metrics["macro_f1"],
                },
                output_dir / "best_summary.json",
            )
            print(f"[Epoch {epoch}/{total_epochs}] new best checkpoint saved -> {best_checkpoint_path}")

    save_json(history, output_dir / "history.json")
    save_history_csv(history, output_dir / "history.csv")
    plot_training_curves(history, output_dir)
    print(f"[Done] best_epoch={best_epoch} best_val_acc={best_val_acc:.4f}")
    print(f"[Done] training artifacts saved to {output_dir}")
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "best_checkpoint_path": best_checkpoint_path,
    }
