from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.utils import ensure_dir, load_json


def collect_experiment_rows(experiment_dirs: list[Path]) -> list[dict]:
    rows = []
    for exp_dir in experiment_dirs:
        config_path = exp_dir / "resolved_config.json"
        history_path = exp_dir / "history.json"
        test_path = exp_dir / "test_summary.json"
        if not (config_path.exists() and history_path.exists() and test_path.exists()):
            continue
        config = load_json(config_path)
        history = load_json(history_path)
        test_summary = load_json(test_path)
        best_val = max((row["val_accuracy"] for row in history), default=None)
        rows.append(
            {
                "experiment_name": config.get("experiment_name", exp_dir.name),
                "model_name": config.get("model_name"),
                "pretrained": config.get("pretrained"),
                "attention_type": config.get("attention_type"),
                "epochs": config.get("epochs"),
                "batch_size": config.get("batch_size"),
                "backbone_lr": config.get("backbone_lr"),
                "head_lr": config.get("head_lr"),
                "optimizer": config.get("optimizer"),
                "best_val_accuracy": best_val,
                "test_accuracy": test_summary.get("test_accuracy"),
                "test_macro_f1": test_summary.get("test_macro_f1"),
                "best_epoch": test_summary.get("best_epoch"),
            }
        )
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Export report-ready tables from experiment outputs.")
    parser.add_argument(
        "--experiment-dirs",
        nargs="*",
        default=[],
        help="Specific experiment directories. If omitted, scan outputs/*.",
    )
    parser.add_argument("--output-dir", default="report_assets/generated")
    args = parser.parse_args()

    if args.experiment_dirs:
        experiment_dirs = [Path(path) for path in args.experiment_dirs]
    else:
        experiment_dirs = [path for path in Path("outputs").iterdir() if path.is_dir()]

    output_dir = ensure_dir(args.output_dir)
    rows = collect_experiment_rows(experiment_dirs)
    rows.sort(key=lambda item: item["experiment_name"])
    write_csv(rows, output_dir / "experiment_summary.csv")

    baseline_rows = [row for row in rows if row["model_name"] == "resnet18" and row["attention_type"] in (None, "none")]
    write_csv(baseline_rows, output_dir / "baseline_and_ablation.csv")

    attention_rows = [row for row in rows if row["attention_type"] not in (None, "none") or row["model_name"] == "vit_tiny"]
    write_csv(attention_rows, output_dir / "attention_and_transformer.csv")

    print(f"Exported {len(rows)} experiment rows to {output_dir}")


if __name__ == "__main__":
    main()
