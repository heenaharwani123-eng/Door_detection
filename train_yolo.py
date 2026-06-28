from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO object detection model on this dataset."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Dataset root containing train/valid/test folders.",
    )
    parser.add_argument(
        "--class-name",
        default="object",
        help="Name for class id 0. Change this to your real object name.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics model checkpoint, for example yolov8n.pt or yolo11n.pt.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="custom_yolo")
    parser.add_argument(
        "--device",
        default=None,
        help="Training device, for example 0, cpu, or mps. Leave empty for auto.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the final validation pass on the test split.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision. Mostly useful for GPU training.",
    )
    return parser.parse_args()


def count_files(path: Path, suffixes: tuple[str, ...]) -> int:
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)


def validate_dataset(root: Path) -> None:
    required_dirs = [
        root / "train" / "images",
        root / "train" / "labels",
        root / "valid" / "images",
        root / "valid" / "labels",
        root / "test" / "images",
        root / "test" / "labels",
    ]
    missing = [str(path) for path in required_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset folders:\n" + "\n".join(missing))

    image_suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for split in ("train", "valid", "test"):
        images = count_files(root / split / "images", image_suffixes)
        labels = count_files(root / split / "labels", (".txt",))
        if images == 0:
            raise ValueError(f"No images found in {root / split / 'images'}")
        if images != labels:
            raise ValueError(
                f"{split} has {images} images but {labels} label files. "
                "Every image should have a matching YOLO .txt label."
            )


def write_data_yaml(root: Path, class_name: str) -> Path:
    data_yaml = root / "data.yaml"
    root_posix = root.as_posix()
    contents = "\n".join(
        [
            f"path: {root_posix}",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "",
            "names:",
            f"  0: {class_name}",
            "",
        ]
    )
    data_yaml.write_text(contents, encoding="utf-8")
    return data_yaml


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    validate_dataset(data_root)
    data_yaml = write_data_yaml(data_root, args.class_name)
    project_dir = Path(args.project)
    if not project_dir.is_absolute():
        project_dir = data_root / project_dir

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install it with:\n"
            "  pip install ultralytics\n"
            "Then run this script again."
        ) from exc

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "workers": args.workers,
        "project": str(project_dir),
        "name": args.name,
        "amp": args.amp,
        "cache": False,
        "exist_ok": True,
    }
    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    if not args.no_validate:
        model.val(data=str(data_yaml), split="test", imgsz=args.imgsz, batch=args.batch)

    best_weights = project_dir / args.name / "weights" / "best.pt"
    print(f"Training complete. Best weights: {best_weights}")
    print(f"Ultralytics result object: {results}")


if __name__ == "__main__":
    main()
