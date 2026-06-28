from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

# Put your video path here, then run: python predict_video.py
VIDEO_PATH = r"D:\Door_Detection\video1.mp4"

WEIGHTS_PATH = ROOT / "runs" / "custom_yolo" / "weights" / "best.pt"
OUTPUT_PATH = ROOT / "output_truck_open.mp4"
DETECTED_FRAMES_DIR = ROOT / "truck_back_frames"
TARGET_CLASS_NAME = "truck_back"
ALERT_TEXT = "THE TRUCK IS OPEN"
NOT_AVAILABLE_TEXT = "not available"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the trained YOLO model on a video and show an alert when the truck is open."
    )
    parser.add_argument(
        "video",
        type=Path,
        nargs="?",
        default=VIDEO_PATH or None,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=WEIGHTS_PATH,
        help="Path to trained YOLO weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path for the annotated output video.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=DETECTED_FRAMES_DIR,
        help="Folder where frames with truck_back detections will be saved.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Minimum confidence score for showing the truck-open message.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=416,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 0 or cpu. Leave empty for auto.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a preview window; only save the output video.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save an output video; only show the preview window.",
    )
    return parser.parse_args()


def draw_alert(frame, confidence: float) -> None:
    label = f"{ALERT_TEXT} ({confidence:.2f})"
    cv2.rectangle(frame, (18, 18), (520, 78), (0, 0, 255), thickness=-1)
    cv2.putText(
        frame,
        label,
        (32, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_not_available(frame) -> None:
    cv2.rectangle(frame, (18, 18), (300, 78), (80, 80, 80), thickness=-1)
    cv2.putText(
        frame,
        NOT_AVAILABLE_TEXT,
        (32, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def open_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")
    return writer


def best_target_confidence(result, target_class_name: str) -> float:
    if result.boxes is None or len(result.boxes) == 0:
        return 0.0

    names = result.names or {}
    best_conf = 0.0
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = names.get(class_id, str(class_id))

        # This dataset is one class. If the model still calls it "object",
        # treat class 0 as truck_back.
        if class_name == target_class_name or class_id == 0:
            best_conf = max(best_conf, float(box.conf.item()))

    return best_conf


def main() -> None:
    args = parse_args()
    if args.video is None:
        raise SystemExit(
            "Set VIDEO_PATH near the top of predict_video.py, or pass a video path:\n"
            "  python predict_video.py path\\to\\video.mp4"
        )
    video_path = args.video.resolve()
    weights_path = args.weights.resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            "Train the model first with: python train_yolo.py"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if not args.no_save:
        writer = open_writer(args.output.resolve(), fps, width, height)
    frames_dir = args.frames_dir.resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    predict_kwargs = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    frame_count = 0
    alert_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(frame, **predict_kwargs)
            result = results[0]
            annotated = result.plot()

            best_conf = best_target_confidence(result, TARGET_CLASS_NAME)
            detected = best_conf >= args.conf

            if detected:
                alert_count += 1
                draw_alert(annotated, best_conf)
                frame_path = frames_dir / f"truck_back_frame_{frame_count:06d}_{best_conf:.2f}.jpg"
                cv2.imwrite(str(frame_path), annotated)
                print(f"Frame {frame_count}: {ALERT_TEXT} ({best_conf:.2f})")
            else:
                draw_not_available(annotated)
                print(f"Frame {frame_count}: {NOT_AVAILABLE_TEXT}")

            if writer is not None:
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("Truck Open Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames.")
    print(f"Truck-open alert shown on {alert_count} frames.")
    print(f"Saved truck_back frames to: {frames_dir}")
    if writer is not None:
        print(f"Saved annotated video to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
