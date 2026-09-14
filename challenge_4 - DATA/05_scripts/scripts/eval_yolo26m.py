from pathlib import Path
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "runs_yolo26"
    / "yolo26m_5class_baseline-3"
    / "weights"
    / "best.pt"
)

DATA_YAML = (
    BASE_DIR
    / "dataset_yolo26"
    / "data.yaml"
)


def main():
    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
    )

    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"mAP@50:     {metrics.box.map50:.4f}")
    print(f"mAP@50-95:  {metrics.box.map:.4f}")
    print(f"Precision:   {metrics.box.mp:.4f}")
    print(f"Recall:      {metrics.box.mr:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()