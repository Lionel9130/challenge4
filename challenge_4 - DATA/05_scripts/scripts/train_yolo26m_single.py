from pathlib import Path
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_YAML = (
    BASE_DIR
    / "dataset_yolo26_single"
    / "data.yaml"
)

RUNS_DIR = (
    BASE_DIR
    / "runs_yolo26_single"
)

MODEL_NAME = "yolo26m.pt"


def main():
    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        device=0,
        workers=0,
        project=str(RUNS_DIR),
        name="yolo26m_single_baseline",
        exist_ok=False,
        pretrained=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()