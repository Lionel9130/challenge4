from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "cropped_640"
OUTPUT_DIR = BASE_DIR / "sam2_refined"

MODEL_ID = "facebook/sam2.1-hiera-small"

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png"
]


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


model = build_sam2_hf(
    MODEL_ID,
    device=device
)

predictor = SAM2ImagePredictor(
    model
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def find_image(stem):
    for ext in IMAGE_EXTENSIONS:
        image_path = INPUT_DIR / f"{stem}{ext}"

        if image_path.exists():
            return image_path

    return None


def read_yolo_boxes(txt_path, width, height):
    boxes = []

    with txt_path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = parts[0]

            cx = float(parts[1]) * width
            cy = float(parts[2]) * height
            bw = float(parts[3]) * width
            bh = float(parts[4]) * height

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            boxes.append(
                {
                    "class_id": class_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            )

    return boxes


def mask_to_bbox(mask):
    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max())
    )


def bbox_to_yolo(
    bbox,
    width,
    height
):
    x1, y1, x2, y2 = bbox

    cx = (
        (x1 + x2) / 2
    ) / width

    cy = (
        (y1 + y2) / 2
    ) / height

    bw = (
        x2 - x1
    ) / width

    bh = (
        y2 - y1
    ) / height

    return cx, cy, bw, bh


txt_files = sorted(
    INPUT_DIR.glob("*.txt")
)

print(
    f"Found {len(txt_files)} label files."
)


for txt_index, txt_path in enumerate(
    txt_files,
    start=1
):

    stem = txt_path.stem

    image_path = find_image(
        stem
    )

    if image_path is None:
        print(
            f"[SKIP] No image for {stem}"
        )
        continue


    image_bgr = cv2.imread(
        str(image_path)
    )

    if image_bgr is None:
        continue


    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    height, width = image_rgb.shape[:2]


    boxes = read_yolo_boxes(
        txt_path,
        width,
        height
    )


    if not boxes:
        continue


    print(
        f"[{txt_index}/{len(txt_files)}] "
        f"{stem} | boxes={len(boxes)}"
    )


    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=(device == "cuda")
    ):
        predictor.set_image(
            image_rgb
        )


        refined_labels = []

        preview = Image.fromarray(
            image_rgb.copy()
        )

        draw = ImageDraw.Draw(
            preview
        )


        for box_index, box in enumerate(
            boxes,
            start=1
        ):

            prompt_box = np.array(
                [
                    box["x1"],
                    box["y1"],
                    box["x2"],
                    box["y2"]
                ],
                dtype=np.float32
            )


            masks, scores, _ = predictor.predict(
                box=prompt_box,
                multimask_output=True
            )


            best_index = int(
                np.argmax(scores)
            )

            best_mask = masks[
                best_index
            ]

            best_score = float(
                scores[best_index]
            )


            refined_bbox = mask_to_bbox(
                best_mask
            )

            if refined_bbox is None:
                continue


            cx, cy, bw, bh = bbox_to_yolo(
                refined_bbox,
                width,
                height
            )


            refined_labels.append(
                f"{box['class_id']} "
                f"{cx:.6f} "
                f"{cy:.6f} "
                f"{bw:.6f} "
                f"{bh:.6f}"
            )


            draw.rectangle(
                (
                    box["x1"],
                    box["y1"],
                    box["x2"],
                    box["y2"]
                ),
                outline="green",
                width=3
            )


            rx1, ry1, rx2, ry2 = refined_bbox

            draw.rectangle(
                (
                    rx1,
                    ry1,
                    rx2,
                    ry2
                ),
                outline="blue",
                width=3
            )


            draw.text(
                (
                    rx1,
                    max(
                        0,
                        ry1 - 18
                    )
                ),
                f"SAM={best_score:.3f}",
                fill="yellow"
            )


    output_txt = (
        OUTPUT_DIR
        / f"{stem}.txt"
    )

    with output_txt.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(
                refined_labels
            )
        )

        if refined_labels:
            f.write("\n")


    output_preview = (
        OUTPUT_DIR
        / f"{stem}_preview.jpg"
    )

    preview.save(
        output_preview
    )


print()
print("Refinement complete.")