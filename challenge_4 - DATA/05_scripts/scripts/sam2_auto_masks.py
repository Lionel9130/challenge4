from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from sam2.build_sam import build_sam2_hf
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "sam2_test"
OUTPUT_DIR = BASE_DIR / "sam2_auto_output"

MODEL_ID = "facebook/sam2.1-hiera-small"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


# Automatic mask generation settings
POINTS_PER_SIDE = 64
POINTS_PER_BATCH = 64

PRED_IOU_THRESH = 0.55
STABILITY_SCORE_THRESH = 0.80

CROP_N_LAYERS = 1
BOX_NMS_THRESH = 0.70


# Visualization / filtering settings
MIN_AREA_RATIO = 0.00005
MAX_AREA_RATIO = 0.05

MIN_PRED_IOU = 0.65
MIN_STABILITY = 0.85

MAX_ASPECT_RATIO = 6.0

MAX_CANDIDATES_TO_DRAW = 100

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


model = build_sam2_hf(
    MODEL_ID,
    device=device
)


mask_generator = SAM2AutomaticMaskGenerator(
    model=model,
    points_per_side=POINTS_PER_SIDE,
    points_per_batch=POINTS_PER_BATCH,
    pred_iou_thresh=PRED_IOU_THRESH,
    stability_score_thresh=STABILITY_SCORE_THRESH,
    crop_n_layers=CROP_N_LAYERS,
    box_nms_thresh=BOX_NMS_THRESH,
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


image_paths = sorted(
    path
    for path in INPUT_DIR.iterdir()
    if path.suffix.lower() in IMAGE_EXTENSIONS
)


print(f"Found {len(image_paths)} images.")


for image_index, image_path in enumerate(
    image_paths,
    start=1,
):

    print()
    print(
        f"[{image_index}/{len(image_paths)}] "
        f"Processing {image_path.name}"
    )


    image_bgr = cv2.imread(
        str(image_path)
    )

    if image_bgr is None:
        print("Failed to read image.")
        continue


    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    height, width = image_rgb.shape[:2]
    image_area = width * height


    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=(device == "cuda"),
    ):

        masks = mask_generator.generate(
            image_rgb
        )


    print(
        f"Raw masks: {len(masks)}"
    )


    candidates = []

    for mask_data in masks:
        area = mask_data["area"]
        area_ratio = area / image_area

        if area_ratio < MIN_AREA_RATIO:
            continue

        if area_ratio > MAX_AREA_RATIO:
            continue

        predicted_iou = float(
            mask_data["predicted_iou"]
        )

        stability_score = float(
            mask_data["stability_score"]
        )

        if predicted_iou < MIN_PRED_IOU:
            continue

        if stability_score < MIN_STABILITY:
            continue

        x, y, w, h = mask_data["bbox"]

        if w <= 0 or h <= 0:
            continue

        aspect_ratio = max(
            w / max(h, 1),
            h / max(w, 1)
        )

        if aspect_ratio > MAX_ASPECT_RATIO:
            continue

        candidates.append(
            {
                "bbox": (
                    int(x),
                    int(y),
                    int(x + w),
                    int(y + h),
                ),
                "area_ratio": area_ratio,
                "predicted_iou": predicted_iou,
                "stability_score": stability_score,
                "mask": mask_data["segmentation"],
            }
        )

    candidates.sort(
        key=lambda item: (
            item["predicted_iou"]
            + item["stability_score"]
        ),
        reverse=True,
    )


    print(
        f"Candidates after size filter: "
        f"{len(candidates)}"
    )


    preview = Image.fromarray(
        image_rgb.copy()
    )

    draw = ImageDraw.Draw(
        preview
    )


    draw_candidates = candidates[
        :MAX_CANDIDATES_TO_DRAW
    ]


    for candidate_index, candidate in enumerate(
        draw_candidates,
        start=1,
    ):

        x1, y1, x2, y2 = candidate["bbox"]

        draw.rectangle(
            (
                x1,
                y1,
                x2,
                y2,
            ),
            outline="red",
            width=2,
        )

        draw.text(
            (
                x1,
                max(
                    0,
                    y1 - 12,
                ),
            ),
            str(candidate_index),
            fill="yellow",
        )


    preview_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_all_candidates.jpg"
    )

    preview.save(
        preview_path
    )


    info_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_candidates.txt"
    )


    with info_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for candidate_index, candidate in enumerate(
            candidates,
            start=1,
        ):

            x1, y1, x2, y2 = candidate["bbox"]

            f.write(
                f"{candidate_index} "
                f"bbox=({x1},{y1},{x2},{y2}) "
                f"iou={candidate['predicted_iou']:.4f} "
                f"stability={candidate['stability_score']:.4f} "
                f"area_ratio={candidate['area_ratio']:.6f}\n"
            )


    print(
        f"Preview: {preview_path}"
    )

    print(
        f"Candidate info: {info_path}"
    )


print()
print("Automatic mask generation complete.")