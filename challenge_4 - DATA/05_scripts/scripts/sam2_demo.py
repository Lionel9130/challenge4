from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "sam2_test"
OUTPUT_DIR = BASE_DIR / "sam2_grid_output"

MODEL_ID = "facebook/sam2.1-hiera-small"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}

BOX_HALF_SIZE = 140

GRID_SIZE = 3

MAX_AREA_RATIO = 0.15
MIN_AREA_RATIO = 0.0002


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


def get_bbox(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())

    return xmin, ymin, xmax, ymax


def box_area_ratio(bbox, width, height):
    xmin, ymin, xmax, ymax = bbox

    box_area = (
        (xmax - xmin)
        * (ymax - ymin)
    )

    image_area = width * height

    return box_area / image_area


def bbox_center(bbox):
    xmin, ymin, xmax, ymax = bbox

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    return cx, cy


def normalized_distance(
    bbox,
    point_x,
    point_y,
    box_half_size
):
    cx, cy = bbox_center(bbox)

    dx = cx - point_x
    dy = cy - point_y

    distance = np.sqrt(
        dx * dx + dy * dy
    )

    max_distance = np.sqrt(
        2 * (box_half_size ** 2)
    )

    return min(
        distance / max_distance,
        1.0
    )


def final_candidate_score(
    sam_score,
    area_ratio,
    distance_ratio
):
    size_score = 1.0 - min(
        area_ratio / MAX_AREA_RATIO,
        1.0
    )

    proximity_score = 1.0 - distance_ratio

    final_score = (
        0.50 * sam_score
        + 0.30 * proximity_score
        + 0.20 * size_score
    )

    return final_score


image_paths = sorted(
    path
    for path in INPUT_DIR.iterdir()
    if path.suffix.lower() in IMAGE_EXTENSIONS
)


print(
    f"Found {len(image_paths)} images."
)


for image_index, image_path in enumerate(
    image_paths,
    start=1
):

    print()
    print(
        f"[{image_index}/{len(image_paths)}] "
        f"{image_path.name}"
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

    center_x = width // 2
    center_y = height // 2


    box_x1 = max(
        0,
        center_x - BOX_HALF_SIZE
    )

    box_y1 = max(
        0,
        center_y - BOX_HALF_SIZE
    )

    box_x2 = min(
        width - 1,
        center_x + BOX_HALF_SIZE
    )

    box_y2 = min(
        height - 1,
        center_y + BOX_HALF_SIZE
    )


    grid_x = np.linspace(
        box_x1 + BOX_HALF_SIZE * 0.25,
        box_x2 - BOX_HALF_SIZE * 0.25,
        GRID_SIZE
    )

    grid_y = np.linspace(
        box_y1 + BOX_HALF_SIZE * 0.25,
        box_y2 - BOX_HALF_SIZE * 0.25,
        GRID_SIZE
    )


    candidates = []


    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=(device == "cuda")
    ):

        predictor.set_image(
            image_rgb
        )

        for gy in grid_y:
            for gx in grid_x:

                point_x = int(gx)
                point_y = int(gy)

                point_coords = np.array(
                    [[point_x, point_y]],
                    dtype=np.float32
                )

                point_labels = np.array(
                    [1],
                    dtype=np.int32
                )

                prompt_box = np.array(
                    [
                        box_x1,
                        box_y1,
                        box_x2,
                        box_y2
                    ],
                    dtype=np.float32
                )

                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=prompt_box,
                    multimask_output=True
                )


                for mask_index, mask in enumerate(masks):

                    sam_score = float(
                        scores[mask_index]
                    )

                    bbox = get_bbox(mask)

                    if bbox is None:
                        continue

                    area_ratio = box_area_ratio(
                        bbox,
                        width,
                        height
                    )

                    if area_ratio > MAX_AREA_RATIO:
                        continue

                    if area_ratio < MIN_AREA_RATIO:
                        continue

                    distance_ratio = normalized_distance(
                        bbox,
                        center_x,
                        center_y,
                        BOX_HALF_SIZE
                    )

                    final_score = final_candidate_score(
                        sam_score,
                        area_ratio,
                        distance_ratio
                    )

                    candidates.append(
                        {
                            "mask": mask,
                            "bbox": bbox,
                            "sam_score": sam_score,
                            "final_score": final_score,
                            "point": (
                                point_x,
                                point_y
                            ),
                            "area_ratio": area_ratio
                        }
                    )


    if not candidates:
        print("No valid candidate found.")
        continue


    candidates.sort(
        key=lambda item: item["final_score"],
        reverse=True
    )

    best = candidates[0]


    preview = Image.fromarray(
        image_rgb.copy()
    )

    draw = ImageDraw.Draw(
        preview
    )


    draw.rectangle(
        (
            box_x1,
            box_y1,
            box_x2,
            box_y2
        ),
        outline="green",
        width=3
    )


    for gy in grid_y:
        for gx in grid_x:

            px = int(gx)
            py = int(gy)

            draw.ellipse(
                (
                    px - 4,
                    py - 4,
                    px + 4,
                    py + 4
                ),
                fill="orange"
            )


    best_point_x, best_point_y = best["point"]

    draw.ellipse(
        (
            best_point_x - 7,
            best_point_y - 7,
            best_point_x + 7,
            best_point_y + 7
        ),
        fill="red"
    )


    xmin, ymin, xmax, ymax = best["bbox"]

    draw.rectangle(
        (
            xmin,
            ymin,
            xmax,
            ymax
        ),
        outline="blue",
        width=4
    )


    draw.text(
        (
            xmin,
            max(0, ymin - 35)
        ),
        (
            f"SAM={best['sam_score']:.3f} "
            f"FINAL={best['final_score']:.3f}"
        ),
        fill="yellow"
    )


    preview_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_grid_best.jpg"
    )

    preview.save(
        preview_path
    )


    mask_image = (
        best["mask"].astype(
            np.uint8
        ) * 255
    )

    mask_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_grid_mask.png"
    )

    cv2.imwrite(
        str(mask_path),
        mask_image
    )


    print(
        f"Candidates kept: {len(candidates)}"
    )

    print(
        f"Best point: {best['point']}"
    )

    print(
        f"SAM score: {best['sam_score']:.4f}"
    )

    print(
        f"Final score: {best['final_score']:.4f}"
    )

    print(
        f"Area ratio: {best['area_ratio']:.5f}"
    )

    print(
        f"Box: {best['bbox']}"
    )


print()
print("Grid search complete.")