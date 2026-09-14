from pathlib import Path
from PIL import Image


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "original"
OUTPUT_DIR = BASE_DIR / "cropped_640"


# ============================================================
# Crop settings
# ============================================================

CROP_SIZE = 640

# Keep a clipped box only if at least 40% of its original area
# remains visible inside the crop.
MIN_REMAINING_AREA_RATIO = 0.40

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
]


# ============================================================
# Helper functions
# ============================================================

def find_matching_image(txt_path: Path):
    for ext in IMAGE_EXTENSIONS:
        image_path = txt_path.with_suffix(ext)
        if image_path.exists():
            return image_path

        upper_path = txt_path.with_suffix(ext.upper())
        if upper_path.exists():
            return upper_path

    return None


def read_yolo_labels(txt_path: Path, image_width: int, image_height: int):
    boxes = []

    with txt_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            print(
                f"[WARNING] {txt_path.name}, line {line_number}: "
                f"expected 5 YOLO fields, skipped: {line}"
            )
            continue

        class_id = parts[0]

        try:
            cx_norm = float(parts[1])
            cy_norm = float(parts[2])
            w_norm = float(parts[3])
            h_norm = float(parts[4])
        except ValueError:
            print(
                f"[WARNING] {txt_path.name}, line {line_number}: "
                f"invalid numeric values, skipped: {line}"
            )
            continue

        cx = cx_norm * image_width
        cy = cy_norm * image_height
        box_w = w_norm * image_width
        box_h = h_norm * image_height

        x1 = cx - box_w / 2.0
        y1 = cy - box_h / 2.0
        x2 = cx + box_w / 2.0
        y2 = cy + box_h / 2.0

        boxes.append(
            {
                "class_id": class_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": cx,
                "cy": cy,
            }
        )

    return boxes


def calculate_crop_window(cx, cy, image_width, image_height):
    half = CROP_SIZE / 2.0

    left = round(cx - half)
    top = round(cy - half)

    left = max(0, left)
    top = max(0, top)

    left = min(left, image_width - CROP_SIZE)
    top = min(top, image_height - CROP_SIZE)

    right = left + CROP_SIZE
    bottom = top + CROP_SIZE

    return left, top, right, bottom


def convert_boxes_to_crop(boxes, left, top, right, bottom):
    new_labels = []

    for box in boxes:
        original_x1 = box["x1"]
        original_y1 = box["y1"]
        original_x2 = box["x2"]
        original_y2 = box["y2"]

        original_width = max(0.0, original_x2 - original_x1)
        original_height = max(0.0, original_y2 - original_y1)
        original_area = original_width * original_height

        if original_area <= 0:
            continue

        clipped_x1 = max(original_x1, left)
        clipped_y1 = max(original_y1, top)
        clipped_x2 = min(original_x2, right)
        clipped_y2 = min(original_y2, bottom)

        clipped_width = clipped_x2 - clipped_x1
        clipped_height = clipped_y2 - clipped_y1

        if clipped_width <= 0 or clipped_height <= 0:
            continue

        clipped_area = clipped_width * clipped_height
        remaining_ratio = clipped_area / original_area

        if remaining_ratio < MIN_REMAINING_AREA_RATIO:
            continue

        local_x1 = clipped_x1 - left
        local_y1 = clipped_y1 - top
        local_x2 = clipped_x2 - left
        local_y2 = clipped_y2 - top

        new_cx = ((local_x1 + local_x2) / 2.0) / CROP_SIZE
        new_cy = ((local_y1 + local_y2) / 2.0) / CROP_SIZE
        new_w = (local_x2 - local_x1) / CROP_SIZE
        new_h = (local_y2 - local_y1) / CROP_SIZE

        new_cx = min(max(new_cx, 0.0), 1.0)
        new_cy = min(max(new_cy, 0.0), 1.0)
        new_w = min(max(new_w, 0.0), 1.0)
        new_h = min(max(new_h, 0.0), 1.0)

        new_labels.append(
            f'{box["class_id"]} '
            f'{new_cx:.6f} '
            f'{new_cy:.6f} '
            f'{new_w:.6f} '
            f'{new_h:.6f}'
        )

    return new_labels


def process_one_label_file(txt_path: Path):
    image_path = find_matching_image(txt_path)

    if image_path is None:
        print(f"[SKIP] No matching image for {txt_path.name}")
        return 0

    with Image.open(image_path) as img:
        img = img.convert("RGB")

        image_width, image_height = img.size

        if image_width < CROP_SIZE or image_height < CROP_SIZE:
            print(
                f"[SKIP] {image_path.name}: "
                f"{image_width}x{image_height} is smaller than "
                f"{CROP_SIZE}x{CROP_SIZE}"
            )
            return 0

        boxes = read_yolo_labels(txt_path, image_width, image_height)

        if not boxes:
            print(f"[SKIP] {txt_path.name}: no valid boxes")
            return 0

        generated_count = 0

        for box_index, anchor_box in enumerate(boxes, start=1):
            left, top, right, bottom = calculate_crop_window(
                anchor_box["cx"],
                anchor_box["cy"],
                image_width,
                image_height,
            )

            cropped_image = img.crop((left, top, right, bottom))

            new_labels = convert_boxes_to_crop(
                boxes,
                left,
                top,
                right,
                bottom,
            )

            if not new_labels:
                print(
                    f"[WARNING] {txt_path.name}, object {box_index}: "
                    f"crop contains no valid labels, skipped"
                )
                continue

            new_stem = f"{txt_path.stem}_obj{box_index:03d}"

            output_image_path = OUTPUT_DIR / f"{new_stem}.jpg"
            output_txt_path = OUTPUT_DIR / f"{new_stem}.txt"

            cropped_image.save(output_image_path, quality=95)

            with output_txt_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(new_labels))
                f.write("\n")

            generated_count += 1

            print(
                f"[CREATED] {new_stem} | "
                f"crop=({left},{top})-({right},{bottom}) | "
                f"labels={len(new_labels)}"
            )

    return generated_count


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory does not exist: {INPUT_DIR}")
        return

    txt_files = sorted(INPUT_DIR.glob("*.txt"))

    if not txt_files:
        print(f"[ERROR] No YOLO .txt files found in: {INPUT_DIR}")
        return

    print("=" * 72)
    print("YOLO bbox-centered 640x640 crop generator")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Crop size: {CROP_SIZE}x{CROP_SIZE}")
    print(f"Minimum remaining box area: {MIN_REMAINING_AREA_RATIO:.0%}")
    print(f"Label files found: {len(txt_files)}")
    print("=" * 72)

    total_generated = 0

    for txt_index, txt_path in enumerate(txt_files, start=1):
        print(
            f"\n[{txt_index}/{len(txt_files)}] "
            f"Processing {txt_path.name}"
        )

        total_generated += process_one_label_file(txt_path)

    print()
    print("=" * 72)
    print("Processing complete")
    print(f"Generated image/label pairs: {total_generated}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
