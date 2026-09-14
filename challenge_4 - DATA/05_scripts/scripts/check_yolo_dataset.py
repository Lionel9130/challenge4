from pathlib import Path
from collections import Counter


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "yolo26_raw"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

CLASS_NAMES = {
    0: "pylone",
    1: "mat",
    2: "batiment",
    3: "chateau_eau",
    4: "silo",
}


def find_image_for_label(txt_path: Path):
    for ext in IMAGE_EXTENSIONS:
        image_path = txt_path.with_suffix(ext)

        if image_path.exists():
            return image_path

        upper_path = txt_path.with_suffix(ext.upper())

        if upper_path.exists():
            return upper_path

    return None


def find_label_for_image(image_path: Path):
    txt_path = image_path.with_suffix(".txt")

    if txt_path.exists():
        return txt_path

    return None


def validate_label_file(txt_path: Path):
    errors = []
    class_counter = Counter()
    valid_boxes = 0

    text = txt_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return {
            "empty": True,
            "errors": [],
            "class_counter": class_counter,
            "valid_boxes": 0,
        }

    lines = text.splitlines()

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            errors.append(
                f"line {line_number}: "
                f"expected 5 fields, got {len(parts)}"
            )
            continue

        try:
            class_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            errors.append(
                f"line {line_number}: "
                f"invalid numeric value"
            )
            continue

        if class_id not in CLASS_NAMES:
            errors.append(
                f"line {line_number}: "
                f"invalid class id {class_id}"
            )
            continue

        if not 0.0 <= cx <= 1.0:
            errors.append(
                f"line {line_number}: "
                f"x_center out of range: {cx}"
            )

        if not 0.0 <= cy <= 1.0:
            errors.append(
                f"line {line_number}: "
                f"y_center out of range: {cy}"
            )

        if not 0.0 < width <= 1.0:
            errors.append(
                f"line {line_number}: "
                f"width out of range: {width}"
            )

        if not 0.0 < height <= 1.0:
            errors.append(
                f"line {line_number}: "
                f"height out of range: {height}"
            )

        if (
            0.0 <= cx <= 1.0
            and 0.0 <= cy <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            class_counter[class_id] += 1
            valid_boxes += 1

    return {
        "empty": False,
        "errors": errors,
        "class_counter": class_counter,
        "valid_boxes": valid_boxes,
    }


def main():
    if not INPUT_DIR.exists():
        print(
            f"[ERROR] Input directory not found: "
            f"{INPUT_DIR}"
        )
        return

    image_files = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    txt_files = sorted(
        INPUT_DIR.glob("*.txt")
    )

    print("=" * 72)
    print("YOLO dataset validation")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Images found: {len(image_files)}")
    print(f"Label files found: {len(txt_files)}")
    print("=" * 72)

    missing_labels = []
    missing_images = []
    empty_labels = []
    invalid_labels = []

    total_class_counter = Counter()
    total_boxes = 0

    boxes_per_image = []

    for image_path in image_files:
        txt_path = find_label_for_image(
            image_path
        )

        if txt_path is None:
            missing_labels.append(
                image_path.name
            )

    for txt_path in txt_files:
        image_path = find_image_for_label(
            txt_path
        )

        if image_path is None:
            missing_images.append(
                txt_path.name
            )
            continue

        result = validate_label_file(
            txt_path
        )

        if result["empty"]:
            empty_labels.append(
                txt_path.name
            )

        if result["errors"]:
            invalid_labels.append(
                (
                    txt_path.name,
                    result["errors"],
                )
            )

        total_class_counter.update(
            result["class_counter"]
        )

        total_boxes += result["valid_boxes"]

        boxes_per_image.append(
            (
                txt_path.stem,
                result["valid_boxes"],
            )
        )

    print()
    print("CLASS COUNTS")
    print("-" * 72)

    for class_id, class_name in CLASS_NAMES.items():
        count = total_class_counter[
            class_id
        ]

        print(
            f"{class_id}: "
            f"{class_name:<15} "
            f"{count}"
        )

    print()
    print(f"Total valid boxes: {total_boxes}")

    if boxes_per_image:
        box_counts = [
            count
            for _, count in boxes_per_image
        ]

        print(
            "Average boxes per labeled image: "
            f"{sum(box_counts) / len(box_counts):.2f}"
        )

        print(
            f"Maximum boxes in one image: "
            f"{max(box_counts)}"
        )

    print()
    print("PAIRING CHECK")
    print("-" * 72)

    print(
        f"Images without txt: "
        f"{len(missing_labels)}"
    )

    for name in missing_labels:
        print(f"  {name}")

    print(
        f"Txt without image: "
        f"{len(missing_images)}"
    )

    for name in missing_images:
        print(f"  {name}")

    print()
    print("LABEL CHECK")
    print("-" * 72)

    print(
        f"Empty label files: "
        f"{len(empty_labels)}"
    )

    for name in empty_labels:
        print(f"  {name}")

    print(
        f"Invalid label files: "
        f"{len(invalid_labels)}"
    )

    for file_name, errors in invalid_labels:
        print()
        print(f"  {file_name}")

        for error in errors:
            print(f"    - {error}")

    print()
    print("=" * 72)

    if (
        not missing_labels
        and not missing_images
        and not invalid_labels
    ):
        print(
            "DATASET CHECK PASSED"
        )
    else:
        print(
            "DATASET CHECK FOUND ISSUES"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()