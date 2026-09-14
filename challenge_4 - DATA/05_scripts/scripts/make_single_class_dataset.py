from pathlib import Path
import shutil


BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = BASE_DIR / "dataset_yolo26"
TARGET_DIR = BASE_DIR / "dataset_yolo26_single"

SPLITS = [
    "train",
    "val",
    "test",
]


def convert_label_file(
    source_txt: Path,
    target_txt: Path,
):
    lines_out = []

    text = source_txt.read_text(
        encoding="utf-8"
    )

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            continue

        parts[0] = "0"

        lines_out.append(
            " ".join(parts)
        )

    with target_txt.open(
        "w",
        encoding="utf-8",
    ) as f:

        if lines_out:
            f.write(
                "\n".join(lines_out)
            )
            f.write("\n")


def main():
    for split in SPLITS:

        source_images = (
            SOURCE_DIR
            / "images"
            / split
        )

        source_labels = (
            SOURCE_DIR
            / "labels"
            / split
        )

        target_images = (
            TARGET_DIR
            / "images"
            / split
        )

        target_labels = (
            TARGET_DIR
            / "labels"
            / split
        )

        target_images.mkdir(
            parents=True,
            exist_ok=True,
        )

        target_labels.mkdir(
            parents=True,
            exist_ok=True,
        )

        for image_path in source_images.iterdir():

            if not image_path.is_file():
                continue

            shutil.copy2(
                image_path,
                target_images
                / image_path.name,
            )

        for txt_path in source_labels.glob(
            "*.txt"
        ):

            convert_label_file(
                txt_path,
                target_labels
                / txt_path.name,
            )

        print(
            f"{split}: complete"
        )


    data_yaml = (
        TARGET_DIR
        / "data.yaml"
    )

    yaml_text = f"""path: {TARGET_DIR.as_posix()}

train: images/train
val: images/val
test: images/test

nc: 1

names:
  0: support
"""

    data_yaml.write_text(
        yaml_text,
        encoding="utf-8",
    )


    print()
    print(
        "Single-class dataset created:"
    )

    print(
        TARGET_DIR
    )


if __name__ == "__main__":
    main()