from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

LABEL_DIRS = [
    BASE_DIR / "dataset_yolo26_single" / "labels" / "train",
    BASE_DIR / "dataset_yolo26_single" / "labels" / "val",
    BASE_DIR / "dataset_yolo26_single" / "labels" / "test",
]


def convert_file(txt_path: Path):
    output_lines = []

    text = txt_path.read_text(
        encoding="utf-8"
    )

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            print(
                f"[SKIP] Invalid line in {txt_path.name}: {line}"
            )
            continue

        parts[0] = "0"

        output_lines.append(
            " ".join(parts)
        )

    txt_path.write_text(
        "\n".join(output_lines)
        + ("\n" if output_lines else ""),
        encoding="utf-8",
    )


def main():
    total_files = 0
    total_boxes = 0

    for label_dir in LABEL_DIRS:

        if not label_dir.exists():
            print(
                f"[SKIP] Directory not found: {label_dir}"
            )
            continue

        txt_files = list(
            label_dir.glob("*.txt")
        )

        print(
            f"Processing {label_dir}"
        )

        for txt_path in txt_files:

            old_text = txt_path.read_text(
                encoding="utf-8"
            )

            box_count = sum(
                1
                for line in old_text.splitlines()
                if line.strip()
            )

            convert_file(
                txt_path
            )

            total_files += 1
            total_boxes += box_count


    print()
    print("Complete")
    print(f"Files modified: {total_files}")
    print(f"Boxes modified: {total_boxes}")


if __name__ == "__main__":
    main()