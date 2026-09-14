from pathlib import Path
from collections import Counter
import shutil
import numpy as np

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "yolo26_raw"
DATASET_DIR = BASE_DIR / "dataset_yolo26"

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

TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10

RANDOM_STATE = 42


def find_image(stem: str):
    for ext in IMAGE_EXTENSIONS:
        path = RAW_DIR / f"{stem}{ext}"

        if path.exists():
            return path

        upper_path = RAW_DIR / f"{stem}{ext.upper()}"

        if upper_path.exists():
            return upper_path

    return None


def read_image_classes(txt_path: Path):
    present_classes = set()
    class_counter = Counter()

    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            if class_id not in CLASS_NAMES:
                continue

            present_classes.add(class_id)
            class_counter[class_id] += 1

    return present_classes, class_counter


def clear_split_directories():
    for split in ["train", "val", "test"]:
        image_dir = DATASET_DIR / "images" / split
        label_dir = DATASET_DIR / "labels" / split

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for path in image_dir.iterdir():
            if path.is_file():
                path.unlink()

        for path in label_dir.iterdir():
            if path.is_file():
                path.unlink()


def copy_samples(samples, split_name):
    image_out = DATASET_DIR / "images" / split_name
    label_out = DATASET_DIR / "labels" / split_name

    for sample in samples:
        shutil.copy2(
            sample["image_path"],
            image_out / sample["image_path"].name,
        )

        shutil.copy2(
            sample["txt_path"],
            label_out / sample["txt_path"].name,
        )


def print_split_stats(name, samples):
    class_counter = Counter()

    for sample in samples:
        class_counter.update(
            sample["class_counter"]
        )

    print()
    print(f"{name.upper()}")
    print("-" * 60)
    print(f"Images: {len(samples)}")

    for class_id, class_name in CLASS_NAMES.items():
        print(
            f"{class_id}: "
            f"{class_name:<15} "
            f"{class_counter[class_id]}"
        )


def main():
    txt_files = sorted(
        RAW_DIR.glob("*.txt")
    )

    if not txt_files:
        print(
            f"[ERROR] No txt files found in {RAW_DIR}"
        )
        return

    samples = []

    for txt_path in txt_files:
        image_path = find_image(
            txt_path.stem
        )

        if image_path is None:
            continue

        present_classes, class_counter = (
            read_image_classes(
                txt_path
            )
        )

        multi_hot = np.zeros(
            len(CLASS_NAMES),
            dtype=np.int32,
        )

        for class_id in present_classes:
            multi_hot[class_id] = 1

        samples.append(
            {
                "stem": txt_path.stem,
                "image_path": image_path,
                "txt_path": txt_path,
                "multi_hot": multi_hot,
                "class_counter": class_counter,
            }
        )

    print(
        f"Total paired samples: {len(samples)}"
    )

    X = np.arange(
        len(samples)
    ).reshape(-1, 1)

    Y = np.stack(
        [
            sample["multi_hot"]
            for sample in samples
        ]
    )

    splitter_1 = (
        MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=(
                VAL_RATIO + TEST_RATIO
            ),
            random_state=RANDOM_STATE,
        )
    )

    train_idx, temp_idx = next(
        splitter_1.split(X, Y)
    )

    temp_X = X[temp_idx]
    temp_Y = Y[temp_idx]

    test_fraction_of_temp = (
        TEST_RATIO
        / (VAL_RATIO + TEST_RATIO)
    )

    splitter_2 = (
        MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=test_fraction_of_temp,
            random_state=RANDOM_STATE,
        )
    )

    val_rel_idx, test_rel_idx = next(
        splitter_2.split(
            temp_X,
            temp_Y,
        )
    )

    val_idx = temp_idx[
        val_rel_idx
    ]

    test_idx = temp_idx[
        test_rel_idx
    ]

    train_samples = [
        samples[i]
        for i in train_idx
    ]

    val_samples = [
        samples[i]
        for i in val_idx
    ]

    test_samples = [
        samples[i]
        for i in test_idx
    ]

    clear_split_directories()

    copy_samples(
        train_samples,
        "train",
    )

    copy_samples(
        val_samples,
        "val",
    )

    copy_samples(
        test_samples,
        "test",
    )

    print_split_stats(
        "train",
        train_samples,
    )

    print_split_stats(
        "val",
        val_samples,
    )

    print_split_stats(
        "test",
        test_samples,
    )

    print()
    print("=" * 60)
    print("Split complete")
    print("=" * 60)


if __name__ == "__main__":
    main()