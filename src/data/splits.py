
from pathlib import Path
import re


TRAIN_CITIES = {"austin", "chicago", "kitsap", "tyrol-w"}
VAL_CITIES = {"vienna"}


def get_city_name(tile_name):
    return re.sub(r"\d+$", "", tile_name.lower())


def create_splits(data_root):

    data_root = Path(data_root)

    train_images = []
    train_masks = []

    val_images = []
    val_masks = []

    unmatched = []

    for tile_dir in sorted(data_root.iterdir()):

        if not tile_dir.is_dir():
            continue

        city = get_city_name(tile_dir.name)

        if city not in TRAIN_CITIES and city not in VAL_CITIES:
            continue

        image_dir = tile_dir / "images"
        mask_dir = tile_dir / "binary_masks"

        for image_path in sorted(image_dir.glob("*.jpg")):

            mask_path = mask_dir / f"{image_path.stem}.tif"

            if not mask_path.exists():
                unmatched.append(str(image_path))
                continue

            if city in TRAIN_CITIES:
                train_images.append(image_path)
                train_masks.append(mask_path)

            elif city in VAL_CITIES:
                val_images.append(image_path)
                val_masks.append(mask_path)

    print("Dataset split completed")
    print("------------------------")
    print("Training samples:", len(train_images))
    print("Validation samples:", len(val_images))
    print("Unmatched images:", len(unmatched))

    if unmatched:
        print("First unmatched image:", unmatched[0])

    return train_images, train_masks, val_images, val_masks
