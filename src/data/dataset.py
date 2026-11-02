
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


def get_train_transform(crop=512):
    return A.Compose([
        A.RandomCrop(height=crop, width=crop),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ], additional_targets={"boundary": "mask"})


def get_val_transform(crop=512):
    return A.Compose([
        A.CenterCrop(height=crop, width=crop),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ], additional_targets={"boundary": "mask"})


class InriaDataset(Dataset):

    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

        assert len(self.image_paths) == len(self.mask_paths), \
            "Image and mask counts do not match!"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = str(self.image_paths[idx])
        mask_path = str(self.mask_paths[idx])

        # Load RGB image
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load grayscale binary mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")

        mask = (mask > 127).astype(np.uint8)

        # Generate boundary ground truth
        kernel = np.ones((3, 3), dtype=np.uint8)

        boundary = cv2.morphologyEx(
            mask,
            cv2.MORPH_GRADIENT,
            kernel
        )

        boundary = (boundary > 0).astype(np.uint8)

        # Apply synchronized image/mask transformations
        if self.transform is not None:
            augmented = self.transform(
                image=image,
                mask=mask,
                boundary=boundary
            )

            image = augmented["image"]
            mask = augmented["mask"]
            boundary = augmented["boundary"]

        # Convert HWC -> CHW
        image = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        ).float()

        mask = torch.from_numpy(
            np.ascontiguousarray(mask)
        ).float().unsqueeze(0)

        boundary = torch.from_numpy(
            np.ascontiguousarray(boundary)
        ).float().unsqueeze(0)

        # Two target channels: building mask + boundary mask
        target = torch.cat([mask, boundary], dim=0)

        return image, target
