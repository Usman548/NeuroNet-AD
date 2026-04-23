import os
from typing import Optional, Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class AlzheimerDataset(Dataset):
    """Multi-view Alzheimer dataset with axial/coronal/sagittal images."""

    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))

        self.axial_paths = []
        self.coronal_paths = []
        self.sagittal_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            axial_dir = os.path.join(class_dir, "axial")
            coronal_dir = os.path.join(class_dir, "coronal")
            sagittal_dir = os.path.join(class_dir, "sagittal")

            for img_name in sorted(os.listdir(axial_dir)):
                self.axial_paths.append(os.path.join(axial_dir, img_name))
                self.coronal_paths.append(os.path.join(coronal_dir, img_name))
                self.sagittal_paths.append(os.path.join(sagittal_dir, img_name))
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, int]:
        axial_image = Image.open(self.axial_paths[idx]).convert("RGB")
        coronal_image = Image.open(self.coronal_paths[idx]).convert("RGB")
        sagittal_image = Image.open(self.sagittal_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            axial_image = self.transform(axial_image)
            coronal_image = self.transform(coronal_image)
            sagittal_image = self.transform(sagittal_image)

        return axial_image, coronal_image, sagittal_image, label


def default_transforms() -> transforms.Compose:
    """Image transforms used by train and eval."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

