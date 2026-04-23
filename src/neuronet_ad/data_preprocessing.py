import os
from typing import Iterable

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def save_slices_for_nii(nii_file_path: str, output_class_dir: str, file_index: int) -> None:
    """Save axial/coronal/sagittal slices for a single NIfTI volume."""
    nifti_image = nib.load(nii_file_path)
    image_data = np.squeeze(nifti_image.get_fdata())

    axial_dir = os.path.join(output_class_dir, "axial")
    coronal_dir = os.path.join(output_class_dir, "coronal")
    sagittal_dir = os.path.join(output_class_dir, "sagittal")
    os.makedirs(axial_dir, exist_ok=True)
    os.makedirs(coronal_dir, exist_ok=True)
    os.makedirs(sagittal_dir, exist_ok=True)

    num_axial_slices = image_data.shape[2]

    for i in range(num_axial_slices):
        axial_slice = image_data[:, :, i]
        plt.imsave(
            os.path.join(axial_dir, f"{file_index}_{i + 1}.png"),
            axial_slice,
            cmap="gray",
            format="png",
        )

    num_coronal_slices = image_data.shape[1]
    step_size_coronal = max(1, num_coronal_slices // num_axial_slices)
    for i in range(num_axial_slices):
        coronal_slice = image_data[:, i * step_size_coronal, :]
        plt.imsave(
            os.path.join(coronal_dir, f"{file_index}_{i + 1}.png"),
            coronal_slice,
            cmap="gray",
            format="png",
        )

    num_sagittal_slices = image_data.shape[0]
    step_size_sagittal = max(1, num_sagittal_slices // num_axial_slices)
    for i in range(num_axial_slices):
        sagittal_slice = image_data[i * step_size_sagittal, :, :]
        plt.imsave(
            os.path.join(sagittal_dir, f"{file_index}_{i + 1}.png"),
            sagittal_slice,
            cmap="gray",
            format="png",
        )


def preprocess_oasis(
    input_base_dir: str,
    output_base_dir: str,
    class_names: Iterable[str],
) -> None:
    """Convert all .nii files into multi-view PNG slices by class."""
    for class_name in class_names:
        input_class_dir = os.path.join(input_base_dir, class_name)
        output_class_dir = os.path.join(output_base_dir, class_name)

        nii_files = [f for f in os.listdir(input_class_dir) if f.endswith(".nii")]
        for file_index, nii_file in enumerate(nii_files, start=1):
            nii_file_path = os.path.join(input_class_dir, nii_file)
            save_slices_for_nii(nii_file_path, output_class_dir, file_index)

