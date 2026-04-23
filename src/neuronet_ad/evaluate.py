import argparse

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from neuronet_ad.dataset import AlzheimerDataset, default_transforms
from neuronet_ad.model import MultiViewMobileNetV3WithResidualCBAM


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    running_loss = 0.0

    for axial_images, coronal_images, sagittal_images, labels in loader:
        axial_images = axial_images.to(device)
        coronal_images = coronal_images.to(device)
        sagittal_images = sagittal_images.to(device)
        labels = labels.to(device)

        outputs = model(axial_images, sagittal_images, coronal_images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    print(f"Loss: {running_loss / max(len(loader), 1):.4f}")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print(f"Recall: {recall_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NeuroNet-AD model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to 2D multi-view dataset root")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AlzheimerDataset(args.data_dir, transform=default_transforms())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = MultiViewMobileNetV3WithResidualCBAM(num_classes=checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluate(model, dataloader, device)


if __name__ == "__main__":
    main()

