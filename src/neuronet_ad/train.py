import argparse
import os
import time
from typing import Dict

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from neuronet_ad.dataset import AlzheimerDataset, default_transforms
from neuronet_ad.model import MultiViewMobileNetV3WithResidualCBAM


def build_loaders(data_dir: str, batch_size: int, split_ratio: float):
    dataset = AlzheimerDataset(root_dir=data_dir, transform=default_transforms())
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, len(dataset.classes)


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []

    with tqdm(loader, desc="Training", unit="batch") as pbar:
        for axial_images, coronal_images, sagittal_images, labels in pbar:
            axial_images = axial_images.to(device)
            coronal_images = coronal_images.to(device)
            sagittal_images = sagittal_images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(axial_images, sagittal_images, coronal_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))

    metrics = {
        "loss": running_loss / max(len(loader), 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []

    for axial_images, coronal_images, sagittal_images, labels in loader:
        axial_images = axial_images.to(device)
        coronal_images = coronal_images.to(device)
        sagittal_images = sagittal_images.to(device)
        labels = labels.to(device)

        outputs = model(axial_images, sagittal_images, coronal_images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    return {
        "loss": running_loss / max(len(loader), 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train NeuroNet-AD multi-view model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to 2D multi-view dataset root")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, num_classes = build_loaders(args.data_dir, args.batch_size, args.train_split)
    model = MultiViewMobileNetV3WithResidualCBAM(
        num_classes=num_classes, pretrained=not args.no_pretrained
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    for epoch in range(args.epochs):
        start_time = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
            f"train_f1={train_metrics['f1']:.4f} val_f1={val_metrics['f1']:.4f} | "
            f"time={elapsed:.2f}s"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "metrics": val_metrics,
                },
                save_path,
            )
            print(f"Saved best checkpoint to: {save_path}")


if __name__ == "__main__":
    main()

