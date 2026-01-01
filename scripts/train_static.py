# scripts/train_static.py

import os, argparse
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from scripts.dataset import StaticLandmarkDataset
from scripts.models import StaticMLP

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


# ---------------------------
# Training function
# ---------------------------
def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total, loss_sum = 0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()

        total += x.size(0)
        loss_sum += loss.item() * x.size(0)

    return loss_sum / total


# ---------------------------
# Validation + Confusion Matrix
# ---------------------------
def eval_with_confusion_matrix(model, loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)

            all_preds.extend(list(preds.cpu().numpy()))
            all_labels.extend(list(y.cpu().numpy()))

    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(num_classes))   # <-- FIX: Ensures full class count (no mismatch)
    )

    return acc, cm



def main(args):

    train_losses = []
    val_accuracies = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    ds = StaticLandmarkDataset(args.landmark_dir)
    n_classes = len(ds.class_map)

    # Split dataset
    indices = list(range(len(ds)))
    train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=42, shuffle=True)

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # Model setup
    input_dim = 42   # 21 * 2
    model = StaticMLP(input_dim=input_dim, n_classes=n_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    # ------------- Training Loop -------------
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, opt, criterion, device)
        acc, cm = eval_with_confusion_matrix(model, val_loader, device, n_classes)

        train_losses.append(loss)
        val_accuracies.append(acc)

        print(f"Epoch {epoch+1}/{args.epochs} loss={loss:.4f} val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({'model_state': model.state_dict(), 'class_map': ds.class_map}, args.save)

    print("Training done. Best val acc:", best_acc)

    # -------------------- PLOTTING --------------------

    epochs_range = range(1, args.epochs + 1)

    # Training Loss Curve
    plt.figure()
    plt.plot(epochs_range, train_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.show()

    # Validation Accuracy Curve
    plt.figure()
    plt.plot(epochs_range, val_accuracies)
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.show()

    # -------------------- CONFUSION MATRIX PLOT --------------------

    # Fix label ordering → Sort by class index
    labels = [cls for cls, idx in sorted(ds.class_map.items(), key=lambda x: x[1])]

    plt.figure(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=90)
    plt.title("Confusion Matrix (Validation Data)")
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--landmark_dir', required=True)
    parser.add_argument('--save', default='models/static_mlp.pth')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()
    main(args)
