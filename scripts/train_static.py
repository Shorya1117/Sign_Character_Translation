# scripts/train_static.py
import os, argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from scripts.dataset import StaticLandmarkDataset
from scripts.models import StaticMLP
from sklearn.model_selection import train_test_split

def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total, loss_sum = 0, 0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        total += x.size(0)
        loss_sum += loss.item() * x.size(0)
    return loss_sum/total

def eval_epoch(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return correct/total

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = StaticLandmarkDataset(args.landmark_dir)
    # split by samples
    indices = list(range(len(ds)))
    train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=42, shuffle=True)
    from torch.utils.data import Subset
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    in_dim = 42  # 21*2
    n_classes = len(ds.class_map)
    model = StaticMLP(input_dim=in_dim, n_classes=n_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, opt, criterion, device)
        acc = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} loss={loss:.4f} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({'model_state': model.state_dict(), 'class_map': ds.class_map}, args.save)
    print("Training done. Best val acc:", best_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--landmark_dir', required=True)
    parser.add_argument('--save', default='models/static_mlp.pth')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
