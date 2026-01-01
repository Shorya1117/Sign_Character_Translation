# scripts/models.py
import torch
import torch.nn as nn

class StaticMLP(nn.Module):
    def __init__(self, input_dim=42, n_classes=26, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, n_classes)
        )
    def forward(self, x):
        return self.net(x)
