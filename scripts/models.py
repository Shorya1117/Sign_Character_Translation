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

class DynamicLSTM(nn.Module):
    def __init__(self, input_size=42, hidden_size=128, n_layers=2, n_classes=3, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), n_classes)
    def forward(self, x):
        # x: (B, T, F)
        out, (h, c) = self.lstm(x)
        # use last timestep
        last = out[:, -1, :]
        return self.fc(last)
