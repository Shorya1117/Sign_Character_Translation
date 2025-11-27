# scripts/dataset.py

import os
import numpy as np
import glob
import torch
from torch.utils.data import Dataset


# --------------------------------------------------
#   STATIC LANDMARK DATASET (for A–Z single images)
# --------------------------------------------------
class StaticLandmarkDataset(Dataset):
    def __init__(self, landmark_root, class_map=None, flatten=True):
        """
        Auto-detects structure:
        - landmarks/A
        - landmarks/B
        OR
        - landmarks/static/A
        - landmarks/static/B
        """

        # Detect if inside "landmarks" there is only one folder called "static"
        subfolders = [
            d for d in os.listdir(landmark_root)
            if os.path.isdir(os.path.join(landmark_root, d))
        ]

        # If landmarks/static exists → go inside automatically
        if len(subfolders) == 1 and subfolders[0].lower() == "static":
            landmark_root = os.path.join(landmark_root, "static")

        self.samples = []

        # Read each class folder containing .npy files
        for cls in os.listdir(landmark_root):
            cls_dir = os.path.join(landmark_root, cls)
            if not os.path.isdir(cls_dir):
                continue

            for f in os.listdir(cls_dir):
                if f.endswith(".npy"):
                    self.samples.append((os.path.join(cls_dir, f), cls))

        # Build class names
        self.class_names = sorted(list(set([s[1] for s in self.samples])))

        # Create class → index mapping
        if class_map is None:
            self.class_map = {n: i for i, n in enumerate(self.class_names)}
        else:
            self.class_map = class_map

        self.flatten = flatten

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]

        data = np.load(path, allow_pickle=True).item()
        lm = data["landmarks"]  # shape: (21,3)

        # Use only X,Y
        feat = lm[:, :2]  # (21,2)

        if self.flatten:
            feat = feat.reshape(-1).astype(np.float32)  # 42-dim feature

        label = self.class_map[cls]

        return torch.from_numpy(feat).float(), label


# --------------------------------------------------
#   DYNAMIC SEQUENCE DATASET (for video sequences)
# --------------------------------------------------
class DynamicSequenceDataset(Dataset):
    def __init__(self, landmark_dir, class_map=None, max_len=60):
        self.files = [f for f in glob.glob(os.path.join(landmark_dir, "*.npy"))]
        self.max_len = max_len

        # Extract class names from filenames like: "hello_001.npy"
        self.class_names = sorted(
            list(
                set([os.path.basename(f).split("_")[0] for f in self.files])
            )
        )

        if class_map is None:
            self.class_map = {n: i for i, n in enumerate(self.class_names)}
        else:
            self.class_map = class_map

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        seq = data["landmarks"]  # shape: (T, 21, 3)

        # keep x,y only
        seq = seq[:, :2]

        T = seq.shape[0]

        # Pad or truncate to max_len
        if T >= self.max_len:
            seq = seq[:self.max_len]
        else:
            pad = np.zeros((self.max_len - T, 21, 2), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)

        # Flatten each frame
        seq = seq.reshape(self.max_len, -1).astype(np.float32)  # (max_len,42)

        label_name = data["label"]
        label = self.class_map[label_name]

        return torch.from_numpy(seq).float(), label
