import torch
from torch.utils.data import Dataset

class SimpleBinaryDataset(Dataset):
    def __init__(self):
        self.data = torch.tensor([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0],
            [1.0, 0.6],
            [9.0, 11.0],
            [8.0, 2.0],
            [10.0, 2.0],
            [9.0, 3.0],
            [8.0, 1.0],
        ])
        self.labels = torch.tensor([0, 0, 1, 1, 0, 1, 1, 1, 1, 1], dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]