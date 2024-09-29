# colbert_v2/train/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def create_dataloader(texts, labels, batch_size=Config.BATCH_SIZE):
    dataset = TextDataset(texts, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
