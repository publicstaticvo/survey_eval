import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class PaperPairDataset(Dataset):
    """
    内存版：一次性把所有特征读进 RAM；大数据请改用 HDF5。
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.topic2group = dict(iter(df.groupby("topic_id")))
        self.pairs = []          # [(topic_id, idx_i, idx_j)]
        for tid, g in self.topic2group.items():
            pos_idx = g.index[g.label == 1].tolist()
            neg_idx = g.index[g.label == 0].tolist()
            for i in pos_idx:
                for j in neg_idx:
                    self.pairs.append((tid, i, j))
        self.df = df

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        tid, i, j = self.pairs[item]
        g = self.topic2group[tid]
        x1 = g.loc[i, "feature0":].values.astype("float32")
        x2 = g.loc[j, "feature0":].values.astype("float32")
        return torch.tensor(x1), torch.tensor(x2)   # (pos, neg)

class PaperPairDataModule(pl.LightningDataModule):
    def __init__(self, train_csv, val_csv, batch_size=512, num_workers=4):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv   = val_csv
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = PaperPairDataset(self.train_csv)
        self.val_ds   = PaperPairDataset(self.val_csv)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          persistent_workers=True)