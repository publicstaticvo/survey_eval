import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import RetrievalMAP

class LitRankNet(pl.LightningModule):
    def __init__(self, input_dim, hidden=128, lr=1e-3, margin=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1)   # 输出一个标量分
        )
        self.loss_fn = nn.MarginRankingLoss(margin=margin)
        self.val_map = RetrievalMAP()

    def forward(self, x):
        # x: [B, D] -> [B, 1]
        return self.net(x).squeeze(1)

    def training_step(self, batch, idx):
        x1, x2 = batch          # pos, neg
        s1 = self(x1)
        s2 = self(x2)
        y = torch.ones_like(s1) # 希望 s1 > s2
        loss = self.loss_fn(s1, s2, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, idx):
        x1, x2 = batch
        s1 = self(x1)
        s2 = self(x2)
        y = torch.ones_like(s1)
        loss = self.loss_fn(s1, s2, y)
        self.log("val_loss", loss)
        # 计算 MAP 需要全排序，这里简单把 pos/neg 分数拼一起
        scores = torch.cat([s1, s2]).detach()
        labels = torch.cat([torch.ones_like(s1), torch.zeros_like(s2)]).long()
        self.val_map.update(scores, labels, torch.zeros_like(scores))
        return loss

    def on_validation_epoch_end(self):
        self.log("val_MAP", self.val_map.compute(), prog_bar=True)
        self.val_map.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)