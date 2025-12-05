from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from dataset import PaperPairDataModule
from model import LitRankNet

def main():
    dm = PaperPairDataModule("train.csv", "val.csv", batch_size=1024)
    # 自动推断特征维度
    sample = dm.train_ds[0][0]
    input_dim = sample.shape[0]

    model = LitRankNet(input_dim=input_dim, hidden=256, lr=1e-3)

    ckpt = ModelCheckpoint(monitor="val_MAP", mode="max", save_top_k=1,
                           filename="best-{epoch:02d}-{val_MAP:.4f}")
    early = EarlyStopping(monitor="val_MAP", patience=5, mode="max")

    trainer = Trainer(max_epochs=50, gpus=1, precision=16,
                      callbacks=[ckpt, early])
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()