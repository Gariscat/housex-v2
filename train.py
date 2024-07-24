from config import *
from dataset import HouseXDataset
from model import HouseXModel
from torch.utils.data import DataLoader
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(project="housex-v2")

if __name__ == '__main__':
    drop_detection_path = '/root/housex-v2/annotations/detected_drops.json'
    genre_annotation_path = '/root/housex-v2/annotations/partition-1-5.json'
    dataset = HouseXDataset(drop_detection_path, genre_annotation_path)
    torch.save('~/partition-1-5-dataset.pth')
    # dataset = torch.load("proto_dataset.pth")
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(42))
    model = HouseXModel(extractor_name='densenet201')
    trainer = L.Trainer(max_epochs=10, logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=data_loader)