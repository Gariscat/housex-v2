from config import *
from dataset import HouseXDataset
from model import HouseXModel
from torch.utils.data import DataLoader, random_split
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger

torch_rng = torch.Generator().manual_seed(42)
wandb_logger = WandbLogger(project="housex-v2")

if __name__ == '__main__':
    drop_detection_path = '/root/housex-v2/annotations/detected_drops.json'
    genre_annotation_path = '/root/housex-v2/annotations/partition-1-5.json'
    dataset = HouseXDataset(drop_detection_path, genre_annotation_path)
    torch.save(dataset, '/root/partition-1-5-dataset.pth')
    """
    dataset = torch.load("/root/partition-1-5-dataset.pth")
    """
    train_set, val_set = random_split(dataset, [0.8, 0.2], generator=torch_rng)
    
    class_cnt = sum([y for _, y in train_set])
    for genre, score in zip(ALL_GENRES, class_cnt.numpy().tolist()):
        print(genre, score)
    lw = 1 / class_cnt
    lw /= lw.sum()
    lw = torch.tensor(lw, dtype=torch.float32)
    
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, generator=torch_rng)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, generator=torch_rng)
    model = HouseXModel(extractor_name='densenet201', loss_weight=lw)
    trainer = L.Trainer(max_epochs=10, logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
