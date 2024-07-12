from config import *
from dataset import HouseXDataset
from model import HouseXModel
from torch.utils.data import DataLoader
import torch
import lightning as L

if __name__ == '__main__':
    drop_detection_path = "detected_drops.json"
    genre_annotation_path = "/Users/ca7ax/housex-v2/project-4-100-clean.json"
    dataset = HouseXDataset(drop_detection_path, genre_annotation_path)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(42))
    model = HouseXModel(extractor_name='vit_b_16')
    trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=model, train_dataloaders=data_loader)