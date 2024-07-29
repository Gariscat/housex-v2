from config import *
from dataset import HouseXDataset
from model import HouseXModel
from torch.utils.data import DataLoader, random_split
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from easydict import EasyDict as edict
from copy import deepcopy

torch_rng = torch.Generator().manual_seed(42)

if __name__ == '__main__':
    """train_set = HouseXDataset(
        audio_dir='/root/part-1-5/',
        drop_detection_path='/root/part-1-5/detected_drops.json',
        genre_annotation_path='/root/part-1-5/partition-1-5-refined.json'
    )
    torch.save(train_set, '/root/partition-1-5-dataset.pth')
    val_set = HouseXDataset(
        audio_dir='/root/part-6-10/',
        drop_detection_path='/root/part-6-10/detected_drops.json',
        genre_annotation_path='/root/part-6-10/partition-6-10-refined.json'
    )
    torch.save(val_set, '/root/partition-6-10-dataset.pth')
    """
    train_set = torch.load("/root/partition-1-5-dataset.pth")
    val_set = torch.load("/root/partition-6-10-dataset.pth")
    
    ### train_set, val_set = random_split(dataset, [0.8, 0.2], generator=torch_rng)
    
    class_cnt = sum([y for _, y in train_set])
    for genre, score in zip(ALL_GENRES, class_cnt.numpy().tolist()):
        print(genre, score)
    lw = 1 / class_cnt
    lw /= lw.sum()
    lw = torch.tensor(lw, dtype=torch.float32)
    
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, generator=torch_rng)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, generator=torch_rng)
    
    model_config = edict({
        'extractor_name': 'densenet201',
        'transformer_num_layers': 3,
        'loss_weight': None,
        'learning_rate': 1e-4,
    })
    
    model = HouseXModel(model_config)
    wb_config = deepcopy(model_config)
    wb_config.loss_weight = 'weighted' if wb_config is not None else None
    wb_config['comment'] = 'train/val: 1-5/6-10'
    
    wandb_logger = WandbLogger(
        project="housex-v2-1-5",
        config=wb_config,
        save_dir='/root'
    )
    
    trainer = L.Trainer(
        max_epochs=30,
        logger=wandb_logger,
        log_every_n_steps=1,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
