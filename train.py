from config import *
from dataset import HouseXDataset, create_splits
from model import HouseXModel
from torch.utils.data import DataLoader, random_split
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from easydict import EasyDict as edict
from copy import deepcopy
from argparse import ArgumentParser

torch_rng = torch.Generator().manual_seed(42)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--extractor_name', type=str, default='resnet18')
    parser.add_argument('--transformer_num_layers', type=int, default=1)
    parser.add_argument('--loss_weight', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=3)
    parser.add_argument('--data_mode', type=str, default='full')
    args = parser.parse_args()
    
    model_config = edict({
        'extractor_name': args.extractor_name,
        'transformer_num_layers': args.transformer_num_layers,
        'loss_weight': args.loss_weight,
        'learning_rate': args.learning_rate,
        'd_model': args.d_model,
        'n_head': args.n_head,
    })
    
    model = HouseXModel(model_config)
    wb_config = deepcopy(model_config)
    wb_config.loss_weight = 'weighted' if wb_config is not None else None
    wb_config['comment'] = 'use-chroma'
    wb_config['batch_size'] = 4
    wb_config['data_mode'] = args.data_mode
    
    train_test_ratio = [0.8, 0.2]
    train_split, test_split = create_splits(
        audio_dirs=['/root/part-1-5/', '/root/part-6-10/'],
        split_ratio=train_test_ratio,
        rng_seed=42,
        mode=wb_config['data_mode']
    )
    
    train_set = HouseXDataset(data_list=train_split, use_chroma=True, audio_standalone_dir='/root/standalone_train/')
    val_set = HouseXDataset(data_list=test_split, use_chroma=True, audio_standalone_dir='/root/standalone_test/')
    torch.save(train_set, '/root/train_set.pth')
    torch.save(val_set, '/root/test_set.pth')
    
    """
    train_set = torch.load('/root/train_set.pth')
    val_set = torch.load('/root/test_set.pth')
    """
    ### train_set, val_set = random_split(dataset, [0.8, 0.2], generator=torch_rng)
    
    class_cnt = sum([y for _, y in train_set])
    for genre, score in zip(ALL_GENRES, class_cnt.numpy().tolist()):
        print(genre, score)
    lw = 1 / class_cnt
    lw /= lw.sum()
    lw = torch.tensor(lw, dtype=torch.float32)

    train_loader = DataLoader(train_set, batch_size=wb_config['batch_size'], shuffle=True, generator=torch_rng)
    val_loader = DataLoader(val_set, batch_size=wb_config['batch_size'], shuffle=False, generator=torch_rng)
    
    wandb_logger = WandbLogger(
        project="housex-v2-grid",
        config=wb_config,
        save_dir='/root'
    )
    
    trainer = L.Trainer(
        max_epochs=10,
        logger=wandb_logger,
        log_every_n_steps=1,
        val_check_interval=0.25,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
